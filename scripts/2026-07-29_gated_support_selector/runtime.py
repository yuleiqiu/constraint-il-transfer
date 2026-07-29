"""Runtime sampling adapter for source-default gated corrective selection."""

import numpy as np

from selector import (
    SelectorConfig,
    chunk_geometry,
    point_clearance_xy,
    select_candidate,
    should_trigger,
)


def _unguided_context(common, policy, obs, centers, radii):
    return common.make_guidance_context(
        policy,
        obs,
        centers,
        radii,
        scale=0.0,
        clearance_margin=0.0,
    )


def _source_normalize(raw_chunks, context):
    raw = np.asarray(raw_chunks, dtype=np.float64)
    scale = np.asarray(context.action_scale, dtype=np.float64).reshape(-1)
    offset = np.asarray(context.action_offset, dtype=np.float64).reshape(-1)
    if np.any(np.abs(scale) < 1e-12):
        raise ValueError("Source action normalization contains a zero scale")
    return (raw - offset.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)


def choose_action_chunk(
    *,
    common,
    source_policy,
    corrective_policy,
    obs,
    raw_env,
    object_records,
    target_object_name,
    nominal_seed,
    candidate_seeds,
    target_contact_seen,
    obstacle_contact_seen,
    config=SelectorConfig(),
):
    """Return a source chunk or the least-deviating eligible corrective sample."""

    centers, radii, obstacle_names = common.oracle_obstacle_geometry(
        raw_env, object_records, target_object_name
    )
    target_position = common.target_object_position(
        raw_env, object_records, target_object_name
    )
    source_obs = common.policy_observation(obs, source_policy)
    start_position = common.current_eef_position(source_obs)
    source_context = _unguided_context(
        common, source_policy, source_obs, centers, radii
    )
    source_queue_length = len(source_policy.policy.action_queue)
    source_normalized, source_raw, _ = common.sample_action_chunk(
        source_policy,
        source_obs,
        source_context,
        noise_seed=nominal_seed,
    )
    if len(source_policy.policy.action_queue) != source_queue_length:
        raise RuntimeError("Source sampling mutated its rollout action queue")

    nominal = chunk_geometry(
        start_position, source_raw, target_position, centers, radii
    )
    start_clearance = point_clearance_xy(start_position, centers, radii)
    triggered, suppression_reasons = should_trigger(
        nominal_progress_m=nominal["progress_m"],
        nominal_clearance_m=nominal["clearance_m"],
        start_clearance_m=start_clearance,
        target_contact_seen=target_contact_seen,
        obstacle_contact_seen=obstacle_contact_seen,
        obstacle_count=len(radii),
        config=config,
    )
    diagnostic = {
        "mode": "source",
        "nominal_seed": int(nominal_seed),
        "candidate_seeds": [],
        "obstacle_names": obstacle_names,
        "start_clearance_m": start_clearance,
        "nominal_progress_m": nominal["progress_m"],
        "nominal_clearance_m": nominal["clearance_m"],
        "candidate_required_progress_m": (
            nominal["progress_m"] - config.progress_tolerance_m
            if config.progress_rule == "source_relative"
            else -config.progress_tolerance_m
        ),
        "triggered": triggered,
        "suppression_reasons": suppression_reasons,
        "eligible_candidate_count": 0,
        "selected_candidate_index": None,
        "selected_candidate_seed": None,
        "selected_progress_m": nominal["progress_m"],
        "selected_clearance_m": nominal["clearance_m"],
        "selected_normalized_rms_deviation": 0.0,
    }
    if not triggered:
        return source_raw, diagnostic

    corrective_obs = common.policy_observation(obs, corrective_policy)
    corrective_context = _unguided_context(
        common, corrective_policy, corrective_obs, centers, radii
    )
    corrective_queue_length = len(corrective_policy.policy.action_queue)
    candidate_raw = []
    for seed in candidate_seeds:
        _, raw, _ = common.sample_action_chunk(
            corrective_policy,
            corrective_obs,
            corrective_context,
            noise_seed=seed,
        )
        candidate_raw.append(raw)
    if len(corrective_policy.policy.action_queue) != corrective_queue_length:
        raise RuntimeError("Corrective sampling mutated its rollout action queue")

    candidate_raw = np.asarray(candidate_raw, dtype=np.float64)
    candidate_source_normalized = _source_normalize(
        candidate_raw, source_context
    )
    selected_index, candidate_rows = select_candidate(
        nominal_progress_m=nominal["progress_m"],
        nominal_normalized_chunk=source_normalized,
        candidate_raw_chunks=candidate_raw,
        candidate_normalized_chunks=candidate_source_normalized,
        start_position=start_position,
        target_position=target_position,
        centers=centers,
        radii=radii,
        config=config,
    )
    diagnostic["candidate_seeds"] = [int(seed) for seed in candidate_seeds]
    diagnostic["eligible_candidate_count"] = int(
        sum(row["eligible"] for row in candidate_rows)
    )
    diagnostic["candidate_metrics"] = candidate_rows
    if selected_index is None:
        diagnostic["suppression_reasons"] = ["no_eligible_corrective_candidate"]
        return source_raw, diagnostic

    selected = candidate_rows[selected_index]
    diagnostic.update(
        {
            "mode": "corrective",
            "selected_candidate_index": selected_index,
            "selected_candidate_seed": int(candidate_seeds[selected_index]),
            "selected_progress_m": selected["progress_m"],
            "selected_clearance_m": selected["clearance_m"],
            "selected_normalized_rms_deviation": selected[
                "normalized_rms_deviation"
            ],
        }
    )
    return candidate_raw[selected_index].astype(np.float32), diagnostic
