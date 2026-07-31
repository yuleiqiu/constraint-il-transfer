from __future__ import annotations

import torch

from toy_recovery.data import EvaluationRecord
from toy_recovery.evaluation import (
    GEOMETRY_METHOD,
    LATENT_METHOD,
    SOURCE_METHOD,
    exact_paired_pvalue,
    rapid_gate,
    summarize_records,
)
from toy_recovery.training import DynamicsBatch, dynamics_validation


class ActionAwarePredictor(torch.nn.Module):
    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return action[:, 0, :1].repeat(1, latent.shape[1])


def test_action_shuffle_validation_detects_action_use() -> None:
    batch_size, latent_dim = 32, 4
    action = torch.zeros(batch_size, 16, 2)
    action[:, 0, 0] = torch.linspace(-1.0, 1.0, batch_size)
    target = action[:, 0, :1].repeat(1, latent_dim)
    batch = DynamicsBatch(
        latent=torch.zeros(batch_size, latent_dim),
        action=action,
        future_latent=target,
    )
    result = dynamics_validation(ActionAwarePredictor(), [batch])
    assert result["passed"]
    assert result["action_shuffle_ratio"] >= 1.2


def test_latent_dynamics_overfits_one_batch() -> None:
    from toy_recovery.models import LatentDynamics

    torch.manual_seed(2)
    model = LatentDynamics(latent_dim=4, hidden_dim=32)
    latent = torch.randn(8, 4)
    action = torch.randn(8, 15, 2)
    target = torch.cat(
        (action[:, 0], action[:, 1]), dim=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        initial = torch.mean((model(latent, action) - target) ** 2).item()
    for _ in range(120):
        loss = torch.mean((model(latent, action) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.1


def _record(method: str, seed: int, success: bool) -> EvaluationRecord:
    return EvaluationRecord(
        method=method,
        difficulty=1,
        layout_seed=seed,
        layout_id=str(seed),
        policy_seed=0,
        success=success,
        collision=not success,
        timeout=False,
        steps=10,
        path_length=0.5,
        minimum_clearance=0.01,
        guidance_activations=int(method != SOURCE_METHOD),
        inference_seconds=0.123,
    )


def test_paired_summary_is_deterministic_without_runtime() -> None:
    records = [
        _record(SOURCE_METHOD, 1, False),
        _record(GEOMETRY_METHOD, 1, True),
        _record(SOURCE_METHOD, 2, True),
        _record(GEOMETRY_METHOD, 2, False),
    ]
    first = summarize_records(records, include_runtime=False)
    second = summarize_records(list(reversed(records)), include_runtime=False)
    assert first == second
    paired = first["paired"][f"{GEOMETRY_METHOD}/D1"]
    assert paired["failure_to_success"] == 1
    assert paired["success_to_failure"] == 1
    assert exact_paired_pvalue(1, 1) == 1.0


def test_rapid_gate_requires_exact_d0_parity() -> None:
    records = []
    for method in (SOURCE_METHOD, GEOMETRY_METHOD, LATENT_METHOD):
        records.append(
            EvaluationRecord(
                method=method,
                difficulty=0,
                layout_seed=0,
                layout_id="d0",
                policy_seed=0,
                success=True,
                collision=False,
                timeout=False,
                steps=10,
                path_length=0.5,
                minimum_clearance=None,
                guidance_activations=0,
            )
        )
    for seed in range(100):
        records.append(_record(SOURCE_METHOD, seed, False))
        records.append(_record(GEOMETRY_METHOD, seed, seed < 10))
        records.append(_record(LATENT_METHOD, seed, False))
    ranking = {
        "geometry": {
            "auroc": 0.7,
            "top_k_precision": 0.3,
            "random_precision": 0.1,
        },
        "global_latent": {
            "auroc": 0.5,
            "top_k_precision": 0.1,
            "random_precision": 0.1,
        },
    }
    assert rapid_gate(records, ranking)["passed"]
    records[1] = EvaluationRecord(
        **{
            **records[1].__dict__,
            "success": False,
            "timeout": True,
        }
    )
    assert not rapid_gate(records, ranking)["d0_retention_passed"]
