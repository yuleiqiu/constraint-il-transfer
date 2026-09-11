"""Corrected full-episode paired evaluation, with in-run videos and audit data."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reeval_paired', ROOT / 'scripts/2026-08-05_lan_o3dp_paper_aligned/paired_eval.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
C = P.rollout_common


def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AuditedEnv:
    def __init__(self, env):
        self.env = env
        self.raw = C.raw_env_from_wrapper(env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_to(self, state):
        self.raw.rng = np.random.default_rng(self.layout_seed)
        obs = self.env.reset_to(state)
        self.reset_audit = {
            'goal_before': self.raw.goal_xy.copy(),
            'obstacle_before': self.raw.obstacle_xy.copy(),
            'physical_goal': self.raw.sim.model.body_pos[self.raw.goal_tray_body_id, :2].copy(),
            'physical_obstacle': self.raw.sim.model.body_pos[self.raw.obstacle_body_id, :2].copy(),
        }
        self.raw.goal_xy = self.reset_audit['physical_goal'].copy()
        self.raw.obstacle_xy = self.reset_audit['physical_obstacle'].copy()
        self.reset_audit['restored_state_hash'] = P.state_hash(self.env.get_state())
        self.reset_audit['initial_obs_hashes'] = {k: sha_array(v) for k, v in obs.items()}
        self.initial_obs = {k: np.asarray(v).copy() for k, v in obs.items()}
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_done = False
        return obs

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        result = self.env.step(action)
        self.last_done = bool(result[2])
        return result


class Evaluation:
    def __init__(self, cli):
        self.cli = cli
        self.bank = P.load_state_bank(cli.state_bank)
        self.policy, checkpoint = P.paper_guided_policy_from_checkpoint(
            ckpt_path=str(cli.checkpoint), device=torch.device('cuda'))
        env, _ = P.FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint, env_name='PickPlaceCanTabletopRaisedMilk',
            render=False, render_offscreen=True, verbose=False)
        self.env = AuditedEnv(env)
        self.raw = self.env.raw
        self.args = SimpleNamespace(horizon=600, terminate_on_success=True,
            terminate_on_collision=False, target_object_name='Can', xy_only=True,
            disable_hand_collision=False, obstacle_cost_type='huber_hinge',
            obstacle_cost_smoothing=.01, obstacle_cost_smoothing_ratio=None,
            tangent_ratio=0., guidance_phase='target_lifted')

    def run(self, episode, condition, folder):
        folder.mkdir(parents=True, exist_ok=False)
        record = self.bank['states'][f'PickPlaceCanTabletopRaisedMilk:828:{episode}']
        assert record['layout_seed'] == 828000 + episode
        assert P.state_hash(record['state']) == record['state_hash']
        self.env.layout_seed = record['layout_seed']
        # Materialize the episode XML first; then the sampler performs its seeded reset.
        seed(828000000 + episode)
        self.env.reset_to(record['state'])
        trace, states, actions, chunks, chunk_obs, clouds = [], [], [], [], [], []
        self.chunk_initial_obs = None
        video_steps = []
        start = time.monotonic()
        writer = imageio.get_writer(str(folder / 'rollout.partial.mp4'), fps=10,
                                   codec='libx264', quality=7)

        def frame(env, obs, step, success, collision):
            can = self.raw.sim.data.body_xpos[self.raw.can_body_id].copy()
            eef = C.current_raw_eef_position(self.raw).copy()
            distances = [float(np.linalg.norm(self.raw.sim.data.site_xpos[
                self.raw.robots[0].eef_site_id[arm]] - can)) for arm in self.raw.robots[0].arms]
            physical_goal = self.raw.sim.model.body_pos[self.raw.goal_tray_body_id, :2].copy()
            assert np.array_equal(self.raw.goal_xy, physical_goal)
            half = self.raw.goal_tray_inner_size / 2.
            xy_ok = bool(np.all(np.abs(can[:2] - physical_goal) < half))
            z_ok = bool(self.raw.table_offset[2] < can[2] < self.raw.table_offset[2] + .1)
            retreat_ok = bool(1. - np.tanh(10. * min(distances)) < .6)
            assert bool(C.task_success(self.env)) == (xy_ok and z_ok and retreat_ok)
            trace.append({'step': step, 'can': can, 'eef': eef,
                'goal_xy': physical_goal, 'goal_half_size': half,
                'eef_can_distance': min(distances), 'goal_xy_ok': xy_ok,
                'goal_z_ok': z_ok, 'retreat_ok': retreat_ok,
                'success': bool(success), 'collision_any': bool(collision),
                'done': self.env.last_done,
                'obs_hashes': {k: sha_array(v) for k, v in obs.items()}})
            states.append(self.raw.sim.get_state().flatten().copy())
            if step:
                actions.append(self.env.last_action.copy())
            if step % 2 == 0 or success or self.env.last_done or step == 600:
                pixels = self.env.render(mode='rgb_array', camera_name='agentview', height=384, width=384)
                im = Image.new('RGB', (384, 432), 'black')
                im.paste(Image.fromarray(np.asarray(pixels, dtype=np.uint8)), (0, 48))
                draw = ImageDraw.Draw(im)
                draw.text((8, 5), f'Layout {episode:02d} | {condition} | step {step}/600', fill='white')
                draw.text((8, 25), f'Task {int(success)} | Contact {int(collision)} | XY/Z/retreat {int(xy_ok)}/{int(z_ok)}/{int(retreat_ok)}', fill='white')
                writer.append_data(np.asarray(im))
                video_steps.append(step)

        def chunk(env, obs, step, current, cloud, context):
            chunks.append({'step': step, 'guidance_context': context is not None,
                           'obstacle_points': len(cloud), 'cloud_hash': sha_array(cloud),
                           'torch_rng_hash': sha_array(torch.get_rng_state().numpy()),
                           'cuda_rng_hash': sha_array(torch.cuda.get_rng_state().cpu().numpy())})
            chunk_obs.append({k: np.asarray(v).copy() for k, v in obs.items()})
            clouds.append(np.asarray(cloud).copy())

        try:
            metrics = P.rollout_condition(self.policy, self.env, record['state'], condition,
                828000000 + episode, self.args,
                guidance_scale=45. if condition == 'guided' else 0.,
                safety_distance=.08 if condition == 'guided' else None,
                frame_callback=frame, chunk_callback=chunk)
        finally:
            writer.close()
        assert self.env.reset_audit['restored_state_hash'] == record['state_hash']
        assert len(actions) == metrics['horizon'] and len(trace) == len(actions) + 1
        dump(folder / 'trace.json', trace)
        dump(folder / 'chunks.json', chunks)
        dump(folder / 'reset.json', self.env.reset_audit)
        np.savez_compressed(folder / 'trajectory.npz', actions=np.asarray(actions), states=np.asarray(states),
                            video_steps=np.asarray(video_steps))
        obs_data = {'initial_' + k: v for k, v in self.env.initial_obs.items()}
        if chunk_obs:
            obs_data.update({'chunks_' + k: np.stack([o[k] for o in chunk_obs]) for k in chunk_obs[0]})
        for i, cloud in enumerate(clouds):
            obs_data[f'obstacle_cloud_{i:03d}'] = cloud
        np.savez_compressed(folder / 'observations.npz', **obs_data)
        (folder / 'rollout.partial.mp4').replace(folder / 'rollout.mp4')
        result = {**metrics, 'episode': episode, 'policy_seed': 828000000 + episode,
            'initial_state_hash': record['state_hash'], 'layout_seed': record['layout_seed'],
            'termination': 'success' if metrics['success'] else ('horizon' if metrics['horizon'] == 600 else 'env_done'),
            'wall_seconds': time.monotonic() - start, 'video_frames': len(video_steps),
            'folder': str(folder.relative_to(self.cli.out)), 'reset_audit': self.env.reset_audit}
        dump(folder / 'result.json', result)
        print(json.dumps(safe({'completed': result['folder'], 'success': metrics['success'],
            'collision': metrics['collision_any'], 'steps': metrics['horizon'], 'seconds': result['wall_seconds']})), flush=True)
        return result


def compare(left, right, same_condition):
    a, b = json.loads((left/'result.json').read_text()), json.loads((right/'result.json').read_text())
    assert a['initial_state_hash'] == b['initial_state_hash']
    assert a['reset_audit']['initial_obs_hashes'] == b['reset_audit']['initial_obs_hashes'], 'initial observations differ'
    if same_condition:
        assert json.loads((left/'trace.json').read_text()) == json.loads((right/'trace.json').read_text()), 'repeat full trace differs'
        assert json.loads((left/'chunks.json').read_text()) == json.loads((right/'chunks.json').read_text()), 'repeat chunk inputs or RNG differs'
        with np.load(left/'trajectory.npz') as x, np.load(right/'trajectory.npz') as y:
            for k in x.files:
                assert np.array_equal(x[k], y[k]), f'repeat {k} differs'
        for k in ('success', 'safe_success', 'collision_any', 'horizon'):
            assert a[k] == b[k]
    return {'initial_observations_equal': True, 'same_condition_full_repeat_equal': bool(same_condition)}


def main():
    defaults = json.loads((ROOT/'outputs/tabletop_wide_support/raised_milk_paired_seed828_n50/baseline/manifest.json').read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=Path(defaults['checkpoint']))
    ap.add_argument('--state-bank', type=Path, default=Path(defaults['state_bank']))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--stop', type=int, required=True)
    cli = ap.parse_args()
    cli.out = cli.out.resolve()
    assert 0 <= cli.start < cli.stop <= 50
    shard = cli.out / f'shard_{cli.start:02d}_{cli.stop-1:02d}'
    shard.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    manifest = {**vars(cli), 'checkpoint_sha256': sha_file(cli.checkpoint),
        'state_bank_sha256': sha_file(cli.state_bank), 'script_sha256': sha_file(Path(__file__)),
        'paired_sampler_sha256': sha_file(Path(P.__file__)),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'egl_device': os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'horizon': 600, 'guidance': {'rho':45, 'Q':.08, 'smoothing':.01, 'phase':'target_lifted', 'xy_only':True, 'tangent_ratio':0},
        'git_heads': {name: subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
            for name,path in [('root',ROOT),('robomimic',ROOT/'third_party/robomimic'),('robosuite',ROOT/'third_party/robosuite')]}}
    dump(shard/'manifest.json', manifest)
    ev = Evaluation(cli)
    try:
        episode = cli.start
        for rep in range(2):
            for cond in ('baseline', 'guided'):
                ev.run(episode, cond, shard/'preflight'/f'r{rep}_{cond}')
        checks = {cond: compare(shard/'preflight'/f'r0_{cond}', shard/'preflight'/f'r1_{cond}', True) for cond in ('baseline','guided')}
        checks['pair'] = compare(shard/'preflight/r0_baseline', shard/'preflight/r0_guided', False)
        dump(shard/'preflight_pass.json', checks)
        print('Preflight passed; awaiting shared START_MAIN marker.', flush=True)
        while not (cli.out/'START_MAIN').exists():
            time.sleep(2)
        for episode in range(cli.start, cli.stop):
            folders = {cond: shard/'episodes'/f'ep{episode:02d}'/cond for cond in ('baseline','guided')}
            for cond, folder in folders.items():
                ev.run(episode, cond, folder)
            check = compare(folders['baseline'], folders['guided'], False)
            if episode == cli.start:
                for cond in folders:
                    compare(shard/'preflight'/f'r0_{cond}', folders[cond], True)
            dump(shard/'episodes'/f'ep{episode:02d}'/'pair_audit.json', check)
        dump(shard/'complete.json', {'episodes':cli.stop-cli.start, 'rollouts':2*(cli.stop-cli.start)})
    finally:
        ev.raw.close()


if __name__ == '__main__':
    main()
