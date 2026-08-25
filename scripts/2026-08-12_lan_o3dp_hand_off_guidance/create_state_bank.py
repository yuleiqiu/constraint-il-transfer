"""Create fresh fixed PickPlace initial states without running a policy."""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


SCRIPT_DIR = Path(__file__).resolve().parent
PAIRED_EVAL_PATH = (
    SCRIPT_DIR.parent / "2026-08-05_lan_o3dp_paper_aligned" / "paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("lan_paper_paired_eval", PAIRED_EVAL_PATH)
PAIRED_EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIRED_EVAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", default=PAIRED_EVAL.DEFAULT_ENVS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.out_dir))
    args.out_dir.mkdir(parents=True)
    bank_path = args.out_dir / "initial_state_bank.pkl"
    bank = {"format_version": 1, "states": {}}
    checkpoint = FileUtils.load_dict_from_checkpoint(str(args.agent.resolve()))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)
    for env_name in args.envs:
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint,
            env_name=env_name,
            render=False,
            render_offscreen=True,
        )
        try:
            for episode in range(args.n_rollouts):
                layout_seed = args.seed * 1000 + episode
                random.seed(layout_seed)
                np.random.seed(layout_seed)
                env.reset()
                state = env.get_state()
                state_key = "{}:{}:{}".format(env_name, args.seed, episode)
                bank["states"][state_key] = {
                    "layout_seed": layout_seed,
                    "state": state,
                    "state_hash": PAIRED_EVAL.state_hash(state),
                }
                PAIRED_EVAL.save_state_bank(bank_path, bank)
        finally:
            PAIRED_EVAL.rollout_common.raw_env_from_wrapper(env).close()
    manifest = {
        "agent": str(args.agent.resolve()),
        "envs": args.envs,
        "seed": args.seed,
        "n_rollouts": args.n_rollouts,
        "state_bank": str(bank_path.resolve()),
        "state_count": len(bank["states"]),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote {} states to {}".format(len(bank["states"]), bank_path))


if __name__ == "__main__":
    main()
