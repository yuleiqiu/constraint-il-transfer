#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toy_recovery.data import read_evaluation_records
from toy_recovery.evaluation import rapid_gate, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--fixed-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_evaluation_records(args.episodes)
    fixed_state = json.loads(args.fixed_state.read_text())
    decision = rapid_gate(records, fixed_state["ranking"])
    payload = {
        "schema": "toy-recovery-rapid-screen-v1",
        "summary": summarize_records(records),
        "recoverability_ranking": fixed_state["ranking"],
        "achieved_recoverability_coverage": fixed_state["achieved_coverage"],
        "gate": decision,
        "reguide_implemented": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not decision["passed"]:
        raise SystemExit(decision["decision"])


if __name__ == "__main__":
    main()
