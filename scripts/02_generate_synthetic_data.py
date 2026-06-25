"""Generate and split deterministic synthetic trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.data_split import split_trajectories
from wmagentattack.io_utils import write_jsonl
from wmagentattack.synthetic_generator import (
    flatten_steps,
    generate_synthetic_trajectories,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data" / "splits"
    )
    args = parser.parse_args()
    trajectories = generate_synthetic_trajectories(args.count, args.seed)
    splits = split_trajectories(trajectories, seed=args.seed)
    stats = {}
    for name, records in splits.items():
        write_jsonl(args.out_dir / f"{name}_trajectories.jsonl", records)
        steps = flatten_steps(records)
        write_jsonl(args.out_dir / f"{name}_steps.jsonl", steps)
        stats[name] = {
            "trajectories": len(records),
            "steps": len(steps),
            "attack_success_rate": sum(
                record.final_attack_success for record in records
            )
            / len(records),
        }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
