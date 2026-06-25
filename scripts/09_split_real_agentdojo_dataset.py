"""Create leakage-resistant train/val/test splits for real AgentDojo traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl, write_jsonl
from wmagentattack.schema import TrajectoryRecord


def _bucketed_split(
    trajectories: list[TrajectoryRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[TrajectoryRecord]]:
    """Split by trajectory while roughly preserving attack-success labels."""

    buckets: dict[tuple[str, bool], list[TrajectoryRecord]] = {}
    for trajectory in trajectories:
        key = (trajectory.domain, trajectory.final_attack_success)
        buckets.setdefault(key, []).append(trajectory)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        train_end = int(len(bucket) * train_ratio)
        val_end = train_end + int(len(bucket) * val_ratio)
        # Tiny positive buckets need at least one held-out example when possible.
        if len(bucket) >= 3 and train_end == len(bucket):
            train_end = len(bucket) - 1
        if len(bucket) >= 5 and val_end == train_end:
            val_end = train_end + 1
        splits["train"].extend(bucket[:train_end])
        splits["val"].extend(bucket[train_end:val_end])
        splits["test"].extend(bucket[val_end:])

    for split in splits.values():
        rng.shuffle(split)
    return splits


def _steps(trajectories: list[TrajectoryRecord]):
    return [step for trajectory in trajectories for step in trajectory.steps]


def _summary(splits: dict[str, list[TrajectoryRecord]]) -> dict:
    payload = {}
    for name, trajectories in splits.items():
        steps = _steps(trajectories)
        payload[name] = {
            "trajectories": len(trajectories),
            "steps": len(steps),
            "positive_attack_trajectories": sum(
                t.final_attack_success for t in trajectories
            ),
            "negative_attack_or_clean_trajectories": sum(
                not t.final_attack_success for t in trajectories
            ),
            "domains": dict(Counter(t.domain for t in trajectories)),
        }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "real_agentdojo" / "trajectories.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "real_agentdojo" / "splits",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    trajectories = [
        TrajectoryRecord.model_validate(row) for row in read_jsonl(args.input)
    ]
    splits = _bucketed_split(
        trajectories,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_trajectories in splits.items():
        write_jsonl(args.out_dir / f"{name}_trajectories.jsonl", split_trajectories)
        write_jsonl(args.out_dir / f"{name}_steps.jsonl", _steps(split_trajectories))

    summary = {
        "scope": "AgentDojo sandbox only",
        "input": str(args.input.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "seed": args.seed,
        "splits": _summary(splits),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
