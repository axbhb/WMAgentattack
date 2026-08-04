"""Create train/val/test splits for real AgentDojo traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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
    """Legacy trajectory split; user tasks may occur in multiple splits."""

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


def _task_count_targets(
    count: int, train_ratio: float, val_ratio: float
) -> tuple[int, int]:
    if count < 3:
        raise ValueError("Grouped splitting requires at least three tasks per domain")
    train = max(1, int(round(count * train_ratio)))
    val = max(1, int(round(count * val_ratio)))
    if train + val >= count:
        overflow = train + val - count + 1
        train = max(1, train - overflow)
    return train, val


def _grouped_user_task_split(
    trajectories: list[TrajectoryRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    search_iterations: int = 5000,
) -> dict[str, list[TrajectoryRecord]]:
    """Split complete ``(domain, task_id)`` groups with balance search."""

    if search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    grouped: dict[tuple[str, str], list[TrajectoryRecord]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[(trajectory.domain, trajectory.task_id)].append(trajectory)
    tasks_by_domain: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in grouped:
        tasks_by_domain[task[0]].append(task)
    targets = {
        domain: _task_count_targets(len(tasks), train_ratio, val_ratio)
        for domain, tasks in tasks_by_domain.items()
    }
    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    if split_ratios["test"] <= 0:
        raise ValueError("train_ratio + val_ratio must be below one")

    def candidate_score(candidate: dict[str, list[TrajectoryRecord]]) -> float:
        score = 0.0
        for domain in tasks_by_domain:
            domain_rows = [
                row for row in trajectories if row.domain == domain
            ]
            totals = (
                len(domain_rows),
                sum(bool(row.final_attack_success) for row in domain_rows),
                sum(bool(row.final_task_success) for row in domain_rows),
            )
            for split_name, target_ratio in split_ratios.items():
                rows = [
                    row for row in candidate[split_name] if row.domain == domain
                ]
                values = (
                    len(rows),
                    sum(bool(row.final_attack_success) for row in rows),
                    sum(bool(row.final_task_success) for row in rows),
                )
                for value, total in zip(values, totals, strict=True):
                    if total:
                        score += (value / total - target_ratio) ** 2
        totals = (
            len(trajectories),
            sum(bool(row.final_attack_success) for row in trajectories),
            sum(bool(row.final_task_success) for row in trajectories),
        )
        for split_name, target_ratio in split_ratios.items():
            rows = candidate[split_name]
            values = (
                len(rows),
                sum(bool(row.final_attack_success) for row in rows),
                sum(bool(row.final_task_success) for row in rows),
            )
            for value, total in zip(values, totals, strict=True):
                if total:
                    score += (value / total - target_ratio) ** 2
        return score

    rng = random.Random(seed)
    best = None
    best_score = float("inf")
    for _ in range(search_iterations):
        candidate = {"train": [], "val": [], "test": []}
        for domain, domain_tasks in sorted(tasks_by_domain.items()):
            tasks = sorted(domain_tasks)
            rng.shuffle(tasks)
            train_end, val_count = targets[domain]
            val_end = train_end + val_count
            assignment = {
                "train": tasks[:train_end],
                "val": tasks[train_end:val_end],
                "test": tasks[val_end:],
            }
            for split_name, split_tasks in assignment.items():
                for task in split_tasks:
                    candidate[split_name].extend(grouped[task])
        score = candidate_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
    if best is None:
        raise AssertionError("Grouped split search produced no candidate")
    for split in best.values():
        rng.shuffle(split)
    return best


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
            "user_tasks": len({(t.domain, t.task_id) for t in trajectories}),
        }
    task_sets = {
        name: {(t.domain, t.task_id) for t in trajectories}
        for name, trajectories in splits.items()
    }
    payload["task_overlap"] = {
        "train_val": len(task_sets["train"] & task_sets["val"]),
        "train_test": len(task_sets["train"] & task_sets["test"]),
        "val_test": len(task_sets["val"] & task_sets["test"]),
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
    parser.add_argument(
        "--split-unit",
        choices=("user_task", "trajectory"),
        default="user_task",
        help=(
            "user_task keeps every (domain, task_id) in one split; trajectory "
            "reproduces the legacy label-bucketed split"
        ),
    )
    parser.add_argument("--group-search-iterations", type=int, default=5000)
    args = parser.parse_args()

    trajectories = [
        TrajectoryRecord.model_validate(row) for row in read_jsonl(args.input)
    ]
    if args.split_unit == "user_task":
        splits = _grouped_user_task_split(
            trajectories,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            search_iterations=args.group_search_iterations,
        )
    else:
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
        "split_unit": args.split_unit,
        "group_search_iterations": (
            args.group_search_iterations
            if args.split_unit == "user_task"
            else None
        ),
        "splits": _summary(splits),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
