"""Create suite-balanced outer folds for leakage-safe Dreamer OOF features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["domain"]), str(row["task_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _outer_assignments(
    tasks: set[tuple[str, str]], *, folds: int, seed: int
) -> dict[tuple[str, str], int]:
    by_suite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in sorted(tasks):
        by_suite[task[0]].append(task)
    assignments = {}
    for suite, suite_tasks in sorted(by_suite.items()):
        shuffled = suite_tasks[:]
        random.Random(f"{seed}|{suite}").shuffle(shuffled)
        for index, task in enumerate(shuffled):
            assignments[task] = index % folds
    return assignments


def _inner_validation_tasks(
    development: set[tuple[str, str]],
    *,
    fold: int,
    seed: int,
    per_suite: int,
) -> set[tuple[str, str]]:
    by_suite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in sorted(development):
        by_suite[task[0]].append(task)
    selected = set()
    for suite, suite_tasks in sorted(by_suite.items()):
        ordered = sorted(
            suite_tasks,
            key=lambda task: hashlib.sha256(
                f"{seed}|{fold}|{suite}|{task[1]}".encode()
            ).hexdigest(),
        )
        if len(ordered) <= per_suite:
            raise ValueError(f"Too few development tasks in {suite}")
        selected.update(ordered[:per_suite])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--inner-val-tasks-per-suite", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    steps = _read_jsonl(args.steps)
    trajectories = _read_jsonl(args.trajectories)
    step_tasks = {_task_key(row) for row in steps}
    trajectory_tasks = {_task_key(row) for row in trajectories}
    if step_tasks != trajectory_tasks:
        raise ValueError("Step and trajectory task sets differ")
    assignments = _outer_assignments(
        step_tasks, folds=args.folds, seed=args.seed
    )
    held_coverage = set()
    manifest_folds = []
    for fold in range(args.folds):
        held = {task for task, value in assignments.items() if value == fold}
        development = step_tasks - held
        inner_val = _inner_validation_tasks(
            development,
            fold=fold,
            seed=args.seed,
            per_suite=args.inner_val_tasks_per_suite,
        )
        train = development - inner_val
        if train & inner_val or train & held or inner_val & held:
            raise AssertionError("Fold task partitions overlap")
        if train | inner_val | held != step_tasks:
            raise AssertionError("Fold task partitions are incomplete")
        held_coverage.update(held)
        fold_dir = args.output_dir / f"fold{fold}"
        files = {}
        for split, split_tasks in (
            ("train", train),
            ("val", inner_val),
            ("held", held),
        ):
            step_path = fold_dir / f"{split}_steps.jsonl"
            trajectory_path = fold_dir / f"{split}_trajectories.jsonl"
            _write_jsonl(
                step_path,
                [row for row in steps if _task_key(row) in split_tasks],
            )
            _write_jsonl(
                trajectory_path,
                [row for row in trajectories if _task_key(row) in split_tasks],
            )
            files[f"{split}_steps"] = {
                "path": str(step_path.resolve()),
                "sha256": _sha256(step_path),
            }
            files[f"{split}_trajectories"] = {
                "path": str(trajectory_path.resolve()),
                "sha256": _sha256(trajectory_path),
            }
        manifest_folds.append(
            {
                "fold": fold,
                "train_task_count": len(train),
                "val_task_count": len(inner_val),
                "held_task_count": len(held),
                "held_tasks_by_suite": {
                    suite: sum(task[0] == suite for task in held)
                    for suite in sorted({task[0] for task in step_tasks})
                },
                "train_tasks": [list(task) for task in sorted(train)],
                "val_tasks": [list(task) for task in sorted(inner_val)],
                "held_tasks": [list(task) for task in sorted(held)],
                "files": files,
            }
        )
    if held_coverage != step_tasks:
        raise AssertionError("Outer held folds do not cover every task")
    manifest = {
        "scope": "suite_balanced_grouped_train_outer_crossfit",
        "source_steps": str(args.steps.resolve()),
        "source_steps_sha256": _sha256(args.steps),
        "source_trajectories": str(args.trajectories.resolve()),
        "source_trajectories_sha256": _sha256(args.trajectories),
        "task_count": len(step_tasks),
        "fold_count": args.folds,
        "assignment_seed": args.seed,
        "inner_val_tasks_per_suite": args.inner_val_tasks_per_suite,
        "every_task_held_exactly_once": True,
        "folds": manifest_folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("fold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
