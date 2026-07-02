"""Annotate AgentDojo splits with clean-conditioned preservation labels.

The raw AgentDojo ``utility`` label answers whether a single attacked run solved
the user task.  For utility preservation, we only want to train/evaluate that
label on tasks that the same victim model can solve in clean conditions.  This
script keeps all steps for skill/risk learning, but sets ``preservation_trainable``
to false for clean trajectories and attacked trajectories whose clean
solvability is below a configurable threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl, write_jsonl
from wmagentattack.schema import StepRecord, TrajectoryRecord


def _load_clean_rates(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["suite"], row["user_task_id"]): float(row["base_success_rate"])
        for row in payload.get("tasks", [])
    }


def _annotate_step(
    row: dict[str, Any],
    *,
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
) -> StepRecord:
    step = StepRecord.model_validate(row)
    base_rate = clean_rates.get((step.domain, step.task_id))
    is_attacked = step.attack_action is not None
    trainable = (
        is_attacked
        and base_rate is not None
        and base_rate >= min_base_success_rate
    )
    return step.model_copy(
        update={
            "base_task_success_rate": base_rate,
            "preservation_trainable": trainable,
        }
    )


def _annotate_trajectory(
    row: dict[str, Any],
    *,
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
) -> TrajectoryRecord:
    trajectory = TrajectoryRecord.model_validate(row)
    steps = [
        _annotate_step(
            step.model_dump(mode="json"),
            clean_rates=clean_rates,
            min_base_success_rate=min_base_success_rate,
        )
        for step in trajectory.steps
    ]
    return trajectory.model_copy(update={"steps": steps})


def _summarize_steps(steps: list[StepRecord]) -> dict[str, Any]:
    attacked = [step for step in steps if step.attack_action is not None]
    trainable = [step for step in steps if step.preservation_trainable]
    return {
        "steps": len(steps),
        "attacked_steps": len(attacked),
        "preservation_trainable_steps": len(trainable),
        "preservation_trainable_attacked_steps": sum(
            step.attack_action is not None for step in trainable
        ),
        "missing_base_rate_steps": sum(
            step.base_task_success_rate is None for step in steps
        ),
        "domain_counts": dict(Counter(step.domain for step in steps)),
        "trainable_domain_counts": dict(Counter(step.domain for step in trainable)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing train/val/test steps and trajectories JSONL.",
    )
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-base-success-rate", type=float, default=0.5)
    args = parser.parse_args()

    clean_rates = _load_clean_rates(args.clean_solvability_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "conditional_preservation_labels",
        "split_dir": str(args.split_dir.resolve()),
        "clean_solvability_json": str(args.clean_solvability_json.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "min_base_success_rate": args.min_base_success_rate,
        "clean_rate_tasks": len(clean_rates),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        step_path = args.split_dir / f"{split}_steps.jsonl"
        trajectory_path = args.split_dir / f"{split}_trajectories.jsonl"
        steps = [
            _annotate_step(
                row,
                clean_rates=clean_rates,
                min_base_success_rate=args.min_base_success_rate,
            )
            for row in read_jsonl(step_path)
        ]
        trajectories = [
            _annotate_trajectory(
                row,
                clean_rates=clean_rates,
                min_base_success_rate=args.min_base_success_rate,
            )
            for row in read_jsonl(trajectory_path)
        ]
        write_jsonl(args.out_dir / f"{split}_steps.jsonl", steps)
        write_jsonl(args.out_dir / f"{split}_trajectories.jsonl", trajectories)
        summary["splits"][split] = {
            **_summarize_steps(steps),
            "trajectories": len(trajectories),
            "preservation_trainable_trajectories": sum(
                bool(trajectory.steps)
                and trajectory.steps[-1].preservation_trainable
                for trajectory in trajectories
            ),
        }

    (args.out_dir / "conditional_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
