"""Create continuous, leakage-resistant utility labels for AgentDojo splits."""

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
from wmagentattack.probability_labels import (
    build_training_context_evidence,
    build_training_global_evidence,
    context_for_row,
    estimate_probability_label,
    global_for_row,
    load_clean_evidence,
)
from wmagentattack.schema import StepRecord, TrajectoryRecord


def _annotate_trajectory(
    row: dict[str, Any],
    *,
    split: str,
    clean_evidence,
    training_context,
    training_global,
    global_prior_strength: float,
    clean_prior_strength: float,
    context_max_strength: float,
    observation_strength: float,
) -> TrajectoryRecord:
    trajectory = TrajectoryRecord.model_validate(row)
    key = (trajectory.domain, trajectory.task_id)
    clean = clean_evidence.get(key)
    if clean is None:
        raise KeyError(f"Missing clean evidence for {key}")
    attacked = bool(trajectory.steps and trajectory.steps[0].attack_action is not None)
    context = context_for_row(
        trajectory.model_dump(mode="json"),
        training_context=training_context,
        split=split,
    )
    global_attack = global_for_row(
        trajectory.model_dump(mode="json"),
        training_global=training_global,
        split=split,
    )
    label = estimate_probability_label(
        clean=clean,
        global_attack=global_attack,
        attacked=attacked,
        observed_success=trajectory.final_task_success,
        context=context,
        split="loo" if split == "train" else "train_only",
        global_prior_strength=global_prior_strength,
        clean_prior_strength=clean_prior_strength,
        context_max_strength=context_max_strength,
        observation_strength=observation_strength,
    )
    update = {
        "utility_probability_target": label.utility_probability,
        "preservation_probability_target": label.preservation_probability,
        "probability_label_alpha": label.alpha,
        "probability_label_beta": label.beta,
        "probability_label_variance": label.variance,
        "probability_label_confidence": label.confidence,
        "probability_label_source": label.source,
    }
    steps = [step.model_copy(update=update) for step in trajectory.steps]
    return trajectory.model_copy(update={"steps": steps})


def _summary(trajectories: list[TrajectoryRecord]) -> dict[str, Any]:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    attacked = [step for step in steps if step.attack_action is not None]
    targets = [step.utility_probability_target for step in attacked]
    preservation = [
        step.preservation_probability_target
        for step in attacked
        if step.preservation_probability_target is not None
    ]
    confidences = [step.probability_label_confidence for step in attacked]
    return {
        "trajectories": len(trajectories),
        "steps": len(steps),
        "attacked_steps": len(attacked),
        "continuous_target_steps": len(targets),
        "utility_probability_min": min(targets) if targets else None,
        "utility_probability_mean": sum(targets) / len(targets) if targets else None,
        "utility_probability_max": max(targets) if targets else None,
        "preservation_probability_mean": (
            sum(preservation) / len(preservation) if preservation else None
        ),
        "confidence_mean": sum(confidences) / len(confidences) if confidences else None,
        "source_counts": dict(Counter(step.probability_label_source for step in attacked)),
        "domain_counts": dict(Counter(step.domain for step in steps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--global-prior-strength", type=float, default=1.0)
    parser.add_argument("--clean-prior-strength", type=float, default=2.0)
    parser.add_argument("--context-max-strength", type=float, default=4.0)
    parser.add_argument("--observation-strength", type=float, default=1.0)
    args = parser.parse_args()

    clean_payload = json.loads(args.clean_solvability_json.read_text(encoding="utf-8"))
    clean_evidence = load_clean_evidence(clean_payload)
    raw_splits = {
        split: read_jsonl(args.split_dir / f"{split}_trajectories.jsonl")
        for split in ("train", "val", "test")
    }
    training_context = build_training_context_evidence(raw_splits["train"])
    training_global = build_training_global_evidence(raw_splits["train"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, rows in raw_splits.items():
        trajectories = [
            _annotate_trajectory(
                row,
                split=split,
                clean_evidence=clean_evidence,
                training_context=training_context,
                training_global=training_global,
                global_prior_strength=args.global_prior_strength,
                clean_prior_strength=args.clean_prior_strength,
                context_max_strength=args.context_max_strength,
                observation_strength=args.observation_strength,
            )
            for row in rows
        ]
        write_jsonl(args.out_dir / f"{split}_trajectories.jsonl", trajectories)
        write_jsonl(
            args.out_dir / f"{split}_steps.jsonl",
            [step for trajectory in trajectories for step in trajectory.steps],
        )
        summaries[split] = _summary(trajectories)

    summary = {
        "scope": "continuous_probability_preservation_labels",
        "method": "empirical_global_plus_weak_clean_plus_capped_train_attack_location",
        "leakage_control": {
            "train": "leave-one-out attack-location context",
            "val_test": "training-split attack-location context only",
        },
        "split_dir": str(args.split_dir.resolve()),
        "clean_solvability_json": str(args.clean_solvability_json.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "global_prior_strength": args.global_prior_strength,
        "clean_prior_strength": args.clean_prior_strength,
        "training_global_attack_successes": training_global.successes,
        "training_global_attack_attempts": training_global.attempts,
        "training_global_attack_rate": (
            training_global.successes / training_global.attempts
        ),
        "context_max_strength": args.context_max_strength,
        "observation_strength": args.observation_strength,
        "clean_task_count": len(clean_evidence),
        "training_context_count": len(training_context),
        "splits": summaries,
    }
    (args.out_dir / "probability_label_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
