"""Train/evaluate the frozen low-data victim-event IO-HMM baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_hmm_world_model import (
    IOHMMConfig,
    HierarchicalDiscreteIOHMM,
    SmoothedContextMarkovBaseline,
    evaluate_markov_baseline,
    evaluate_next_events,
)
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import TrajectoryRecord


def _context_token(trajectory: TrajectoryRecord, step_index: int, mode: str) -> str:
    step = trajectory.steps[step_index]
    attack = step.attack_action or "clean"
    parts = [f"domain={trajectory.domain}"]
    if mode in {"domain_attack", "domain_attack_location"}:
        parts.append(f"attack={attack}")
    if mode == "domain_attack_location":
        parts.append(f"location={step.attack_location or 'none'}")
    return "|".join(parts)


def _load_sequences(path: Path, mode: str):
    trajectories = [
        TrajectoryRecord.model_validate(row) for row in read_jsonl(path)
    ]
    observations = [
        [step.selected_skill for step in trajectory.steps]
        for trajectory in trajectories
    ]
    inputs = [
        [_context_token(trajectory, index, mode) for index in range(len(trajectory.steps))]
        for trajectory in trajectories
    ]
    groups = {(item.domain, item.task_id) for item in trajectories}
    return trajectories, observations, inputs, groups


def _overlap(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> list[str]:
    return [f"{suite}|{task}" for suite, task in sorted(left & right)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backoff-strength", type=float, default=2.0)
    parser.add_argument(
        "--context-mode",
        choices=["domain", "domain_attack", "domain_attack_location"],
        default="domain_attack",
    )
    parser.add_argument(
        "--smoke-allow-task-overlap",
        action="store_true",
        help="Permit legacy trajectory-level splits, but mark results non-confirmatory.",
    )
    args = parser.parse_args()

    train_rows, train_obs, train_inputs, train_groups = _load_sequences(
        args.train, args.context_mode
    )
    val_rows, val_obs, val_inputs, val_groups = _load_sequences(
        args.validation, args.context_mode
    )
    test_data = (
        _load_sequences(args.test, args.context_mode) if args.test is not None else None
    )
    test_groups = test_data[3] if test_data else set()
    overlap = {
        "train_validation": _overlap(train_groups, val_groups),
        "train_test": _overlap(train_groups, test_groups),
        "validation_test": _overlap(val_groups, test_groups),
    }
    if any(overlap.values()) and not args.smoke_allow_task_overlap:
        raise ValueError(
            "Task-group overlap detected; use grouped files or explicitly mark a smoke run "
            "with --smoke-allow-task-overlap"
        )

    model = HierarchicalDiscreteIOHMM(
        IOHMMConfig(
            num_states=args.states,
            max_iterations=args.iterations,
            restarts=args.restarts,
            random_seed=args.seed,
            backoff_strength=args.backoff_strength,
        )
    ).fit(train_obs, train_inputs)
    counterbaseline = SmoothedContextMarkovBaseline().fit(train_obs, train_inputs)
    metrics: dict[str, Any] = {
        "train": evaluate_next_events(model, train_obs, train_inputs),
        "validation": evaluate_next_events(model, val_obs, val_inputs),
    }
    if test_data:
        metrics["test"] = evaluate_next_events(model, test_data[1], test_data[2])
    baseline_metrics: dict[str, Any] = {
        "train": evaluate_markov_baseline(counterbaseline, train_obs, train_inputs),
        "validation": evaluate_markov_baseline(counterbaseline, val_obs, val_inputs),
    }
    if test_data:
        baseline_metrics["test"] = evaluate_markov_baseline(
            counterbaseline, test_data[1], test_data[2]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "io_hmm_model.json"
    model.save(model_path)
    report = {
        "scope": "victim skill dynamics only; AgentDojo sandbox",
        "confirmatory": not any(overlap.values()),
        "model_type": "hierarchical_discrete_io_hmm",
        "config": model.to_dict()["config"],
        "context_mode": args.context_mode,
        "data": {
            "train_trajectories": len(train_rows),
            "validation_trajectories": len(val_rows),
            "test_trajectories": len(test_data[0]) if test_data else 0,
            "task_group_overlap": overlap,
        },
        "metrics": metrics,
        "counterbaseline": {
            "model_type": "smoothed_context_first_order_markov",
            "metrics": baseline_metrics,
            "validation_io_hmm_minus_markov_mean_nll": (
                metrics["validation"]["mean_event_nll"]
                - baseline_metrics["validation"]["mean_event_nll"]
            ),
        },
        "training_log_likelihood": model.training_log_likelihood,
        "observation_vocab": model.observation_vocab,
        "input_vocab_size": len(model.input_vocab),
        "model_path": str(model_path.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
