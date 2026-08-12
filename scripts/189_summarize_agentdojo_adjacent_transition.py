"""Apply the frozen adjacent-transition gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import (
    OBSERVED_OUTCOME_TARGETS,
    evaluate_adjacent_transition_gate,
)
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import (
    exact_sign_test,
    paired_bootstrap,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _select(
    rows: Sequence[Mapping[str, Any]], *, condition: str, variant: str, seed: int
) -> list[Mapping[str, Any]]:
    output = [
        row
        for row in rows
        if row["condition"] == condition
        and row["variant"] == variant
        and int(row["training_seed"]) == seed
    ]
    if not output:
        raise ValueError(f"empty prediction surface: {condition}/{variant}/{seed}")
    return output


def _task_map(
    rows: Sequence[Mapping[str, Any]], metric: str, *, tail_only: bool = False
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if tail_only and not row["has_next_action"]:
            continue
        value = row[metric]
        if value is not None:
            grouped[str(row["task_name"])].append(float(value))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _variant_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seeds: Sequence[int],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    by_seed = {}
    task_gains = []
    for seed in seeds:
        baseline = _select(
            predictions, condition="tail_action_only", variant=variant, seed=seed
        )
        candidate = _select(
            predictions,
            condition="tail_action_plus_observed_outcomes",
            variant=variant,
            seed=seed,
        )
        if {row["event_id"] for row in baseline} != {
            row["event_id"] for row in candidate
        }:
            raise ValueError("paired confirmation event surfaces differ")
        baseline_nll = _task_map(baseline, "next_action_nll", tail_only=True)
        candidate_nll = _task_map(candidate, "next_action_nll", tail_only=True)
        baseline_accuracy = _task_map(
            baseline, "next_action_correct", tail_only=True
        )
        candidate_accuracy = _task_map(
            candidate, "next_action_correct", tail_only=True
        )
        gains = {
            task: baseline_nll[task] - candidate_nll[task]
            for task in baseline_nll
        }
        task_gains.append(gains)
        outcome = _task_map(candidate, "outcome_bce")
        outcome_prior = _task_map(candidate, "outcome_prior_bce")
        error = _task_map(candidate, "execution_error_bce")
        error_prior = _task_map(candidate, "execution_error_prior_bce")
        by_seed[str(seed)] = {
            "baseline_tail_action_nll": float(np.mean(list(baseline_nll.values()))),
            "candidate_tail_action_nll": float(np.mean(list(candidate_nll.values()))),
            "tail_action_nll_gain": float(np.mean(list(gains.values()))),
            "baseline_tail_action_accuracy": float(
                np.mean(list(baseline_accuracy.values()))
            ),
            "candidate_tail_action_accuracy": float(
                np.mean(list(candidate_accuracy.values()))
            ),
            "tail_action_accuracy_gain": float(
                np.mean(
                    [
                        candidate_accuracy[task] - baseline_accuracy[task]
                        for task in baseline_accuracy
                    ]
                )
            ),
            "candidate_outcome_bce": float(np.mean(list(outcome.values()))),
            "train_prior_outcome_bce": float(np.mean(list(outcome_prior.values()))),
            "outcome_bce_gain_over_train_prior": float(
                np.mean([outcome_prior[task] - outcome[task] for task in outcome])
            ),
            "execution_error_bce": float(np.mean(list(error.values()))),
            "train_prior_execution_error_bce": float(
                np.mean(list(error_prior.values()))
            ),
            "execution_error_bce_gain": float(
                np.mean([error_prior[task] - error[task] for task in error])
            ),
            "legal_prediction_rate": float(
                np.mean([row["legal_prediction"] for row in candidate])
            ),
        }
    tasks = set(task_gains[0])
    if any(set(values) != tasks for values in task_gains):
        raise ValueError("seed task surfaces differ")
    paired_task_gains = {
        task: float(np.mean([values[task] for values in task_gains]))
        for task in sorted(tasks)
    }
    seed_nll = [by_seed[str(seed)]["tail_action_nll_gain"] for seed in seeds]
    seed_accuracy = [
        by_seed[str(seed)]["tail_action_accuracy_gain"] for seed in seeds
    ]
    seed_outcome = [
        by_seed[str(seed)]["outcome_bce_gain_over_train_prior"] for seed in seeds
    ]
    seed_error = [by_seed[str(seed)]["execution_error_bce_gain"] for seed in seeds]
    return {
        "variant": variant,
        "confirmation_tasks": len(tasks),
        "by_seed": by_seed,
        "tail_action_nll_gain_by_seed": dict(zip(map(str, seeds), seed_nll)),
        "tail_action_accuracy_gain_by_seed": dict(
            zip(map(str, seeds), seed_accuracy)
        ),
        "outcome_bce_gain_by_seed": dict(zip(map(str, seeds), seed_outcome)),
        "mean_tail_action_nll_gain": float(np.mean(seed_nll)),
        "mean_tail_action_accuracy_gain": float(np.mean(seed_accuracy)),
        "mean_outcome_bce_gain_over_train_prior": float(np.mean(seed_outcome)),
        "mean_execution_error_bce_gain": float(np.mean(seed_error)),
        "paired_task_nll_gains": paired_task_gains,
        "positive_task_fraction": sum(
            value > 0.0 for value in paired_task_gains.values()
        )
        / len(paired_task_gains),
        "paired_bootstrap": paired_bootstrap(
            list(paired_task_gains.values()),
            draws=int(protocol["uncertainty"]["paired_task_bootstrap_draws"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
        ),
        "paired_sign_test": exact_sign_test(list(paired_task_gains.values())),
        "all_predictions_legal": all(
            by_seed[str(seed)]["legal_prediction_rate"] == 1.0 for seed in seeds
        ),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AgentDojo observed adjacent-transition results",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| representation | tail NLL gain | tail accuracy gain | outcome BCE gain vs prior | execution-error BCE gain | positive tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["variants"].items():
        lines.append(
            f"| {name} | {row['mean_tail_action_nll_gain']:+.6f} | {row['mean_tail_action_accuracy_gain']:+.6f} | {row['mean_outcome_bce_gain_over_train_prior']:+.6f} | {row['mean_execution_error_bce_gain']:+.6f} | {row['positive_task_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            *[
                f"- {name}: **{'PASS' if passed else 'FAIL'}**"
                for name, passed in summary["gate_checks"].items()
            ],
            "",
            "The paired interval and exact sign test are retained as counterevidence, not post-result gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    if not audit["passed"]:
        raise ValueError("preflight did not pass")
    if metrics["neural_training_runs"] != int(
        protocol["fixed_budget"]["neural_training_runs"]
    ):
        raise ValueError("training run budget incomplete")
    if metrics["predictions_sha256"] != file_sha256(args.predictions):
        raise ValueError("prediction hash mismatch")
    expected_confirmation_rows = sum(
        sum(
            1
            for event in dataset["events"]
            if event["task_name"] in set(fold["test_tasks"])
        )
        for fold in dataset["folds"]
    )
    expected_predictions = (
        expected_confirmation_rows
        * len(protocol["training"]["conditions"])
        * len(protocol["training"]["variants"])
        * len(protocol["training"]["training_seeds"])
    )
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"prediction surface incomplete: {len(predictions)} != {expected_predictions}"
        )
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]
    variants = {
        variant: _variant_summary(
            predictions, variant=variant, seeds=seeds, protocol=protocol
        )
        for variant in protocol["training"]["variants"]
    }
    primary = variants[protocol["acceptance_gate"]["primary_variant"]]
    checks = evaluate_adjacent_transition_gate(
        action_nll_seed_gains=list(
            primary["tail_action_nll_gain_by_seed"].values()
        ),
        action_accuracy_seed_gains=list(
            primary["tail_action_accuracy_gain_by_seed"].values()
        ),
        action_task_gains=list(primary["paired_task_nll_gains"].values()),
        outcome_bce_seed_gains=list(primary["outcome_bce_gain_by_seed"].values()),
        execution_error_bce_gain=float(primary["mean_execution_error_bce_gain"]),
        all_predictions_legal=bool(primary["all_predictions_legal"]),
        gates=protocol["acceptance_gate"],
    )
    passed = all(checks.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": (
            "GO_RETAIN_OBSERVED_ADJACENT_TRANSITION_OBJECTIVE"
            if passed
            else "NO_GO_OBSERVED_ADJACENT_MULTITASK_OBJECTIVE_NOT_USEFUL"
        ),
        "gate_passed": passed,
        "gate_checks": checks,
        "variants": variants,
        "dataset": {
            "event_rows": audit["event_rows"],
            "trajectories": audit["trajectories"],
            "adjacent_transitions": audit["adjacent_transitions"],
            "outcome_positive_rows": audit["outcome_positive_rows"],
            "sha256": file_sha256(args.dataset),
            "audit_sha256": file_sha256(args.audit),
        },
        "run": {
            "training_runs": metrics["neural_training_runs"],
            "prediction_rows": len(predictions),
            "predictions_sha256": file_sha256(args.predictions),
            "run_metrics_sha256": file_sha256(args.run_metrics),
            "new_llm_calls": 0,
            "new_tool_executions": 0,
            "real_external_endpoint_calls": 0,
            "new_attack_generation": 0,
            "dreamer_runs": 0,
        },
        "counterevidence_policy": {
            "bootstrap_and_sign_test_are_not_hard_gates": True,
            "no_post_result_reruns": True,
            "no_hyperparameter_selection": True,
        },
    }
    _write(args.output, summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
