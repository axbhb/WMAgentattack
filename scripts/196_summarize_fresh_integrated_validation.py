"""Apply the frozen fresh-task integrated-validation acceptance gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256


A = "agentdojo_only_tail_plus_outcomes"
B = "pooled_shared_tail_plus_outcomes"
C = "pooled_source_head_tail_only"
D = "pooled_source_head_tail_plus_outcomes"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _task_macro(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[str(row["task_name"])].append(float(value))
    per_task = {task: _mean(values) for task, values in sorted(grouped.items())}
    return _mean(list(per_task.values())), per_task


def _surface(
    rows: Sequence[Mapping[str, Any]], *, condition: str, seed: int, prediction_type: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["condition"] == condition
        and int(row["training_seed"]) == seed
        and row["prediction_type"] == prediction_type
    ]


def _paired_task_gains(
    left: Mapping[str, float], right: Mapping[str, float], *, higher_is_better: bool
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired task surfaces differ")
    if higher_is_better:
        return {task: float(right[task] - left[task]) for task in sorted(left)}
    return {task: float(left[task] - right[task]) for task in sorted(left)}


def summarize(
    *, protocol: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    seeds = [int(value) for value in protocol["model_validation"]["training_seeds"]]
    conditions = list(protocol["model_validation"]["conditions"])
    condition_seed: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
    all_legal = True
    task_metrics: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for condition in conditions:
        for seed in seeds:
            current = _surface(
                predictions, condition=condition, seed=seed, prediction_type="current_action"
            )
            transition = _surface(
                predictions, condition=condition, seed=seed, prediction_type="transition"
            )
            tail = [row for row in transition if row.get("has_next_action") == 1.0]
            current_nll, current_nll_tasks = _task_macro(current, "action_nll")
            current_acc, current_acc_tasks = _task_macro(current, "action_correct")
            tail_nll, tail_nll_tasks = _task_macro(tail, "next_action_nll")
            tail_acc, tail_acc_tasks = _task_macro(tail, "next_action_correct")
            outcome_bce, outcome_bce_tasks = _task_macro(transition, "outcome_bce")
            outcome_prior_bce, outcome_prior_tasks = _task_macro(
                transition, "outcome_prior_bce"
            )
            error_bce, error_bce_tasks = _task_macro(transition, "execution_error_bce")
            error_prior_bce, error_prior_tasks = _task_macro(
                transition, "execution_error_prior_bce"
            )
            for metric_name, values in (
                ("current_nll", current_nll_tasks),
                ("current_accuracy", current_acc_tasks),
                ("tail_nll", tail_nll_tasks),
                ("tail_accuracy", tail_acc_tasks),
                ("outcome_bce", outcome_bce_tasks),
                ("outcome_prior_bce", outcome_prior_tasks),
                ("error_bce", error_bce_tasks),
                ("error_prior_bce", error_prior_tasks),
            ):
                task_metrics[(condition, seed, "task", metric_name)] = values
            all_legal &= all(float(row["legal_prediction"]) == 1.0 for row in current)
            all_legal &= all(float(row["legal_prediction"]) == 1.0 for row in transition)
            condition_seed[condition][str(seed)] = {
                "current_action_rows": len(current),
                "transition_rows": len(transition),
                "tail_rows": len(tail),
                "tasks": len(current_nll_tasks),
                "tail_tasks": len(tail_nll_tasks),
                "current_action_nll": current_nll,
                "current_action_accuracy": current_acc,
                "tail_action_nll": tail_nll,
                "tail_action_accuracy": tail_acc,
                "outcome_bce": outcome_bce,
                "outcome_prior_bce": outcome_prior_bce,
                "execution_error_bce": error_bce,
                "execution_error_prior_bce": error_prior_bce,
            }

    current_nll_seed_gains = []
    current_acc_seed_gains = []
    current_nll_vs_a = []
    current_acc_vs_a = []
    tail_nll_seed_gains = []
    tail_acc_seed_gains = []
    outcome_seed_gains = []
    error_seed_gains = []
    current_task_gains: dict[str, list[float]] = defaultdict(list)
    tail_task_gains: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        d = condition_seed[D][str(seed)]
        b = condition_seed[B][str(seed)]
        a = condition_seed[A][str(seed)]
        c = condition_seed[C][str(seed)]
        current_nll_seed_gains.append(b["current_action_nll"] - d["current_action_nll"])
        current_acc_seed_gains.append(d["current_action_accuracy"] - b["current_action_accuracy"])
        current_nll_vs_a.append(a["current_action_nll"] - d["current_action_nll"])
        current_acc_vs_a.append(d["current_action_accuracy"] - a["current_action_accuracy"])
        tail_nll_seed_gains.append(c["tail_action_nll"] - d["tail_action_nll"])
        tail_acc_seed_gains.append(d["tail_action_accuracy"] - c["tail_action_accuracy"])
        outcome_seed_gains.append(d["outcome_prior_bce"] - d["outcome_bce"])
        error_seed_gains.append(
            d["execution_error_prior_bce"] - d["execution_error_bce"]
        )
        nll_tasks = _paired_task_gains(
            task_metrics[(B, seed, "task", "current_nll")],
            task_metrics[(D, seed, "task", "current_nll")],
            higher_is_better=False,
        )
        for task, gain in nll_tasks.items():
            current_task_gains[task].append(gain)
        tail_tasks = _paired_task_gains(
            task_metrics[(C, seed, "task", "tail_nll")],
            task_metrics[(D, seed, "task", "tail_nll")],
            higher_is_better=False,
        )
        for task, gain in tail_tasks.items():
            tail_task_gains[task].append(gain)

    task_mean_current_gains = {
        task: _mean(values) for task, values in sorted(current_task_gains.items())
    }
    task_mean_tail_gains = {
        task: _mean(values) for task, values in sorted(tail_task_gains.items())
    }
    positive_current_fraction = sum(
        value > 0.0 for value in task_mean_current_gains.values()
    ) / max(1, len(task_mean_current_gains))
    positive_tail_fraction = sum(
        value > 0.0 for value in task_mean_tail_gains.values()
    ) / max(1, len(task_mean_tail_gains))
    gate = protocol["acceptance_gate"]
    minimum_seed_replication = int(gate["minimum_threshold_positive_training_seeds"])
    checks = {
        "current_action_nll_gain_vs_pooled_shared": _mean(current_nll_seed_gains)
        >= float(gate["minimum_current_action_nll_gain_vs_pooled_shared"]),
        "current_action_accuracy_gain_vs_pooled_shared": _mean(current_acc_seed_gains)
        >= float(gate["minimum_current_action_accuracy_gain_vs_pooled_shared"]),
        "current_action_nll_seed_replication": sum(
            value >= float(gate["minimum_current_action_nll_gain_vs_pooled_shared"])
            for value in current_nll_seed_gains
        )
        >= minimum_seed_replication,
        "current_action_accuracy_seed_replication": sum(
            value >= float(gate["minimum_current_action_accuracy_gain_vs_pooled_shared"])
            for value in current_acc_seed_gains
        )
        >= minimum_seed_replication,
        "current_action_positive_task_fraction": positive_current_fraction
        >= float(gate["minimum_current_action_positive_task_fraction"]),
        "current_action_nll_noninferior_to_agentdojo_only": -_mean(current_nll_vs_a)
        <= float(gate["maximum_current_action_nll_degradation_vs_agentdojo_only"]),
        "current_action_accuracy_noninferior_to_agentdojo_only": -_mean(current_acc_vs_a)
        <= float(gate["maximum_current_action_accuracy_degradation_vs_agentdojo_only"]),
        "tail_action_nll_gain_vs_tail_only": _mean(tail_nll_seed_gains)
        >= float(gate["minimum_tail_action_nll_gain_vs_tail_only"]),
        "tail_action_accuracy_gain_vs_tail_only": _mean(tail_acc_seed_gains)
        >= float(gate["minimum_tail_action_accuracy_gain_vs_tail_only"]),
        "tail_action_nll_seed_replication": sum(
            value >= float(gate["minimum_tail_action_nll_gain_vs_tail_only"])
            for value in tail_nll_seed_gains
        )
        >= minimum_seed_replication,
        "tail_action_accuracy_seed_replication": sum(
            value >= float(gate["minimum_tail_action_accuracy_gain_vs_tail_only"])
            for value in tail_acc_seed_gains
        )
        >= minimum_seed_replication,
        "outcome_bce_gain_over_train_prior": _mean(outcome_seed_gains)
        >= float(gate["minimum_outcome_bce_gain_over_train_prior"]),
        "all_predictions_legal": all_legal,
        "fixed_training_budget_complete": int(metrics["neural_training_runs"])
        == int(protocol["fixed_budget"]["neural_training_runs"]),
        "zero_external_or_attack_activity": (
            metrics["real_external_endpoint_calls"] == 0
            and metrics["attack_episodes"] == 0
            and metrics["dreamer_runs"] == 0
        ),
    }
    passed = all(checks.values())
    decision = (
        "GO_FRESH_INTEGRATED_STRUCTURED_WORLD_MODEL"
        if passed
        else "NO_GO_FRESH_INTEGRATED_STRUCTURED_WORLD_MODEL"
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "passed": passed,
        "checks": checks,
        "gate_clauses_passed": sum(checks.values()),
        "gate_clauses_total": len(checks),
        "condition_seed_metrics": condition_seed,
        "primary_comparisons": {
            "source_head_vs_pooled_shared": {
                "current_action_nll_seed_gains": current_nll_seed_gains,
                "current_action_nll_mean_gain": _mean(current_nll_seed_gains),
                "current_action_accuracy_seed_gains": current_acc_seed_gains,
                "current_action_accuracy_mean_gain": _mean(current_acc_seed_gains),
                "positive_task_fraction": positive_current_fraction,
                "task_nll_gains": task_mean_current_gains,
            },
            "source_head_vs_agentdojo_only": {
                "current_action_nll_seed_gains": current_nll_vs_a,
                "current_action_nll_mean_gain": _mean(current_nll_vs_a),
                "current_action_accuracy_seed_gains": current_acc_vs_a,
                "current_action_accuracy_mean_gain": _mean(current_acc_vs_a),
            },
            "outcome_multitask_vs_tail_only": {
                "tail_action_nll_seed_gains": tail_nll_seed_gains,
                "tail_action_nll_mean_gain": _mean(tail_nll_seed_gains),
                "tail_action_accuracy_seed_gains": tail_acc_seed_gains,
                "tail_action_accuracy_mean_gain": _mean(tail_acc_seed_gains),
                "positive_task_fraction": positive_tail_fraction,
                "task_nll_gains": task_mean_tail_gains,
                "outcome_bce_seed_gains_over_prior": outcome_seed_gains,
                "outcome_bce_mean_gain_over_prior": _mean(outcome_seed_gains),
                "execution_error_bce_seed_gains": error_seed_gains,
                "execution_error_bce_mean_gain": _mean(error_seed_gains),
            },
        },
        "counterevidence": {
            "failed_gate_clauses": [key for key, value in checks.items() if not value],
            "tasks_harmed_by_source_head": sorted(
                task for task, value in task_mean_current_gains.items() if value <= 0.0
            ),
            "tasks_harmed_by_outcome_multitask": sorted(
                task for task, value in task_mean_tail_gains.items() if value <= 0.0
            ),
            "historical_agentdojo_training_is_attack_heavy": True,
            "fresh_confirmation_is_clean_only": True,
            "fresh_task_count_is_twelve": True,
        },
        "authorization": {
            "architecture_retention": "AUTHORIZED" if passed else "NOT_AUTHORIZED",
            "attack_generation": "NOT_AUTHORIZED",
            "dreamer_training": "NOT_AUTHORIZED",
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    p = summary["primary_comparisons"]
    checks = summary["checks"]
    lines = [
        "# Fresh integrated Structured world-model validation",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Primary results",
        "",
        "| Comparison | NLL gain | Accuracy gain | Positive-task fraction |",
        "|---|---:|---:|---:|",
        (
            "| Source-specific head vs pooled shared | "
            f"{p['source_head_vs_pooled_shared']['current_action_nll_mean_gain']:.6f} | "
            f"{p['source_head_vs_pooled_shared']['current_action_accuracy_mean_gain']:.6f} | "
            f"{p['source_head_vs_pooled_shared']['positive_task_fraction']:.3f} |"
        ),
        (
            "| Source-specific head vs AgentDojo-only | "
            f"{p['source_head_vs_agentdojo_only']['current_action_nll_mean_gain']:.6f} | "
            f"{p['source_head_vs_agentdojo_only']['current_action_accuracy_mean_gain']:.6f} | — |"
        ),
        (
            "| Outcome multitask vs tail-only | "
            f"{p['outcome_multitask_vs_tail_only']['tail_action_nll_mean_gain']:.6f} | "
            f"{p['outcome_multitask_vs_tail_only']['tail_action_accuracy_mean_gain']:.6f} | "
            f"{p['outcome_multitask_vs_tail_only']['positive_task_fraction']:.3f} |"
        ),
        "",
        "## Frozen gate",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Counterevidence",
            "",
            f"- Failed clauses: {', '.join(summary['counterevidence']['failed_gate_clauses']) or 'none'}.",
            f"- Tasks harmed by source-specific head: {len(summary['counterevidence']['tasks_harmed_by_source_head'])}.",
            f"- Tasks harmed by outcome multitask objective: {len(summary['counterevidence']['tasks_harmed_by_outcome_multitask'])}.",
            "- Historical AgentDojo training remains attack-heavy while the sealed confirmation set is clean-only.",
            "- This result does not authorize attack generation, Dreamer training, utility/value heads, or a planner.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    summary = summarize(protocol=protocol, predictions=predictions, metrics=metrics)
    summary["predictions_sha256"] = file_sha256(args.predictions)
    summary["run_metrics_sha256"] = file_sha256(args.run_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
