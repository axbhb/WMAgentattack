"""Apply the frozen independent dynamics/evidence gates to the three-arm probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


LOWER_IS_BETTER = {
    "task_macro_action_nll",
    "task_macro_error_recovery_action_nll",
    "task_macro_evidence_status_nll",
    "task_macro_supported_brier",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _aggregate_runs(run_metrics: Mapping[str, Any]) -> dict[str, Any]:
    output = defaultdict(lambda: defaultdict(dict))
    metrics = (
        "task_macro_action_nll",
        "task_macro_action_accuracy",
        "task_macro_stop_accuracy",
        "task_macro_argument_key_f1",
        "task_macro_error_recovery_action_nll",
        "task_macro_evidence_status_nll",
        "task_macro_evidence_status_accuracy",
        "task_macro_supported_brier",
    )
    for variant in run_metrics["variants"]:
        runs = [row for row in run_metrics["runs"] if row["variant"] == variant]
        for split in ("training", "calibration", "confirmation"):
            for metric in metrics:
                output[variant][split][metric] = _mean(
                    [row["metrics"][split].get(metric) for row in runs]
                )
    return {
        variant: {split: dict(values) for split, values in splits.items()}
        for variant, splits in output.items()
    }


def _metric_gain(candidate: float | None, baseline: float | None, metric: str) -> float | None:
    if candidate is None or baseline is None:
        return None
    return (
        float(baseline - candidate)
        if metric in LOWER_IS_BETTER
        else float(candidate - baseline)
    )


def _run_comparisons(
    run_metrics: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = {
        "observable_vs_semantic": ("semantic_markov", "observable_execution"),
        "ledger_vs_observable": (
            "observable_execution",
            "observable_execution_ledger_v2",
        ),
    }
    metrics = (
        "task_macro_action_nll",
        "task_macro_action_accuracy",
        "task_macro_evidence_status_nll",
        "task_macro_supported_brier",
    )
    output = {}
    for name, (baseline, candidate) in pairs.items():
        output[name] = {}
        for split in ("calibration", "confirmation"):
            split_rows = {}
            for metric in metrics:
                split_rows[f"{metric}_gain"] = _metric_gain(
                    aggregate[candidate][split][metric],
                    aggregate[baseline][split][metric],
                    metric,
                )
                seed_gains = []
                for seed in run_metrics["training_seeds"]:
                    base_run = next(
                        row
                        for row in run_metrics["runs"]
                        if row["variant"] == baseline
                        and int(row["training_seed"]) == int(seed)
                    )
                    candidate_run = next(
                        row
                        for row in run_metrics["runs"]
                        if row["variant"] == candidate
                        and int(row["training_seed"]) == int(seed)
                    )
                    seed_gains.append(
                        {
                            "training_seed": int(seed),
                            "gain": _metric_gain(
                                candidate_run["metrics"][split].get(metric),
                                base_run["metrics"][split].get(metric),
                                metric,
                            ),
                        }
                    )
                split_rows[f"{metric}_seed_gains"] = seed_gains
            output[name][split] = split_rows
    return output


def _per_task_metric(
    predictions: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    split: str,
    prediction_type: str,
    metric: str,
) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in predictions:
        if (
            row["variant"] == variant
            and row["split"] == split
            and row["prediction_type"] == prediction_type
            and row.get(metric) is not None
        ):
            grouped[str(row["task_id"])].append(float(row[metric]))
    return {task_id: float(np.mean(values)) for task_id, values in grouped.items()}


def _bootstrap_interval(values: np.ndarray, *, seed: int, draws: int) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _paired_task_comparisons(
    predictions: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    pairs = {
        "observable_vs_semantic": ("semantic_markov", "observable_execution"),
        "ledger_vs_observable": (
            "observable_execution",
            "observable_execution_ledger_v2",
        ),
    }
    metric_types = {
        "action_nll": "dynamics",
        "action_correct": "dynamics",
        "status_nll": "evidence",
        "supported_brier": "evidence",
    }
    output = {}
    draws = int(protocol["uncertainty"]["paired_task_bootstrap_draws"])
    seed = int(protocol["uncertainty"]["bootstrap_seed"])
    for pair_name, (baseline, candidate) in pairs.items():
        output[pair_name] = {}
        for split in ("calibration", "confirmation"):
            output[pair_name][split] = {}
            for metric, prediction_type in metric_types.items():
                base = _per_task_metric(
                    predictions,
                    variant=baseline,
                    split=split,
                    prediction_type=prediction_type,
                    metric=metric,
                )
                current = _per_task_metric(
                    predictions,
                    variant=candidate,
                    split=split,
                    prediction_type=prediction_type,
                    metric=metric,
                )
                tasks = sorted(set(base) & set(current))
                values = np.asarray(
                    [
                        base[task] - current[task]
                        if metric in {"action_nll", "status_nll", "supported_brier"}
                        else current[task] - base[task]
                        for task in tasks
                    ],
                    dtype=float,
                )
                output[pair_name][split][metric] = {
                    "tasks": len(tasks),
                    "mean_gain": float(values.mean()) if len(values) else None,
                    "median_gain": float(np.median(values)) if len(values) else None,
                    "positive_tasks": int((values > 0).sum()),
                    "zero_tasks": int((values == 0).sum()),
                    "negative_tasks": int((values < 0).sum()),
                    "bootstrap_95_interval": _bootstrap_interval(
                        values,
                        seed=seed + sum(ord(ch) for ch in f"{pair_name}:{split}:{metric}"),
                        draws=draws,
                    ),
                    "task_gains": {
                        task: float(value) for task, value in zip(tasks, values)
                    },
                }
    return output


def _positive_seed_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(row["gain"] is not None and float(row["gain"]) > 0.0 for row in rows)


def summarize(
    protocol: Mapping[str, Any],
    run_metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = _aggregate_runs(run_metrics)
    comparisons = _run_comparisons(run_metrics, aggregate)
    paired = _paired_task_comparisons(predictions, protocol)
    thresholds = protocol["frozen_gates"]
    observable = comparisons["observable_vs_semantic"]
    ledger = comparisons["ledger_vs_observable"]

    observable_dynamics_conditions = {
        "calibration_action_nll_gain": observable["calibration"]["task_macro_action_nll_gain"]
        >= float(thresholds["minimum_action_nll_gain"]),
        "confirmation_action_nll_gain": observable["confirmation"]["task_macro_action_nll_gain"]
        >= float(thresholds["minimum_action_nll_gain"]),
        "confirmation_positive_training_seeds": _positive_seed_count(
            observable["confirmation"]["task_macro_action_nll_seed_gains"]
        )
        >= int(thresholds["minimum_positive_training_seeds"]),
        "confirmation_positive_tasks": paired["observable_vs_semantic"]["confirmation"]["action_nll"]["positive_tasks"]
        >= int(thresholds["minimum_confirmation_positive_tasks"]),
        "confirmation_evidence_nll_noninferior": observable["confirmation"]["task_macro_evidence_status_nll_gain"]
        >= -float(thresholds["maximum_cross_head_nll_regression"]),
    }
    observable_evidence_conditions = {
        "calibration_evidence_nll_gain": observable["calibration"]["task_macro_evidence_status_nll_gain"]
        >= float(thresholds["minimum_evidence_nll_gain"]),
        "confirmation_evidence_nll_gain": observable["confirmation"]["task_macro_evidence_status_nll_gain"]
        >= float(thresholds["minimum_evidence_nll_gain"]),
        "confirmation_supported_brier_nonnegative": observable["confirmation"]["task_macro_supported_brier_gain"]
        >= 0.0,
        "confirmation_positive_training_seeds": _positive_seed_count(
            observable["confirmation"]["task_macro_evidence_status_nll_seed_gains"]
        )
        >= int(thresholds["minimum_positive_training_seeds"]),
        "confirmation_positive_tasks": paired["observable_vs_semantic"]["confirmation"]["status_nll"]["positive_tasks"]
        >= int(thresholds["minimum_confirmation_positive_tasks"]),
    }
    ledger_dynamics_conditions = {
        "calibration_action_nll_gain": ledger["calibration"]["task_macro_action_nll_gain"]
        >= float(thresholds["minimum_action_nll_gain"]),
        "confirmation_action_nll_gain": ledger["confirmation"]["task_macro_action_nll_gain"]
        >= float(thresholds["minimum_action_nll_gain"]),
        "confirmation_positive_training_seeds": _positive_seed_count(
            ledger["confirmation"]["task_macro_action_nll_seed_gains"]
        )
        >= int(thresholds["minimum_positive_training_seeds"]),
        "confirmation_positive_tasks": paired["ledger_vs_observable"]["confirmation"]["action_nll"]["positive_tasks"]
        >= int(thresholds["minimum_confirmation_positive_tasks"]),
        "confirmation_evidence_nll_noninferior": ledger["confirmation"]["task_macro_evidence_status_nll_gain"]
        >= -float(thresholds["maximum_cross_head_nll_regression"]),
    }
    ledger_evidence_conditions = {
        "calibration_evidence_nll_gain": ledger["calibration"]["task_macro_evidence_status_nll_gain"]
        >= float(thresholds["minimum_evidence_nll_gain"]),
        "confirmation_evidence_nll_gain": ledger["confirmation"]["task_macro_evidence_status_nll_gain"]
        >= float(thresholds["minimum_evidence_nll_gain"]),
        "calibration_supported_brier_nonnegative": ledger["calibration"]["task_macro_supported_brier_gain"]
        >= 0.0,
        "confirmation_supported_brier_nonnegative": ledger["confirmation"]["task_macro_supported_brier_gain"]
        >= 0.0,
        "confirmation_positive_training_seeds": _positive_seed_count(
            ledger["confirmation"]["task_macro_evidence_status_nll_seed_gains"]
        )
        >= int(thresholds["minimum_positive_training_seeds"]),
        "confirmation_positive_tasks": paired["ledger_vs_observable"]["confirmation"]["status_nll"]["positive_tasks"]
        >= int(thresholds["minimum_confirmation_positive_tasks"]),
        "confirmation_dynamics_nll_noninferior": ledger["confirmation"]["task_macro_action_nll_gain"]
        >= -float(thresholds["maximum_cross_head_nll_regression"]),
    }
    gates = {
        "observable_dynamics_increment": {
            "conditions": observable_dynamics_conditions,
            "passed": all(observable_dynamics_conditions.values()),
        },
        "observable_evidence_increment": {
            "conditions": observable_evidence_conditions,
            "passed": all(observable_evidence_conditions.values()),
        },
        "ledger_dynamics_increment": {
            "conditions": ledger_dynamics_conditions,
            "passed": all(ledger_dynamics_conditions.values()),
        },
        "ledger_evidence_increment": {
            "conditions": ledger_evidence_conditions,
            "passed": all(ledger_evidence_conditions.values()),
        },
    }
    accepted_dynamics = (
        "observable_execution_ledger_v2"
        if gates["ledger_dynamics_increment"]["passed"]
        else "observable_execution"
        if gates["observable_dynamics_increment"]["passed"]
        else "semantic_markov"
    )
    accepted_evidence = (
        "observable_execution_ledger_v2"
        if gates["ledger_evidence_increment"]["passed"]
        else "observable_execution"
        if gates["observable_evidence_increment"]["passed"]
        else "semantic_markov"
    )
    any_increment = any(row["passed"] for row in gates.values())
    decision = (
        "CUSTOM_PANEL_V2_ARCHITECTURE_INCREMENT_PROVISIONAL_GO"
        if any_increment
        else "CUSTOM_PANEL_V2_ARCHITECTURE_NO_INCREMENT"
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "aggregate_metrics": aggregate,
        "comparisons": comparisons,
        "paired_task_counterevidence": paired,
        "gates": gates,
        "accepted_heads": {
            "victim_dynamics": accepted_dynamics,
            "evidence_progress": accepted_evidence,
        },
        "permissions": {
            "small_clean_architecture_followup": any_increment,
            "completion_or_reporting_training": False,
            "attack_data": False,
            "h2_attack_planning": False,
            "dreamer_training": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Custom clean panel-v2 architecture ablation",
        "",
        f"Binding decision: `{report['decision']}`",
        "",
        "## Confirmation metrics (mean across frozen training seeds)",
        "",
        "| Variant | Action NLL ↓ | Action acc ↑ | Evidence NLL ↓ | Supported Brier ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, splits in report["aggregate_metrics"].items():
        row = splits["confirmation"]
        lines.append(
            "| "
            + variant
            + " | "
            + " | ".join(
                _format_metric(row[key])
                for key in (
                    "task_macro_action_nll",
                    "task_macro_action_accuracy",
                    "task_macro_evidence_status_nll",
                    "task_macro_supported_brier",
                )
            )
            + " |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, gate in report["gates"].items():
        lines.append(f"- `{name}`: **{'PASS' if gate['passed'] else 'FAIL'}**")
        for condition, passed in gate["conditions"].items():
            lines.append(f"  - {condition}: {passed}")
    lines.extend(
        [
            "",
            "## Accepted head representations",
            "",
            f"- Victim dynamics: `{report['accepted_heads']['victim_dynamics']}`",
            f"- Evidence progress: `{report['accepted_heads']['evidence_progress']}`",
            "",
            "## Boundaries",
            "",
            "Completion/reporting, attack data, H2, and Dreamer remain blocked. "
            "Bootstrap intervals and negative per-task gains are retained as counterevidence; "
            "this fixed small probe cannot establish a large-training claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    expected_runs = len(protocol["frozen_variants"]) * len(
        protocol["training"]["training_seeds"]
    )
    if len(run_metrics["runs"]) != expected_runs:
        raise ValueError("incomplete architecture training runs")
    if run_metrics["protocol_sha256"] != _sha256(args.protocol):
        raise ValueError("training used a different protocol")
    expected_predictions = expected_runs * (
        int(protocol["fixed_budget"]["prefixes"])
        + int(protocol["fixed_budget"]["evidence_rows"])
    )
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"expected {expected_predictions} predictions, found {len(predictions)}"
        )
    output = summarize(protocol, run_metrics, predictions)
    output["protocol_sha256"] = _sha256(args.protocol)
    output["run_metrics_sha256"] = _sha256(args.run_metrics)
    output["predictions_sha256"] = _sha256(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.report.write_text(_markdown(output), encoding="utf-8")
    print(json.dumps({
        "decision": output["decision"],
        "gates": {name: row["passed"] for name, row in output["gates"].items()},
        "accepted_heads": output["accepted_heads"],
        "permissions": output["permissions"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
