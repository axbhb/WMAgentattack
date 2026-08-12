"""Apply the preregistered expansion gate to the 20-task OOF predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test, paired_bootstrap


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: object) -> None:
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


def _task_map(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_name"])].append(float(row[metric]))
    return {task: float(np.mean(values)) for task, values in sorted(grouped.items())}


def _domain_map(task_values: Mapping[str, float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for task, value in task_values.items():
        grouped[task.split("|", 1)[0]].append(float(value))
    return {domain: float(np.mean(values)) for domain, values in sorted(grouped.items())}


def _variant_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seeds: Sequence[int],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    by_seed = {}
    task_gains_by_seed = []
    for seed in seeds:
        baseline_rows = _select(
            predictions, condition="agentdojo_only", variant=variant, seed=seed
        )
        expanded_rows = _select(
            predictions,
            condition="agentdojo_plus_auxiliary",
            variant=variant,
            seed=seed,
        )
        baseline_ids = {str(row["row_id"]) for row in baseline_rows}
        expanded_ids = {str(row["row_id"]) for row in expanded_rows}
        if baseline_ids != expanded_ids:
            raise ValueError("paired AgentDojo confirmation surfaces differ")
        baseline_nll = _task_map(baseline_rows, "action_nll")
        expanded_nll = _task_map(expanded_rows, "action_nll")
        baseline_accuracy = _task_map(baseline_rows, "action_correct")
        expanded_accuracy = _task_map(expanded_rows, "action_correct")
        nll_gains = {
            task: baseline_nll[task] - expanded_nll[task] for task in baseline_nll
        }
        accuracy_gains = {
            task: expanded_accuracy[task] - baseline_accuracy[task]
            for task in baseline_accuracy
        }
        task_gains_by_seed.append(nll_gains)
        by_seed[str(seed)] = {
            "baseline_task_macro_nll": float(np.mean(list(baseline_nll.values()))),
            "expanded_task_macro_nll": float(np.mean(list(expanded_nll.values()))),
            "task_macro_nll_gain": float(np.mean(list(nll_gains.values()))),
            "baseline_task_macro_accuracy": float(np.mean(list(baseline_accuracy.values()))),
            "expanded_task_macro_accuracy": float(np.mean(list(expanded_accuracy.values()))),
            "task_macro_accuracy_gain": float(np.mean(list(accuracy_gains.values()))),
            "legal_prediction_rate_baseline": float(
                np.mean([row["legal_prediction"] for row in baseline_rows])
            ),
            "legal_prediction_rate_expanded": float(
                np.mean([row["legal_prediction"] for row in expanded_rows])
            ),
        }
    tasks = set(task_gains_by_seed[0])
    if any(set(values) != tasks for values in task_gains_by_seed):
        raise ValueError("training-seed task surfaces differ")
    paired_task_gains = {
        task: float(np.mean([values[task] for values in task_gains_by_seed]))
        for task in sorted(tasks)
    }
    seed_nll = [by_seed[str(seed)]["task_macro_nll_gain"] for seed in seeds]
    seed_accuracy = [by_seed[str(seed)]["task_macro_accuracy_gain"] for seed in seeds]
    baseline_nll_mean = float(
        np.mean([by_seed[str(seed)]["baseline_task_macro_nll"] for seed in seeds])
    )
    expanded_nll_mean = float(
        np.mean([by_seed[str(seed)]["expanded_task_macro_nll"] for seed in seeds])
    )
    baseline_accuracy_mean = float(
        np.mean([by_seed[str(seed)]["baseline_task_macro_accuracy"] for seed in seeds])
    )
    expanded_accuracy_mean = float(
        np.mean([by_seed[str(seed)]["expanded_task_macro_accuracy"] for seed in seeds])
    )
    return {
        "variant": variant,
        "confirmation_tasks": len(tasks),
        "by_seed": by_seed,
        "baseline_task_macro_nll": baseline_nll_mean,
        "expanded_task_macro_nll": expanded_nll_mean,
        "task_macro_nll_gain": baseline_nll_mean - expanded_nll_mean,
        "baseline_task_macro_accuracy": baseline_accuracy_mean,
        "expanded_task_macro_accuracy": expanded_accuracy_mean,
        "task_macro_accuracy_gain": expanded_accuracy_mean - baseline_accuracy_mean,
        "nll_gain_by_seed": dict(zip(map(str, seeds), seed_nll)),
        "accuracy_gain_by_seed": dict(zip(map(str, seeds), seed_accuracy)),
        "paired_task_nll_gains": paired_task_gains,
        "paired_domain_nll_gains": _domain_map(paired_task_gains),
        "positive_task_fraction": sum(value > 0.0 for value in paired_task_gains.values()) / len(paired_task_gains),
        "paired_bootstrap": paired_bootstrap(
            list(paired_task_gains.values()),
            draws=int(protocol["uncertainty"]["paired_task_bootstrap_draws"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
        ),
        "paired_sign_test": exact_sign_test(list(paired_task_gains.values())),
        "all_predictions_legal": all(
            by_seed[str(seed)]["legal_prediction_rate_baseline"] == 1.0
            and by_seed[str(seed)]["legal_prediction_rate_expanded"] == 1.0
            for seed in seeds
        ),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Three-source unified action expansion results",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "The comparison is a frozen five-fold, 20-task AgentDojo OOF test. The evaluation rows are identical between the AgentDojo-only and expanded conditions; ToolSandbox and InjecAgent are training-only auxiliary sources.",
        "",
        "| representation | AD-only NLL | expanded NLL | NLL gain | AD-only acc. | expanded acc. | acc. gain | positive tasks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(
            f"| {variant} | {row['baseline_task_macro_nll']:.6f} | {row['expanded_task_macro_nll']:.6f} | {row['task_macro_nll_gain']:+.6f} | {row['baseline_task_macro_accuracy']:.6f} | {row['expanded_task_macro_accuracy']:.6f} | {row['task_macro_accuracy_gain']:+.6f} | {row['positive_task_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            *[f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in summary["gate_checks"].items()],
            "",
            "Confidence intervals and the exact sign test are retained as counterevidence and are not post-hoc gates.",
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
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    if not audit["passed"] or run_metrics["neural_training_runs"] != protocol["fixed_budget"]["neural_training_runs"]:
        raise ValueError("preflight or fixed run budget is incomplete")
    if run_metrics["predictions_sha256"] != file_sha256(args.predictions):
        raise ValueError("prediction hash mismatch")
    expected_rows = sum(
        sum(1 for row in dataset["rows"] if row["source"] == "agentdojo" and row["task_name"] in set(fold["test_tasks"]))
        for fold in dataset["folds"]
    )
    expected_predictions = expected_rows * 2 * 2 * len(protocol["training"]["training_seeds"])
    if len(predictions) != expected_predictions:
        raise ValueError(f"prediction surface incomplete: {len(predictions)} != {expected_predictions}")

    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    variants = {
        variant: _variant_summary(
            predictions, variant=variant, seeds=seeds, protocol=protocol
        )
        for variant in protocol["training"]["variants"]
    }
    primary = variants[protocol["acceptance_gate"]["primary_variant"]]
    counter = variants["structured_markov_v3"]
    gate = protocol["acceptance_gate"]
    checks = {
        "primary_mean_nll_gain": primary["task_macro_nll_gain"] >= float(gate["minimum_mean_nll_gain"]),
        "primary_nll_seed_replication": sum(
            value >= float(gate["minimum_mean_nll_gain"])
            for value in primary["nll_gain_by_seed"].values()
        ) >= int(gate["minimum_threshold_positive_seeds"]),
        "primary_mean_accuracy_gain": primary["task_macro_accuracy_gain"] >= float(gate["minimum_mean_accuracy_gain"]),
        "primary_accuracy_seed_replication": sum(
            value >= float(gate["minimum_mean_accuracy_gain"])
            for value in primary["accuracy_gain_by_seed"].values()
        ) >= int(gate["minimum_threshold_positive_seeds"]),
        "primary_positive_task_fraction": primary["positive_task_fraction"] >= float(gate["minimum_positive_task_fraction"]),
        "structured_counterevidence_not_materially_degraded_nll": counter["task_macro_nll_gain"] >= -float(gate["maximum_counterevidence_nll_degradation"]),
        "structured_counterevidence_not_materially_degraded_accuracy": counter["task_macro_accuracy_gain"] >= -float(gate["maximum_counterevidence_accuracy_degradation"]),
        "all_predictions_legal": primary["all_predictions_legal"] and counter["all_predictions_legal"],
    }
    passed = all(checks.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": (
            "GO_RETAIN_THREE_SOURCE_AUXILIARY_EXPANSION_FOR_FORMAL_AGENTDOJO_TRAINING"
            if passed
            else "NO_GO_NO_REPLICATED_AGENTDOJO_OOF_BENEFIT_FROM_THREE_SOURCE_EXPANSION"
        ),
        "gate_passed": passed,
        "gate_checks": checks,
        "variants": variants,
        "dataset": {
            "source_rows": audit["source_rows"],
            "source_tasks": audit["source_tasks"],
            "candidate_count": audit["candidate_count"],
            "sha256": file_sha256(args.dataset),
            "audit_sha256": file_sha256(args.audit),
        },
        "run": {
            "training_runs": run_metrics["neural_training_runs"],
            "prediction_rows": len(predictions),
            "predictions_sha256": file_sha256(args.predictions),
            "run_metrics_sha256": file_sha256(args.run_metrics),
            "new_llm_calls": 0,
            "new_tool_executions": 0,
            "real_external_endpoint_calls": 0,
            "attack_generation": 0,
            "dreamer_runs": 0,
        },
        "counterevidence_policy": {
            "bootstrap_and_sign_test_are_not_hard_gates": True,
            "no_post_result_reruns": True,
            "no_hyperparameter_selection": True,
        },
    }
    _write_json(args.output, summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
