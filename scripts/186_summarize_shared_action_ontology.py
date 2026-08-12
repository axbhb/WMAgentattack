"""Summarize the frozen shared-action-ontology alignment experiment."""

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

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test, paired_bootstrap


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _task_map(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_name"])].append(float(row[metric]))
    return {task: float(np.mean(values)) for task, values in sorted(grouped.items())}


def _comparison(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    baseline_condition: str,
    candidate_condition: str,
    variant: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    by_seed = {}
    task_nll_gains = []
    for seed in seeds:
        baseline = _select(
            baseline_rows, condition=baseline_condition, variant=variant, seed=seed
        )
        candidate = _select(
            candidate_rows, condition=candidate_condition, variant=variant, seed=seed
        )
        if {row["row_id"] for row in baseline} != {row["row_id"] for row in candidate}:
            raise ValueError("paired confirmation row surfaces differ")
        baseline_nll = _task_map(baseline, "action_nll")
        candidate_nll = _task_map(candidate, "action_nll")
        baseline_accuracy = _task_map(baseline, "action_correct")
        candidate_accuracy = _task_map(candidate, "action_correct")
        nll_gains = {
            task: baseline_nll[task] - candidate_nll[task]
            for task in baseline_nll
        }
        accuracy_gains = {
            task: candidate_accuracy[task] - baseline_accuracy[task]
            for task in baseline_accuracy
        }
        task_nll_gains.append(nll_gains)
        by_seed[str(seed)] = {
            "baseline_nll": float(np.mean(list(baseline_nll.values()))),
            "candidate_nll": float(np.mean(list(candidate_nll.values()))),
            "nll_gain": float(np.mean(list(nll_gains.values()))),
            "baseline_accuracy": float(np.mean(list(baseline_accuracy.values()))),
            "candidate_accuracy": float(np.mean(list(candidate_accuracy.values()))),
            "accuracy_gain": float(np.mean(list(accuracy_gains.values()))),
        }
    tasks = set(task_nll_gains[0])
    if any(set(values) != tasks for values in task_nll_gains):
        raise ValueError("seed task surfaces differ")
    paired = {
        task: float(np.mean([values[task] for values in task_nll_gains]))
        for task in sorted(tasks)
    }
    seed_nll = [by_seed[str(seed)]["nll_gain"] for seed in seeds]
    seed_accuracy = [by_seed[str(seed)]["accuracy_gain"] for seed in seeds]
    return {
        "baseline_condition": baseline_condition,
        "candidate_condition": candidate_condition,
        "variant": variant,
        "by_seed": by_seed,
        "mean_nll_gain": float(np.mean(seed_nll)),
        "mean_accuracy_gain": float(np.mean(seed_accuracy)),
        "nll_gain_by_seed": dict(zip(map(str, seeds), seed_nll)),
        "accuracy_gain_by_seed": dict(zip(map(str, seeds), seed_accuracy)),
        "paired_task_nll_gains": paired,
        "positive_task_fraction": sum(value > 0.0 for value in paired.values()) / len(paired),
        "paired_bootstrap": paired_bootstrap(
            list(paired.values()),
            draws=int(protocol["uncertainty"]["paired_task_bootstrap_draws"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
        ),
        "paired_sign_test": exact_sign_test(list(paired.values())),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    primary = summary["primary"]
    raw = primary["versus_raw_pooled"]
    ad = primary["versus_agentdojo_only"]
    return "\n".join(
        [
            "# Shared action ontology alignment results",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            "The ontology was derived only from public action schemas. It decomposes candidates into operation, object, effect, communication scope, terminal status, and argument shape.",
            "",
            "| primary comparison | NLL gain | accuracy gain | positive tasks |",
            "|---|---:|---:|---:|",
            f"| ontology residual vs raw pooled | {raw['mean_nll_gain']:+.6f} | {raw['mean_accuracy_gain']:+.6f} | {raw['positive_task_fraction']:.1%} |",
            f"| ontology residual vs AgentDojo-only | {ad['mean_nll_gain']:+.6f} | {ad['mean_accuracy_gain']:+.6f} | {ad['positive_task_fraction']:.1%} |",
            "",
            "## Frozen gate",
            "",
            *[f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in summary["gate_checks"].items()],
            "",
            "Ontology-only results are retained as collision counterevidence. No post-result tuning or rerun was used.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    parent = _read_jsonl(args.parent_predictions)
    if file_sha256(args.parent_predictions) != protocol["source"]["frozen_parent_predictions_sha256"]:
        raise ValueError("parent prediction hash mismatch")
    if not audit["passed"]:
        raise ValueError("ontology preflight failed")
    if run_metrics["neural_training_runs"] != protocol["fixed_budget"]["neural_training_runs"]:
        raise ValueError("ontology run budget is incomplete")
    if run_metrics["predictions_sha256"] != file_sha256(args.predictions):
        raise ValueError("ontology prediction hash mismatch")
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    comparisons = {}
    for mode in protocol["training"]["candidate_modes"]:
        comparisons[mode] = {}
        for variant in protocol["training"]["state_variants"]:
            comparisons[mode][variant] = {
                "versus_raw_pooled": _comparison(
                    baseline_rows=parent,
                    candidate_rows=predictions,
                    seeds=seeds,
                    baseline_condition="agentdojo_plus_auxiliary",
                    candidate_condition=mode,
                    variant=variant,
                    protocol=protocol,
                ),
                "versus_agentdojo_only": _comparison(
                    baseline_rows=parent,
                    candidate_rows=predictions,
                    seeds=seeds,
                    baseline_condition="agentdojo_only",
                    candidate_condition=mode,
                    variant=variant,
                    protocol=protocol,
                ),
            }
    gate = protocol["acceptance_gate"]
    mode = gate["primary_candidate_mode"]
    variant = gate["primary_state_variant"]
    primary = comparisons[mode][variant]
    raw = primary["versus_raw_pooled"]
    ad = primary["versus_agentdojo_only"]
    structured_ad = comparisons[mode]["structured_markov_v3"]["versus_agentdojo_only"]
    threshold_seeds = int(gate["minimum_threshold_positive_seeds"])
    checks = {
        "nll_gain_over_raw_pooled": raw["mean_nll_gain"] >= float(gate["minimum_nll_gain_over_raw_pooled"]),
        "accuracy_gain_over_raw_pooled": raw["mean_accuracy_gain"] >= float(gate["minimum_accuracy_gain_over_raw_pooled"]),
        "nll_seed_replication_over_raw": sum(value >= float(gate["minimum_nll_gain_over_raw_pooled"]) for value in raw["nll_gain_by_seed"].values()) >= threshold_seeds,
        "accuracy_seed_replication_over_raw": sum(value >= float(gate["minimum_accuracy_gain_over_raw_pooled"]) for value in raw["accuracy_gain_by_seed"].values()) >= threshold_seeds,
        "nll_gain_over_agentdojo_only": ad["mean_nll_gain"] >= float(gate["minimum_nll_gain_over_agentdojo_only"]),
        "accuracy_gain_over_agentdojo_only": ad["mean_accuracy_gain"] >= float(gate["minimum_accuracy_gain_over_agentdojo_only"]),
        "nll_seed_replication_over_agentdojo_only": sum(value >= float(gate["minimum_nll_gain_over_agentdojo_only"]) for value in ad["nll_gain_by_seed"].values()) >= threshold_seeds,
        "accuracy_seed_replication_over_agentdojo_only": sum(value >= float(gate["minimum_accuracy_gain_over_agentdojo_only"]) for value in ad["accuracy_gain_by_seed"].values()) >= threshold_seeds,
        "positive_task_fraction_over_raw": raw["positive_task_fraction"] >= float(gate["minimum_positive_task_fraction"]),
        "positive_task_fraction_over_agentdojo_only": ad["positive_task_fraction"] >= float(gate["minimum_positive_task_fraction"]),
        "structured_nll_not_materially_degraded": structured_ad["mean_nll_gain"] >= -float(gate["maximum_structured_nll_degradation_vs_agentdojo_only"]),
        "structured_accuracy_not_materially_degraded": structured_ad["mean_accuracy_gain"] >= -float(gate["maximum_structured_accuracy_degradation_vs_agentdojo_only"]),
    }
    passed = all(checks.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": "GO_SHARED_ACTION_ONTOLOGY_REPAIRS_MULTI_SOURCE_TRANSFER" if passed else "NO_GO_SHARED_ACTION_ONTOLOGY_DOES_NOT_REPAIR_MULTI_SOURCE_TRANSFER",
        "gate_passed": passed,
        "gate_checks": checks,
        "primary": primary,
        "all_comparisons": comparisons,
        "ontology_audit": audit,
        "run": {
            "neural_training_runs": run_metrics["neural_training_runs"],
            "prediction_rows": len(predictions),
            "predictions_sha256": file_sha256(args.predictions),
            "run_metrics_sha256": file_sha256(args.run_metrics),
            "new_llm_calls": 0,
            "new_tool_executions": 0,
            "real_external_endpoint_calls": 0,
            "attack_generation": 0,
            "dreamer_runs": 0,
        },
    }
    _write(args.output, summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
