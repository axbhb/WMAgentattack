"""Compare source-residual adapters with frozen pooled and AD-only controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.multisource_suitability import file_sha256

SPEC = importlib.util.spec_from_file_location(
    "ontology_summary", ROOT / "scripts" / "186_summarize_shared_action_ontology.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    candidate = SUMMARY._read_jsonl(args.predictions)
    parent = SUMMARY._read_jsonl(args.parent_predictions)
    if file_sha256(args.parent_predictions) != protocol["source"]["frozen_parent_predictions_sha256"]:
        raise ValueError("parent prediction hash mismatch")
    if file_sha256(args.predictions) != metrics["predictions_sha256"]:
        raise ValueError("adapter prediction hash mismatch")
    if metrics["neural_training_runs"] != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("fixed run budget incomplete")
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]
    candidate_condition = str(protocol["training"]["condition"])
    comparisons = {}
    for variant in protocol["training"]["variants"]:
        comparisons[variant] = {
            "versus_raw_pooled": SUMMARY._comparison(
                baseline_rows=parent,
                candidate_rows=candidate,
                seeds=seeds,
                baseline_condition="agentdojo_plus_auxiliary",
                candidate_condition=candidate_condition,
                variant=variant,
                protocol=protocol,
            ),
            "versus_agentdojo_only": SUMMARY._comparison(
                baseline_rows=parent,
                candidate_rows=candidate,
                seeds=seeds,
                baseline_condition="agentdojo_only",
                candidate_condition=candidate_condition,
                variant=variant,
                protocol=protocol,
            ),
        }
    gate = protocol["acceptance_gate"]
    primary = comparisons[gate["primary_variant"]]
    raw = primary["versus_raw_pooled"]
    ad = primary["versus_agentdojo_only"]
    structured_raw = comparisons["structured_markov_v3"]["versus_raw_pooled"]
    minimum_seeds = int(gate["minimum_threshold_positive_seeds"])
    checks = {
        "nll_gain_over_raw_pooled": raw["mean_nll_gain"] >= float(gate["minimum_nll_gain_over_raw_pooled"]),
        "accuracy_gain_over_raw_pooled": raw["mean_accuracy_gain"] >= float(gate["minimum_accuracy_gain_over_raw_pooled"]),
        "nll_seed_replication": sum(value >= float(gate["minimum_nll_gain_over_raw_pooled"]) for value in raw["nll_gain_by_seed"].values()) >= minimum_seeds,
        "accuracy_seed_replication": sum(value >= float(gate["minimum_accuracy_gain_over_raw_pooled"]) for value in raw["accuracy_gain_by_seed"].values()) >= minimum_seeds,
        "positive_task_fraction": raw["positive_task_fraction"] >= float(gate["minimum_positive_task_fraction"]),
        "nll_noninferior_to_agentdojo_only": ad["mean_nll_gain"] >= -float(gate["maximum_nll_degradation_vs_agentdojo_only"]),
        "accuracy_noninferior_to_agentdojo_only": ad["mean_accuracy_gain"] >= -float(gate["maximum_accuracy_degradation_vs_agentdojo_only"]),
        "structured_nll_not_degraded": structured_raw["mean_nll_gain"] >= -float(gate["maximum_structured_nll_degradation_vs_raw_pooled"]),
        "structured_accuracy_not_degraded": structured_raw["mean_accuracy_gain"] >= -float(gate["maximum_structured_accuracy_degradation_vs_raw_pooled"]),
        "all_predictions_legal": all(row["legal_prediction"] == 1.0 for row in candidate),
    }
    passed = all(checks.values())
    head_only = protocol["protocol_id"] == "0814_source_specific_action_head_v1"
    output = {
        "protocol_id": protocol["protocol_id"],
        "decision": (("GO_SOURCE_SPECIFIC_HEAD_REPAIRS_NEGATIVE_TRANSFER" if passed else "NO_GO_SOURCE_SPECIFIC_HEAD_DOES_NOT_REPAIR_NEGATIVE_TRANSFER") if head_only else ("GO_SOURCE_RESIDUAL_ADAPTER_REPAIRS_NEGATIVE_TRANSFER" if passed else "NO_GO_SOURCE_RESIDUAL_ADAPTER_DOES_NOT_REPAIR_NEGATIVE_TRANSFER")),
        "gate_passed": passed,
        "gate_checks": checks,
        "primary": primary,
        "comparisons": comparisons,
        "run": {
            "neural_training_runs": metrics["neural_training_runs"],
            "prediction_rows": len(candidate),
            "predictions_sha256": file_sha256(args.predictions),
            "run_metrics_sha256": file_sha256(args.run_metrics),
            "new_llm_calls": 0,
            "new_tool_executions": 0,
            "real_external_endpoint_calls": 0,
            "new_attack_generation": 0,
            "dreamer_runs": 0
        }
    }
    SUMMARY._write(args.output, output)
    lines = [
        "# Source residual adapter results", "", f"Decision: `{output['decision']}`", "",
        "| comparison | NLL gain | accuracy gain | positive tasks |", "|---|---:|---:|---:|",
        f"| adapter vs raw pooled | {raw['mean_nll_gain']:+.6f} | {raw['mean_accuracy_gain']:+.6f} | {raw['positive_task_fraction']:.1%} |",
        f"| adapter vs AgentDojo-only | {ad['mean_nll_gain']:+.6f} | {ad['mean_accuracy_gain']:+.6f} | {ad['positive_task_fraction']:.1%} |",
        "", "## Frozen gate", "",
        *[f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in checks.items()], ""
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
