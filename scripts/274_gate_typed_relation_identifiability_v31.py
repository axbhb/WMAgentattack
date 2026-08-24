"""Apply the frozen v31 typed-relation representation and support gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--build-a", type=Path, required=True)
    parser.add_argument("--build-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    threshold = protocol["acceptance_thresholds"]
    dataset_a, dataset_b = args.build_a / "dataset.json", args.build_b / "dataset.json"
    audit_a_path, audit_b_path = args.build_a / "audit.json", args.build_b / "audit.json"
    audit = json.loads(audit_a_path.read_text(encoding="utf-8"))
    per_fold = audit["per_fold"]

    checks = {
        "byte_identical_datasets": dataset_a.read_bytes() == dataset_b.read_bytes(),
        "byte_identical_audits": audit_a_path.read_bytes() == audit_b_path.read_bytes(),
        "confirmation_row_count": audit["confirmation_rows"] == threshold["required_confirmation_rows"],
        "positive_edge_reconstruction_count": audit["positive_relation_edges"] == threshold["required_positive_relation_edges"],
        "zero_gold_reconstruction_errors": not audit["gold_reconstruction_errors"],
        "positive_structural_coverage": audit["positive_structural_coverage"] >= threshold["minimum_positive_structural_coverage"],
        "positive_typed_unit_coverage": audit["positive_typed_unit_coverage"] >= threshold["minimum_positive_typed_unit_coverage"],
        "hard_negatives_per_positive": audit["minimum_hard_negatives_per_positive"] >= threshold["minimum_hard_negatives_per_positive"],
        "combined_pair_accuracy": audit["combined_pair_accuracy"] >= threshold["minimum_combined_pair_accuracy"],
        "gain_over_goal_blind_record_only": (
            audit["combined_pair_accuracy"] - audit["goal_blind_record_only_pair_accuracy"]
            >= threshold["minimum_pair_accuracy_gain_over_record_only"]
        ),
        "all_fold_pair_accuracy": all(
            per_fold[str(index)]["combined_pair_accuracy"] >= threshold["minimum_per_fold_pair_accuracy"]
            for index in range(3)
        ),
        "semantic_vectors_finite": audit["semantic_vectors_finite"],
        "semantic_unit_norm": audit["semantic_unit_norm_max_error"] <= threshold["maximum_semantic_unit_norm_error"],
        "zero_forbidden_output_keys": not audit["forbidden_output_keys_present"],
        "zero_real_endpoints": audit["real_external_endpoint_calls"] == 0,
        "zero_victim_llm_calls": audit["victim_llm_calls"] == 0,
        "zero_sandbox_tool_calls": audit["sandbox_tool_calls"] == 0,
        "zero_model_fits": audit["model_fits"] == 0,
    }
    representation_keys = list(checks)
    representation_ready = all(checks[key] for key in representation_keys)
    support_check = min(audit["positive_edges_by_fold"]) >= threshold["minimum_positive_relation_edges_per_fold_for_model_fit"]
    checks["positive_relation_support_per_fold"] = support_check

    if representation_ready and support_check:
        decision = "GO_SMALL_TYPED_RELATION_MODEL_V31"
    elif representation_ready:
        decision = "GO_TARGETED_RELATION_SUPPORT_PILOT_V31"
    else:
        decision = "NO_GO_TYPED_RELATION_REPRESENTATION_V31"
    payload = {
        "schema_version": "wmagentattack.typed_relation_identifiability_gate.v31",
        "decision": decision,
        "representation_ready": representation_ready,
        "model_fit_data_ready": representation_ready and support_check,
        "gate_checks": checks,
        "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "failed_clauses": [key for key, value in checks.items() if not value],
        "metrics": {
            key: audit[key] for key in (
                "confirmation_rows", "positive_relation_edges", "positive_edges_by_fold",
                "positive_structural_coverage", "positive_typed_unit_coverage",
                "hard_negative_comparisons", "minimum_hard_negatives_per_positive",
                "combined_pair_accuracy", "semantic_pair_accuracy",
                "goal_blind_record_only_pair_accuracy", "combined_pair_margin",
                "semantic_text_count", "semantic_unit_norm_max_error", "per_fold",
                "relation_type_counts_on_positives",
            )
        },
        "artifact_sha256": {
            "dataset": sha256(dataset_a),
            "audit": sha256(audit_a_path),
        },
        "authorization": {
            "one_24_episode_clean_relation_support_pilot": decision == "GO_TARGETED_RELATION_SUPPORT_PILOT_V31",
            "one_small_typed_relation_model_comparison": decision == "GO_SMALL_TYPED_RELATION_MODEL_V31",
            "attack_generation": False,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "planner_or_dreamer": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__":
    main()
