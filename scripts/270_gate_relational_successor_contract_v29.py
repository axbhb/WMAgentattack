"""Gate the two independent v29 data-contract builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--dataset-a", type=Path, required=True)
    p.add_argument("--audit-a", type=Path, required=True)
    p.add_argument("--dataset-b", type=Path, required=True)
    p.add_argument("--audit-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v29 protocol is not frozen")
    audit = json.loads(a.audit_a.read_text(encoding="utf-8"))
    other = json.loads(a.audit_b.read_text(encoding="utf-8"))
    t = protocol["acceptance_thresholds"]
    coverage = [
        metrics
        for suite in audit["split_candidate_coverage"].values()
        for metrics in suite.values()
    ]
    checks = {
        "independent_datasets_byte_identical": a.dataset_a.read_bytes() == a.dataset_b.read_bytes(),
        "independent_audits_byte_identical": a.audit_a.read_bytes() == a.audit_b.read_bytes(),
        "exact_confirmation_rows": audit["confirmation_rows"] == t["required_confirmation_rows"],
        "exact_support_rows": audit["support_rows"] == t["required_support_rows"],
        "exact_confirmation_tasks": audit["confirmation_tasks"] == t["required_confirmation_tasks"],
        "exact_support_tasks": audit["support_tasks"] == t["required_support_tasks"],
        "zero_support_confirmation_task_overlap": not audit["support_confirmation_task_overlap"],
        "zero_record_goal_relation_errors": not audit["record_goal_relation_errors"],
        "positive_record_goal_relations_exist": audit["records_with_goal_links"] >= t["minimum_records_with_goal_links"],
        "multi_goal_record_exists": audit["maximum_goal_links_per_record"] >= t["minimum_maximum_goal_links_per_record"],
        "static_tool_inventory_floor": audit["static_candidate_tools"] >= t["minimum_static_candidate_tools"],
        "webpage_candidate_present": audit["webpage_candidate_present"],
        "all_split_unique_candidate_coverage": all(row["unique_coverage"] >= t["minimum_static_candidate_coverage"] for row in coverage),
        "all_split_occurrence_candidate_coverage": all(row["occurrence_coverage"] >= t["minimum_static_candidate_coverage"] for row in coverage),
        "zero_missing_static_signatures": all(not row["missing_unique_signatures"] for row in coverage),
        "zero_semantic_input_leakage": not audit["semantic_input_leakage"],
        "zero_model_target_leakage": not audit["model_target_leakage"],
        "static_registry_is_outcome_blind": not audit["static_registry_outcome_labels_present"],
        "audit_replicates": audit == other,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.relational_successor_data_gate.v29",
        "decision": "GO_RELATIONAL_SUCCESSOR_MODEL_V29" if passed else "NO_GO_RELATIONAL_SUCCESSOR_DATA_V29",
        "gate_checks": checks,
        "passed_clauses": sum(checks.values()), "total_clauses": len(checks),
        "failed_clauses": [name for name, value in checks.items() if not value],
        "dataset_sha256": sha256(a.dataset_a), "audit_sha256": sha256(a.audit_a),
        "audit": audit,
        "authorization": {
            "one_small_relational_successor_model_comparison": passed,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_generation": False,
            "planner_or_dreamer": False
        }
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__":
    main()
