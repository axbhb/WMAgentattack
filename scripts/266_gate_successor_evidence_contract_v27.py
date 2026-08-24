"""Gate the two independent v27 successor-evidence builds."""

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
        raise ValueError("v27 protocol is not frozen")
    audit = json.loads(a.audit_a.read_text(encoding="utf-8"))
    other = json.loads(a.audit_b.read_text(encoding="utf-8"))
    t = protocol["acceptance_thresholds"]
    fold_count_values = {
        int(row["fold"]): set(row["available_matched_count_values"])
        for row in audit["fold_diagnostics"]
    }
    checks = {
        "independent_datasets_byte_identical": a.dataset_a.read_bytes() == a.dataset_b.read_bytes(),
        "independent_audits_byte_identical": a.audit_a.read_bytes() == a.audit_b.read_bytes(),
        "exact_confirmation_rows": audit["confirmation_rows"] == int(t["required_confirmation_rows"]),
        "exact_support_rows": audit["support_rows"] == int(t["required_support_rows"]),
        "exact_confirmation_tasks": audit["tasks"] == int(t["required_confirmation_tasks"]),
        "exact_support_tasks": audit["support_tasks"] == int(t["required_support_tasks"]),
        "zero_missing_next_states": not audit["missing_next_state_refs"],
        "all_v20_effects_recovered": audit["full_render_matches"] == int(t["required_confirmation_rows"]),
        "all_v21_hard_effects_recovered": audit["hard_render_matches"] == int(t["required_confirmation_rows"]),
        "all_support_rows_renderable": audit["support_rendered_rows"] == int(t["required_support_rows"]),
        "zero_support_confirmation_task_overlap": not audit["support_confirmation_task_overlap"],
        "zero_semantic_input_leakage": not audit["semantic_input_leakage"],
        "zero_model_input_key_leakage": not audit["model_input_key_leakage"],
        "zero_goal_pointer_errors": not audit["goal_pointer_errors"],
        "zero_record_binding_errors": not audit["record_binding_errors"],
        "all_transitions_adjacent": not audit["adjacency_errors"],
        "entity_relation_coverage": audit["relation_coverage"]["entity"]
        >= float(t["minimum_entity_relation_coverage"]),
        "attribute_relation_coverage": audit["relation_coverage"]["attribute"]
        >= float(t["minimum_attribute_relation_coverage"]),
        "operation_coverage": audit["operation_coverage"] >= float(t["minimum_operation_coverage"]),
        "support_contains_count3": int(audit["support_matched_count_values"].get("3", 0))
        >= int(t["minimum_support_count3_rows"]),
        "every_fold_has_required_count_values": all(
            set(t["required_matched_count_values"]) <= fold_count_values.get(fold, set())
            for fold in range(3)
        ),
        "structured_targets_exclude_composites": audit["structured_targets_exclude_composite_effect_tokens"],
        "audit_replicates": audit == other,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.successor_evidence_identifiability_gate.v27",
        "decision": "GO_BOUND_SUCCESSOR_MODEL_V27" if passed else "NO_GO_BOUND_SUCCESSOR_DATA_V27",
        "gate_checks": checks,
        "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "failed_clauses": [key for key, value in checks.items() if not value],
        "dataset_sha256": sha256(a.dataset_a),
        "audit_sha256": sha256(a.audit_a),
        "audit": audit,
        "authorization": {
            "one_small_bound_successor_model_comparison": passed,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_generation": False,
            "planner_or_dreamer": False,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__":
    main()
