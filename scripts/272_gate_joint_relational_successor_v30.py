"""Apply the preregistered v30 joint relational successor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(payload: dict[str, Any], arm: str, suite: str) -> list[dict[str, Any]]:
    return [row for row in payload["runs"] if row["arm"] == arm and row["split_suite"] == suite]


def mean(rows: list[dict[str, Any]], key: str, occurrence: str | None = None) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and (occurrence is None or int(row.get(occurrence, 0)) > 0)]
    return float(np.mean(values)) if values else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: mean(rows, key, occurrence) for key, occurrence in (
        ("task_macro_bce", None), ("positive_task_macro_nll", None), ("positive_task_macro_recall", None),
        ("seen_positive_nll", None), ("seen_positive_recall", None),
        ("unseen_positive_nll", "unseen_positive_occurrences"), ("unseen_positive_recall", "unseen_positive_occurrences"),
        ("unseen_false_positive_rate", None), ("unseen_precision", None),
        ("matched_count3_recall", "matched_count3_occurrences"), ("focused_unseen_recall", "focused_unseen_occurrences"),
        ("query_read_positive_recall", None), ("execution_brier", None), ("pair_assignment_accuracy", None),
        ("rollout_bce", None), ("rollout_positive_nll", None), ("parameter_count", None),
        ("bound_record_precision", None), ("bound_record_recall", None), ("bound_record_f1", None),
        ("bound_record_exact_set_accuracy", None), ("record_goal_relation_precision", None),
        ("record_goal_relation_recall", None), ("record_goal_relation_f1", None),
        ("goal_pointer_f1", None), ("goal_pointer_precision", None), ("goal_pointer_recall", None),
        ("heldout_exact_record_candidate_coverage", None),
    )}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True); p.add_argument("--v28-metrics", type=Path, required=True)
    p.add_argument("--v29-gate", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    protocol = json.loads(a.protocol.read_text()); metrics = json.loads(a.metrics.read_text())
    v28 = json.loads(a.v28_metrics.read_text()); v29 = json.loads(a.v29_gate.read_text())
    if protocol["status"] != "preregistered_before_results": raise ValueError("v30 protocol is not frozen")
    if sha256(a.v28_metrics) != protocol["external_controls"]["v28_metrics_sha256"]: raise ValueError("v28 control hash mismatch")
    if sha256(a.v29_gate) != protocol["external_controls"]["v29_gate_sha256"]: raise ValueError("v29 gate hash mismatch")
    t = protocol["acceptance_thresholds"]
    fixed_rows = selected(metrics, "fixed_v21", "task_disjoint")
    joint_rows = selected(metrics, "joint_relational_successor_v30", "task_disjoint")
    old_fixed_rows = selected(v28, "fixed_v21", "task_disjoint")
    old_model_rows = selected(v28, "bound_successor_records_v28", "task_disjoint")
    fixed, joint, old_fixed, old_model = map(summary, (fixed_rows, joint_rows, old_fixed_rows, old_model_rows))
    fields = ("task_macro_bce", "positive_task_macro_nll", "positive_task_macro_recall", "seen_positive_recall", "rollout_bce", "query_read_positive_recall", "pair_assignment_accuracy")
    reproduction = {key: abs(float(fixed[key]) - float(old_fixed[key])) for key in fields if fixed[key] is not None and old_fixed[key] is not None}
    max_reproduction = max(reproduction.values(), default=float("inf"))
    old_cells = {(row["fold_marker"], row["seed"]): row for row in old_model_rows if int(row["unseen_positive_occurrences"]) > 0}
    gain_cells = []
    for row in joint_rows:
        key = (row["fold_marker"], row["seed"])
        if int(row["unseen_positive_occurrences"]) == 0 or key not in old_cells: continue
        gain_cells.append({"fold": key[0], "seed": key[1], "joint_recall": row["unseen_positive_recall"], "v28_recall": old_cells[key]["unseen_positive_recall"], "gain": float(row["unseen_positive_recall"] - old_cells[key]["unseen_positive_recall"])})
    positive_gain_cells = sum(row["gain"] > 0 for row in gain_cells)
    diagnostics = {suite: summary(selected(metrics, "joint_relational_successor_v30", suite)) for suite in ("tool_family_heldout", "source_heldout")}
    checks = {
        "complete_fixed_budget": metrics.get("completed_model_fits") == protocol["fixed_budget"]["model_fits"] and metrics.get("completed_metric_rows") == protocol["fixed_budget"]["metric_rows"] and metrics.get("runtime_failures") == 0,
        "v29_data_gate_was_go": v29.get("decision") == "GO_RELATIONAL_SUCCESSOR_MODEL_V29",
        "hard_loader_hash_preserved": metrics.get("hard_dataset_sha256") == protocol["data"]["hard_dataset_sha256"],
        "structured_loader_hash_preserved": metrics.get("structured_dataset_sha256") == protocol["data"]["structured_dataset_sha256"],
        "relational_loader_hash_preserved": metrics.get("relational_dataset_sha256") == protocol["data"]["relational_dataset_sha256"],
        "fixed_v21_reproduction": max_reproduction <= t["maximum_fixed_reproduction_absolute_error"],
        "exact_record_candidate_coverage": joint["heldout_exact_record_candidate_coverage"] >= t["minimum_exact_record_candidate_coverage"],
        "record_recall_floor": joint["bound_record_recall"] >= t["minimum_record_recall"],
        "record_f1_floor": joint["bound_record_f1"] >= t["minimum_record_f1"],
        "record_exact_set_floor": joint["bound_record_exact_set_accuracy"] >= t["minimum_record_exact_set_accuracy"],
        "relation_f1_floor": joint["record_goal_relation_f1"] >= t["minimum_relation_f1"],
        "relation_recall_floor": joint["record_goal_relation_recall"] >= t["minimum_relation_recall"],
        "goal_pointer_f1_floor": joint["goal_pointer_f1"] >= t["minimum_goal_pointer_f1"],
        "goal_pointer_gain_vs_v28": joint["goal_pointer_f1"] - old_model["goal_pointer_f1"] >= t["minimum_goal_pointer_f1_gain_vs_v28"],
        "task_unseen_recall_floor": joint["unseen_positive_recall"] >= t["minimum_task_unseen_recall"],
        "task_unseen_recall_gain_vs_v28": joint["unseen_positive_recall"] - old_model["unseen_positive_recall"] >= t["minimum_recall_gain_vs_v28"],
        "gain_cell_stability": positive_gain_cells >= t["minimum_positive_gain_cells"],
        "unseen_nll_ceiling": joint["unseen_positive_nll"] <= t["maximum_task_unseen_nll"],
        "unseen_precision_floor": joint["unseen_precision"] >= t["minimum_unseen_precision"],
        "unseen_fpr_ceiling": joint["unseen_false_positive_rate"] <= t["maximum_unseen_false_positive_rate"],
        "matched_count3_recall_floor": joint["matched_count3_recall"] >= t["minimum_matched_count3_recall"],
        "focused_unseen_recall_floor": joint["focused_unseen_recall"] >= t["minimum_focused_unseen_recall"],
        "seen_recall_noninferiority": joint["seen_positive_recall"] >= fixed["seen_positive_recall"] - t["seen_recall_noninferiority_margin"],
        "one_step_bce_noninferiority": joint["task_macro_bce"] <= fixed["task_macro_bce"] + t["one_step_bce_noninferiority_margin"],
        "rollout_bce_noninferiority": joint["rollout_bce"] <= fixed["rollout_bce"] + t["rollout_bce_noninferiority_margin"],
        "query_read_recall_noninferiority": joint["query_read_positive_recall"] >= fixed["query_read_positive_recall"] - t["query_read_recall_noninferiority_margin"],
        "pair_accuracy_noninferiority": joint["pair_assignment_accuracy"] >= fixed["pair_assignment_accuracy"] - t["pair_accuracy_noninferiority_margin"],
        "tool_family_unseen_recall_floor": diagnostics["tool_family_heldout"]["unseen_positive_recall"] >= t["minimum_tool_family_unseen_recall"],
        "source_unseen_recall_floor": diagnostics["source_heldout"]["unseen_positive_recall"] >= t["minimum_source_unseen_recall"],
        "parameter_ceiling": joint["parameter_count"] <= t["maximum_combined_parameters"],
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.joint_relational_successor_gate.v30",
        "decision": "GO_JOINT_RELATIONAL_SUCCESSOR_V30" if passed else "NO_GO_JOINT_RELATIONAL_SUCCESSOR_V30",
        "gate_checks": checks, "passed_clauses": sum(checks.values()), "total_clauses": len(checks),
        "failed_clauses": [key for key, value in checks.items() if not value],
        "task_disjoint": {"fixed_v21": fixed, "joint_relational_successor_v30": joint, "v28_external": old_model,
            "recall_gain_vs_v28": joint["unseen_positive_recall"] - old_model["unseen_positive_recall"],
            "pointer_f1_gain_vs_v28": joint["goal_pointer_f1"] - old_model["goal_pointer_f1"]},
        "fixed_reproduction_errors": reproduction, "maximum_fixed_reproduction_error": max_reproduction,
        "gain_cells": gain_cells, "positive_gain_cells": positive_gain_cells, "diagnostics": diagnostics,
        "authorization": {"retain_v30_as_primary_open_vocabulary_branch": passed, "one_frozen_96_episode_clean_data_smoke": passed,
            "large_scale_generation": False, "large_world_model_training": False, "attack_generation": False, "planner_or_dreamer": False},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__": main()
