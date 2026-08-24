"""Apply the preregistered v28 bound successor-record gate."""

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
    values = []
    for row in rows:
        if occurrence is not None and int(row.get(occurrence, 0)) == 0:
            continue
        if row.get(key) is not None:
            values.append(float(row[key]))
    return float(np.mean(values)) if values else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: mean(rows, key, occurrence)
        for key, occurrence in (
            ("task_macro_bce", None),
            ("positive_task_macro_nll", None),
            ("positive_task_macro_recall", None),
            ("seen_positive_nll", None),
            ("seen_positive_recall", None),
            ("unseen_positive_nll", "unseen_positive_occurrences"),
            ("unseen_positive_recall", "unseen_positive_occurrences"),
            ("unseen_false_positive_rate", None),
            ("unseen_precision", None),
            ("matched_count3_recall", "matched_count3_occurrences"),
            ("focused_unseen_recall", "focused_unseen_occurrences"),
            ("query_read_positive_recall", None),
            ("execution_brier", None),
            ("pair_assignment_accuracy", None),
            ("rollout_bce", None),
            ("rollout_positive_nll", None),
            ("parameter_count", None),
            ("bound_record_precision", None),
            ("bound_record_recall", None),
            ("bound_record_f1", None),
            ("bound_record_exact_set_accuracy", None),
            ("goal_pointer_f1", None),
            ("goal_pointer_precision", None),
            ("goal_pointer_recall", None),
            ("heldout_exact_record_candidate_coverage", None),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--v26-metrics", type=Path, required=True)
    parser.add_argument("--v27-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    v26 = json.loads(args.v26_metrics.read_text(encoding="utf-8"))
    v27_gate = json.loads(args.v27_gate.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v28 protocol is not frozen")
    if sha256(args.v26_metrics) != protocol["external_controls"]["v26_metrics_sha256"]:
        raise ValueError("v26 external control hash mismatch")
    if sha256(args.v27_gate) != protocol["external_controls"]["v27_gate_sha256"]:
        raise ValueError("v27 gate hash mismatch")
    thresholds = protocol["acceptance_thresholds"]
    fixed_rows = selected(metrics, "fixed_v21", "task_disjoint")
    bound_rows = selected(metrics, "bound_successor_records_v28", "task_disjoint")
    v26_fixed_rows = selected(v26, "fixed_v21", "task_disjoint")
    v26_support_rows = selected(v26, "factorized_ordinal_support_v26", "task_disjoint")
    fixed = summary(fixed_rows)
    bound = summary(bound_rows)
    v26_fixed = summary(v26_fixed_rows)
    v26_support = summary(v26_support_rows)
    reproduction_fields = (
        "task_macro_bce", "positive_task_macro_nll", "positive_task_macro_recall",
        "seen_positive_recall", "rollout_bce", "query_read_positive_recall",
        "pair_assignment_accuracy",
    )
    reproduction_errors = {
        key: abs(float(fixed[key]) - float(v26_fixed[key]))
        for key in reproduction_fields
        if fixed[key] is not None and v26_fixed[key] is not None
    }
    maximum_reproduction_error = max(reproduction_errors.values(), default=float("inf"))
    v26_cells = {
        (row["fold_marker"], row["seed"]): row for row in v26_support_rows
        if int(row["unseen_positive_occurrences"]) > 0
    }
    gain_cells = []
    for row in bound_rows:
        key = (row["fold_marker"], row["seed"])
        if int(row["unseen_positive_occurrences"]) == 0 or key not in v26_cells:
            continue
        gain_cells.append({
            "fold": key[0], "seed": key[1],
            "bound_recall": row["unseen_positive_recall"],
            "v26_recall": v26_cells[key]["unseen_positive_recall"],
            "gain": float(row["unseen_positive_recall"] - v26_cells[key]["unseen_positive_recall"]),
        })
    positive_gain_cells = sum(value["gain"] > 0 for value in gain_cells)
    diagnostics = {
        suite: summary(selected(metrics, "bound_successor_records_v28", suite))
        for suite in ("tool_family_heldout", "source_heldout")
    }
    checks = {
        "complete_fixed_budget": metrics.get("completed_model_fits") == int(protocol["fixed_budget"]["model_fits"])
        and metrics.get("completed_metric_rows") == int(protocol["fixed_budget"]["metric_rows"])
        and metrics.get("runtime_failures") == 0,
        "v27_gate_was_go": v27_gate.get("decision") == "GO_BOUND_SUCCESSOR_MODEL_V27",
        "structured_loader_hash_preserved": metrics.get("structured_dataset_sha256") == protocol["data"]["structured_dataset_sha256"],
        "hard_loader_hash_preserved": metrics.get("hard_dataset_sha256") == protocol["data"]["hard_dataset_sha256"],
        "fixed_v21_exact_reproduction": maximum_reproduction_error <= float(thresholds["maximum_fixed_reproduction_absolute_error"]),
        "exact_record_candidate_coverage": bound["heldout_exact_record_candidate_coverage"] >= float(thresholds["minimum_exact_record_candidate_coverage"]),
        "bound_record_recall_floor": bound["bound_record_recall"] >= float(thresholds["minimum_bound_record_recall"]),
        "bound_record_f1_floor": bound["bound_record_f1"] >= float(thresholds["minimum_bound_record_f1"]),
        "bound_record_exact_set_floor": bound["bound_record_exact_set_accuracy"] >= float(thresholds["minimum_bound_record_exact_set_accuracy"]),
        "goal_pointer_f1_floor": bound["goal_pointer_f1"] >= float(thresholds["minimum_goal_pointer_f1"]),
        "task_unseen_recall_floor": bound["unseen_positive_recall"] >= float(thresholds["minimum_task_unseen_recall"]),
        "task_unseen_recall_gain_vs_v26": bound["unseen_positive_recall"] - v26_support["unseen_positive_recall"] >= float(thresholds["minimum_recall_gain_vs_v26"]),
        "gain_cell_stability": positive_gain_cells >= int(thresholds["minimum_positive_gain_cells"]),
        "unseen_nll_ceiling": bound["unseen_positive_nll"] <= float(thresholds["maximum_task_unseen_nll"]),
        "unseen_precision_floor": bound["unseen_precision"] >= float(thresholds["minimum_unseen_precision"]),
        "unseen_fpr_ceiling": bound["unseen_false_positive_rate"] <= float(thresholds["maximum_unseen_false_positive_rate"]),
        "matched_count3_recall_floor": bound["matched_count3_recall"] >= float(thresholds["minimum_matched_count3_recall"]),
        "focused_unseen_recall_floor": bound["focused_unseen_recall"] >= float(thresholds["minimum_focused_unseen_recall"]),
        "seen_recall_noninferiority": bound["seen_positive_recall"] >= fixed["seen_positive_recall"] - float(thresholds["seen_recall_noninferiority_margin"]),
        "one_step_bce_noninferiority": bound["task_macro_bce"] <= fixed["task_macro_bce"] + float(thresholds["one_step_bce_noninferiority_margin"]),
        "rollout_bce_noninferiority": bound["rollout_bce"] <= fixed["rollout_bce"] + float(thresholds["rollout_bce_noninferiority_margin"]),
        "query_read_recall_noninferiority": bound["query_read_positive_recall"] >= fixed["query_read_positive_recall"] - float(thresholds["query_read_recall_noninferiority_margin"]),
        "pair_accuracy_noninferiority": bound["pair_assignment_accuracy"] >= fixed["pair_assignment_accuracy"] - float(thresholds["pair_accuracy_noninferiority_margin"]),
        "tool_family_unseen_recall_floor": diagnostics["tool_family_heldout"]["unseen_positive_recall"] >= float(thresholds["minimum_tool_family_unseen_recall"]),
        "source_unseen_recall_floor": diagnostics["source_heldout"]["unseen_positive_recall"] >= float(thresholds["minimum_source_unseen_recall"]),
        "parameter_ceiling": bound["parameter_count"] <= float(thresholds["maximum_combined_parameters"]),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.bound_successor_records_gate.v28",
        "decision": "GO_BOUND_SUCCESSOR_RECORDS_V28" if passed else "NO_GO_BOUND_SUCCESSOR_RECORDS_V28",
        "gate_checks": checks, "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "failed_clauses": [key for key, value in checks.items() if not value],
        "task_disjoint": {
            "fixed_v21": fixed, "bound_successor_records_v28": bound,
            "v26_support_external": v26_support,
            "recall_gain_vs_v26": bound["unseen_positive_recall"] - v26_support["unseen_positive_recall"],
        },
        "fixed_reproduction_errors": reproduction_errors,
        "maximum_fixed_reproduction_error": maximum_reproduction_error,
        "gain_cells": gain_cells, "positive_gain_cells": positive_gain_cells,
        "diagnostics": diagnostics,
        "authorization": {
            "retain_v28_as_open_vocabulary_effect_branch": passed,
            "run_frozen_96_episode_data_smoke": passed,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_generation": False,
            "planner_or_dreamer": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__":
    main()
