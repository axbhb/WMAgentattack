"""Apply the preregistered v26 support-conditioned model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(payload: dict[str, Any], arm: str, suite: str) -> list[dict[str, Any]]:
    return [row for row in payload["runs"] if row["arm"] == arm and row["split_suite"] == suite]


def mean(rows: list[dict[str, Any]], key: str, occurrence_key: str | None = None) -> float | None:
    values = []
    for row in rows:
        if occurrence_key is not None and int(row.get(occurrence_key, 0)) == 0:
            continue
        value = row.get(key)
        if value is not None:
            values.append(float(value))
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
        )
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--v23-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(a.metrics.read_text(encoding="utf-8"))
    v23 = json.loads(a.v23_metrics.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v26 protocol is not frozen")
    if sha256(a.v23_metrics) != protocol["external_controls"]["v23_metrics_sha256"]:
        raise ValueError("v23 control hash mismatch")
    thresholds = protocol["acceptance_thresholds"]
    expected_fits = int(protocol["fixed_budget"]["model_fits"])
    expected_rows = int(protocol["fixed_budget"]["metric_rows"])
    fixed_rows = selected(metrics, "fixed_v21", "task_disjoint")
    no_support_rows = selected(metrics, "factorized_ordinal_no_support_v26", "task_disjoint")
    support_rows = selected(metrics, "factorized_ordinal_support_v26", "task_disjoint")
    v23_fixed_rows = selected(v23, "fixed_v21", "task_disjoint")
    v23_raw_rows = selected(v23, "hybrid_e5_raw_v23", "task_disjoint")
    fixed = summary(fixed_rows)
    no_support = summary(no_support_rows)
    support = summary(support_rows)
    v23_fixed = summary(v23_fixed_rows)
    v23_raw = summary(v23_raw_rows)
    reproduction_fields = (
        "task_macro_bce", "positive_task_macro_nll", "positive_task_macro_recall",
        "seen_positive_recall", "rollout_bce", "query_read_positive_recall",
        "pair_assignment_accuracy",
    )
    reproduction_errors = {
        key: abs(float(fixed[key]) - float(v23_fixed[key]))
        for key in reproduction_fields if fixed[key] is not None and v23_fixed[key] is not None
    }
    max_reproduction_error = max(reproduction_errors.values(), default=float("inf"))

    indexed_no_support = {
        (row["fold_marker"], row["seed"]): row for row in no_support_rows
        if int(row["unseen_positive_occurrences"]) > 0
    }
    gain_cells = []
    for row in support_rows:
        key = (row["fold_marker"], row["seed"])
        if int(row["unseen_positive_occurrences"]) == 0 or key not in indexed_no_support:
            continue
        gain_cells.append({
            "fold": key[0],
            "seed": key[1],
            "support_recall": row["unseen_positive_recall"],
            "no_support_recall": indexed_no_support[key]["unseen_positive_recall"],
            "gain": float(row["unseen_positive_recall"] - indexed_no_support[key]["unseen_positive_recall"]),
        })
    positive_gain_cells = sum(row["gain"] > 0 for row in gain_cells)

    diagnostics = {}
    for suite in ("tool_family_heldout", "source_heldout"):
        diagnostics[suite] = summary(selected(metrics, "factorized_ordinal_support_v26", suite))

    checks = {
        "complete_fixed_budget": metrics.get("completed_model_fits") == expected_fits
        and metrics.get("completed_metric_rows") == expected_rows
        and metrics.get("runtime_failures") == 0,
        "fixed_v21_exact_reproduction": max_reproduction_error
        <= float(thresholds["maximum_fixed_reproduction_absolute_error"]),
        "task_unseen_recall_floor": support["unseen_positive_recall"]
        >= float(thresholds["minimum_task_unseen_recall"]),
        "task_unseen_recall_gain_vs_v23_raw": support["unseen_positive_recall"] - v23_raw["unseen_positive_recall"]
        >= float(thresholds["minimum_task_unseen_recall_gain_vs_v23_raw"]),
        "support_gain_vs_no_support": support["unseen_positive_recall"] - no_support["unseen_positive_recall"]
        >= float(thresholds["minimum_support_recall_gain"]),
        "support_gain_cell_stability": positive_gain_cells >= int(thresholds["minimum_positive_gain_cells"]),
        "unseen_nll_ceiling": support["unseen_positive_nll"] <= float(thresholds["maximum_task_unseen_nll"]),
        "unseen_precision_floor": support["unseen_precision"] >= float(thresholds["minimum_unseen_precision"]),
        "unseen_fpr_ceiling": support["unseen_false_positive_rate"] <= float(thresholds["maximum_unseen_false_positive_rate"]),
        "matched_count3_recall_floor": support["matched_count3_recall"] >= float(thresholds["minimum_matched_count3_recall"]),
        "focused_unseen_recall_floor": support["focused_unseen_recall"] >= float(thresholds["minimum_focused_unseen_recall"]),
        "seen_recall_noninferiority": support["seen_positive_recall"]
        >= fixed["seen_positive_recall"] - float(thresholds["seen_recall_noninferiority_margin"]),
        "one_step_bce_noninferiority": support["task_macro_bce"]
        <= fixed["task_macro_bce"] + float(thresholds["one_step_bce_noninferiority_margin"]),
        "rollout_bce_noninferiority": support["rollout_bce"]
        <= fixed["rollout_bce"] + float(thresholds["rollout_bce_noninferiority_margin"]),
        "query_read_recall_noninferiority": support["query_read_positive_recall"]
        >= fixed["query_read_positive_recall"] - float(thresholds["query_read_recall_noninferiority_margin"]),
        "pair_accuracy_noninferiority": support["pair_assignment_accuracy"]
        >= fixed["pair_assignment_accuracy"] - float(thresholds["pair_accuracy_noninferiority_margin"]),
        "tool_family_unseen_recall_floor": diagnostics["tool_family_heldout"]["unseen_positive_recall"]
        >= float(thresholds["minimum_tool_family_unseen_recall"]),
        "source_unseen_recall_floor": diagnostics["source_heldout"]["unseen_positive_recall"]
        >= float(thresholds["minimum_source_unseen_recall"]),
        "parameter_ceiling": support["parameter_count"] <= float(thresholds["maximum_combined_parameters"]),
        "support_loader_contract_preserved": metrics.get("support_dataset_sha256")
        == protocol["data"]["support_dataset_sha256"],
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.support_conditioned_compositional_gate.v26",
        "decision": "GO_SUPPORT_CONDITIONED_COMPOSITIONAL_V26" if passed else "NO_GO_SUPPORT_CONDITIONED_COMPOSITIONAL_V26",
        "gate_checks": checks,
        "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "failed_clauses": [key for key, value in checks.items() if not value],
        "task_disjoint": {
            "fixed_v21": fixed,
            "factorized_ordinal_no_support_v26": no_support,
            "factorized_ordinal_support_v26": support,
            "v23_raw_external": v23_raw,
            "support_recall_gain_vs_no_support": support["unseen_positive_recall"] - no_support["unseen_positive_recall"],
            "support_recall_gain_vs_v23_raw": support["unseen_positive_recall"] - v23_raw["unseen_positive_recall"],
        },
        "fixed_reproduction_errors": reproduction_errors,
        "maximum_fixed_reproduction_error": max_reproduction_error,
        "gain_cells": gain_cells,
        "positive_gain_cells": positive_gain_cells,
        "diagnostics": diagnostics,
        "authorization": {
            "retain_v26_as_open_vocabulary_effect_head": passed,
            "run_frozen_96_episode_data_smoke": passed,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_generation": False,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed_clauses"], "total": payload["total_clauses"]}))


if __name__ == "__main__":
    main()
