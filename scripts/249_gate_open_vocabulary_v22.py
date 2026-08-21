"""Apply the frozen v22 open-vocabulary acceptance gate."""

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


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def by_arm_suite(runs: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in runs:
        grouped[row["arm"]][row["split_suite"]].append(row)
    return grouped


def metric_panel(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = (
        "task_macro_bce",
        "positive_task_macro_nll",
        "positive_task_macro_recall",
        "seen_positive_nll",
        "seen_positive_recall",
        "unseen_positive_nll",
        "unseen_positive_recall",
        "query_read_positive_recall",
        "execution_brier",
        "pair_assignment_accuracy",
        "rollout_bce",
        "rollout_positive_nll",
        "parameter_count",
    )
    panel = {key: mean(rows, key) for key in keys}
    panel["unseen_positive_occurrences_reported"] = int(sum(
        row["unseen_positive_occurrences"] for row in rows
    ))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics["dataset_sha256"] != protocol["data"]["sha256"]:
        raise ValueError("v22 gate data hash mismatch")
    expected = int(protocol["open_vocabulary_gate"]["fixed_budget"]["model_fits"])
    grouped = by_arm_suite(metrics["runs"])
    arms = ("fixed_v21", "independent_candidate_control_v22", "compositional_candidate_v22")
    suites = ("task_disjoint", "tool_family_heldout", "source_heldout")
    panels = {
        arm: {suite: metric_panel(grouped[arm][suite]) for suite in suites}
        for arm in arms
    }
    candidate = panels["compositional_candidate_v22"]
    baseline = panels["fixed_v21"]
    control = panels["independent_candidate_control_v22"]
    thresholds = protocol["open_vocabulary_gate"]["acceptance_thresholds"]
    clauses = {
        "complete_fixed_budget": metrics["completed_runs"] == expected == len(metrics["runs"]),
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "parameter_limit": candidate["task_disjoint"]["parameter_count"] <= thresholds["maximum_parameters"],
        "task_unseen_recall_floor": candidate["task_disjoint"]["unseen_positive_recall"] >= thresholds["minimum_task_unseen_recall"],
        "task_unseen_recall_gain_vs_id_control": (
            candidate["task_disjoint"]["unseen_positive_recall"]
            - control["task_disjoint"]["unseen_positive_recall"]
            >= thresholds["minimum_unseen_recall_gain_vs_independent_control"]
        ),
        "task_unseen_nll_gain_vs_fixed": (
            baseline["task_disjoint"]["unseen_positive_nll"]
            - candidate["task_disjoint"]["unseen_positive_nll"]
            >= thresholds["minimum_unseen_nll_gain_vs_fixed"]
        ),
        "tool_family_unseen_recall_floor": candidate["tool_family_heldout"]["unseen_positive_recall"] >= thresholds["minimum_diagnostic_unseen_recall"],
        "source_unseen_recall_floor": candidate["source_heldout"]["unseen_positive_recall"] >= thresholds["minimum_diagnostic_unseen_recall"],
        "seen_recall_noninferiority": (
            candidate["task_disjoint"]["seen_positive_recall"]
            >= baseline["task_disjoint"]["seen_positive_recall"] - thresholds["seen_recall_noninferiority_margin"]
        ),
        "one_step_bce_noninferiority": (
            candidate["task_disjoint"]["task_macro_bce"]
            <= baseline["task_disjoint"]["task_macro_bce"] + thresholds["one_step_bce_noninferiority_margin"]
        ),
        "rollout_bce_noninferiority": (
            candidate["task_disjoint"]["rollout_bce"]
            <= baseline["task_disjoint"]["rollout_bce"] + thresholds["rollout_bce_noninferiority_margin"]
        ),
        "query_read_recall_noninferiority": (
            candidate["tool_family_heldout"]["query_read_positive_recall"]
            >= baseline["tool_family_heldout"]["query_read_positive_recall"] - thresholds["query_read_recall_noninferiority_margin"]
        ),
    }
    decision = (
        "GO_COMPOSITIONAL_OPEN_VOCABULARY_V22"
        if all(clauses.values())
        else "NO_GO_OPEN_VOCABULARY_V22"
    )
    payload = {
        "schema_version": "wmagentattack.open_vocabulary_gate.v22",
        "protocol_sha256": sha256(args.protocol),
        "metrics_sha256": sha256(args.metrics),
        "decision": decision,
        "passed_clauses": sum(clauses.values()),
        "total_clauses": len(clauses),
        "clauses": clauses,
        "panels": panels,
        "authorization": {
            "medium_scale_generation": decision == "GO_COMPOSITIONAL_OPEN_VOCABULARY_V22",
            "large_scale_generation": False,
            "attack_generation": False,
            "large_world_model_training": False,
        },
    }
    write(args.output, payload)
    print(json.dumps({"decision": decision, "passed": sum(clauses.values()), "total": len(clauses)}))


if __name__ == "__main__":
    main()
