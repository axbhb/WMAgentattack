"""Apply the frozen v24 relation-factorized semantic and scale gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def panel(rows):
    keys = (
        "task_macro_bce",
        "positive_task_macro_nll",
        "positive_task_macro_recall",
        "seen_positive_nll",
        "seen_positive_recall",
        "unseen_positive_nll",
        "unseen_positive_recall",
        "unseen_false_positive_rate",
        "unseen_precision",
        "mean_predicted_unseen_set_size",
        "mean_true_unseen_set_size",
        "decision_threshold",
        "support_weight",
        "query_read_positive_recall",
        "execution_brier",
        "pair_assignment_accuracy",
        "rollout_bce",
        "rollout_positive_nll",
        "parameter_count",
    )
    output = {key: mean(rows, key) for key in keys}
    output["unseen_positive_occurrences_reported"] = int(sum(
        row.get("unseen_positive_occurrences", 0) for row in rows
    ))
    selections = [
        row.get("selection", {}) for row in rows
        if row.get("arm") == "relation_support_set_v24"
    ]
    output["selection_feasible_fraction"] = (
        float(np.mean([bool(value.get("feasible", False)) for value in selections]))
        if selections else None
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    external = protocol["external_controls"]
    for key in ("v23_metrics", "v23_gate", "v22_long_gate", "v22_data_design_gate"):
        path = Path(external[f"{key}_path"])
        if sha256(path) != external[f"{key}_sha256"]:
            raise ValueError(f"v24 external-control hash mismatch: {key}")
    v23_gate = json.loads(
        Path(external["v23_gate_path"]).read_text(encoding="utf-8")
    )
    long_gate = json.loads(
        Path(external["v22_long_gate_path"]).read_text(encoding="utf-8")
    )
    data_gate = json.loads(
        Path(external["v22_data_design_gate_path"]).read_text(encoding="utf-8")
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for row in metrics["runs"]:
        grouped[row["arm"]][row["split_suite"]].append(row)
    suites = ("task_disjoint", "tool_family_heldout", "source_heldout")
    panels = {
        arm: {suite: panel(grouped[arm][suite]) for suite in suites}
        for arm in metrics["arms"]
    }
    baseline = panels["fixed_v21"]
    raw = panels["relation_e5_raw_v24"]
    candidate = panels["relation_support_set_v24"]
    v23_baseline = v23_gate["panels"]["fixed_v21"]
    v23_raw = v23_gate["panels"]["hybrid_e5_raw_v23"]
    v23_calibrated = v23_gate["panels"]["hybrid_e5_calibrated_v23"]
    v22_control = v23_gate["external_independent_control"]
    reproduction_keys = (
        "task_macro_bce",
        "seen_positive_recall",
        "unseen_positive_nll",
        "unseen_positive_recall",
        "rollout_bce",
    )
    reproduction_error = max(abs(
        baseline["task_disjoint"][key] - v23_baseline["task_disjoint"][key]
    ) for key in reproduction_keys)
    thresholds = protocol["acceptance_thresholds"]
    predicted_set_limit = (
        thresholds["maximum_task_mean_predicted_set_multiplier"]
        * candidate["task_disjoint"]["mean_true_unseen_set_size"]
        + thresholds["maximum_task_mean_predicted_set_offset"]
    )
    clauses = {
        "complete_fixed_budget": (
            metrics["completed_model_fits"] == protocol["fixed_budget"]["model_fits"]
            and metrics["completed_metric_rows"]
            == protocol["fixed_budget"]["metric_rows"]
        ),
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "cache_is_label_blind_and_complete": (
            audit["outcome_fields_consumed"] == []
            and audit["real_external_endpoint_calls"] == 0
            and audit["finite"]
            and audit["explained_energy"]
            >= thresholds["minimum_cache_explained_energy"]
            and audit["unit_norm_max_error"]
            <= thresholds["maximum_cache_unit_norm_error"]
            and audit["relation_kernel_symmetric_error"] <= 1e-6
        ),
        "fixed_v21_reproduced": (
            reproduction_error
            <= thresholds["maximum_fixed_reproduction_absolute_error"]
        ),
        "parameter_limit": (
            candidate["task_disjoint"]["parameter_count"]
            <= thresholds["maximum_combined_parameters"]
        ),
        "all_inner_support_rules_feasible": (
            candidate["task_disjoint"]["selection_feasible_fraction"] == 1.0
            and candidate["tool_family_heldout"]["selection_feasible_fraction"] == 1.0
            and candidate["source_heldout"]["selection_feasible_fraction"] == 1.0
        ),
        "task_unseen_recall_floor": (
            candidate["task_disjoint"]["unseen_positive_recall"]
            >= thresholds["minimum_task_unseen_recall"]
        ),
        "task_unseen_recall_gain_vs_v23_raw": (
            candidate["task_disjoint"]["unseen_positive_recall"]
            - v23_raw["task_disjoint"]["unseen_positive_recall"]
            >= thresholds["minimum_task_unseen_recall_gain_vs_v23_raw"]
        ),
        "task_unseen_nll_noninferiority_vs_v23_raw": (
            candidate["task_disjoint"]["unseen_positive_nll"]
            <= v23_raw["task_disjoint"]["unseen_positive_nll"]
            + thresholds["maximum_task_unseen_nll_increase_vs_v23_raw"]
        ),
        "tool_family_unseen_recall_floor": (
            candidate["tool_family_heldout"]["unseen_positive_recall"]
            >= thresholds["minimum_tool_family_unseen_recall"]
        ),
        "source_unseen_recall_floor": (
            candidate["source_heldout"]["unseen_positive_recall"]
            >= thresholds["minimum_source_unseen_recall"]
        ),
        "task_unseen_false_positive_rate": (
            candidate["task_disjoint"]["unseen_false_positive_rate"]
            <= thresholds["maximum_task_unseen_false_positive_rate"]
        ),
        "task_unseen_precision": (
            candidate["task_disjoint"]["unseen_precision"]
            >= thresholds["minimum_task_unseen_precision"]
        ),
        "task_unseen_set_size": (
            candidate["task_disjoint"]["mean_predicted_unseen_set_size"]
            <= predicted_set_limit
        ),
        "seen_recall_noninferiority": (
            candidate["task_disjoint"]["seen_positive_recall"]
            >= baseline["task_disjoint"]["seen_positive_recall"]
            - thresholds["seen_recall_noninferiority_margin"]
        ),
        "one_step_bce_noninferiority": (
            candidate["task_disjoint"]["task_macro_bce"]
            <= baseline["task_disjoint"]["task_macro_bce"]
            + thresholds["one_step_bce_noninferiority_margin"]
        ),
        "rollout_bce_noninferiority": (
            candidate["task_disjoint"]["rollout_bce"]
            <= baseline["task_disjoint"]["rollout_bce"]
            + thresholds["rollout_bce_noninferiority_margin"]
        ),
        "query_read_recall_noninferiority": (
            candidate["tool_family_heldout"]["query_read_positive_recall"]
            >= baseline["tool_family_heldout"]["query_read_positive_recall"]
            - thresholds["query_read_recall_noninferiority_margin"]
        ),
        "support_improves_relation_raw_recall": (
            candidate["task_disjoint"]["unseen_positive_recall"]
            - raw["task_disjoint"]["unseen_positive_recall"]
            >= thresholds["minimum_support_recall_gain_vs_relation_raw"]
        ),
        "support_nll_noninferiority_vs_relation_raw": (
            candidate["task_disjoint"]["unseen_positive_nll"]
            <= raw["task_disjoint"]["unseen_positive_nll"]
            + thresholds["maximum_support_nll_increase_vs_relation_raw"]
        ),
    }
    open_decision = (
        "GO_RELATION_FACTORIZED_DISTRIBUTION_V24"
        if all(clauses.values())
        else "NO_GO_RELATION_FACTORIZED_DISTRIBUTION_V24"
    )
    direct_long = {
        key: value for key, value in long_gate["clauses"].items()
        if key != "v19_effect_rollout_noninferiority"
    }
    recomposed_long = (
        all(direct_long.values()) and clauses["rollout_bce_noninferiority"]
    )
    data_ready = data_gate["decision"].startswith(
        "GO_DATA_GENERATION_PROTOCOL_READY_V22"
    )
    overall_go = (
        open_decision.startswith("GO_") and recomposed_long and data_ready
    )
    decision = (
        "GO_RUN_FROZEN_96_EPISODE_DATA_SMOKE_V24"
        if overall_go
        else "NO_GO_96_EPISODE_DATA_SMOKE_V24"
    )
    payload = {
        "schema_version": "wmagentattack.relation_factorized_distribution_gate.v24",
        "decision": decision,
        "open_vocabulary_decision": open_decision,
        "clauses": clauses,
        "passed": int(sum(bool(value) for value in clauses.values())),
        "total": len(clauses),
        "reproduction_max_absolute_error": reproduction_error,
        "task_predicted_set_limit": predicted_set_limit,
        "panels": panels,
        "external_controls": {
            "v23_raw": v23_raw,
            "v23_calibrated": v23_calibrated,
            "v22_independent": v22_control,
        },
        "recomposed_long_gate": {
            "direct_v22_action_clauses": direct_long,
            "v24_effect_rollout_clause": clauses["rollout_bce_noninferiority"],
            "passed": recomposed_long,
        },
        "data_design_ready": data_ready,
        "authorization": {
            "run_frozen_96_episode_data_smoke": overall_go,
            "medium_scale_generation": False,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_generation": False,
        },
        "hashes": {
            "protocol": sha256(args.protocol),
            "metrics": sha256(args.metrics),
            "cache_audit": sha256(args.cache_audit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "open": open_decision,
        "passed": payload["passed"],
        "total": payload["total"],
    }))


if __name__ == "__main__":
    main()
