"""Apply the frozen v23 semantic-hybrid and recomposed long-horizon gate."""

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
        "task_macro_bce", "positive_task_macro_nll", "positive_task_macro_recall",
        "seen_positive_nll", "seen_positive_recall", "unseen_positive_nll",
        "unseen_positive_recall", "query_read_positive_recall", "execution_brier",
        "pair_assignment_accuracy", "rollout_bce", "rollout_positive_nll", "parameter_count",
    )
    output = {key: mean(rows, key) for key in keys}
    output["unseen_positive_occurrences_reported"] = int(sum(
        row["unseen_positive_occurrences"] for row in rows
    ))
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
    for key in ("v22_metrics", "v22_gate", "v22_long_gate", "v22_data_design_gate"):
        path = Path(external[f"{key}_path"])
        if sha256(path) != external[f"{key}_sha256"]:
            raise ValueError(f"external control hash mismatch: {key}")
    v22_gate = json.loads(Path(external["v22_gate_path"]).read_text(encoding="utf-8"))
    long_gate = json.loads(Path(external["v22_long_gate_path"]).read_text(encoding="utf-8"))
    data_gate = json.loads(Path(external["v22_data_design_gate_path"]).read_text(encoding="utf-8"))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in metrics["runs"]:
        grouped[row["arm"]][row["split_suite"]].append(row)
    suites = ("task_disjoint", "tool_family_heldout", "source_heldout")
    panels = {
        arm: {suite: panel(grouped[arm][suite]) for suite in suites}
        for arm in metrics["arms"]
    }
    baseline = panels["fixed_v21"]
    raw = panels["hybrid_e5_raw_v23"]
    candidate = panels["hybrid_e5_calibrated_v23"]
    external_baseline = v22_gate["panels"]["fixed_v21"]
    control = v22_gate["panels"]["independent_candidate_control_v22"]
    reproduction_keys = (
        "task_macro_bce", "seen_positive_recall", "unseen_positive_nll",
        "unseen_positive_recall", "rollout_bce",
    )
    reproduction_error = max(abs(
        baseline["task_disjoint"][key] - external_baseline["task_disjoint"][key]
    ) for key in reproduction_keys)
    thresholds = protocol["acceptance_thresholds"]
    clauses = {
        "complete_fixed_budget": (
            metrics["completed_model_fits"] == protocol["fixed_budget"]["model_fits"]
            and metrics["completed_metric_rows"] == protocol["fixed_budget"]["metric_rows"]
        ),
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "cache_is_label_blind_and_complete": (
            audit["outcome_fields_consumed"] == [] and audit["real_external_endpoint_calls"] == 0
            and audit["finite"] and audit["explained_energy"] >= thresholds["minimum_cache_explained_energy"]
            and audit["unit_norm_max_error"] <= thresholds["maximum_cache_unit_norm_error"]
        ),
        "fixed_v21_reproduced": reproduction_error <= thresholds["maximum_fixed_reproduction_absolute_error"],
        "parameter_limit": candidate["task_disjoint"]["parameter_count"] <= thresholds["maximum_combined_parameters"],
        "task_unseen_recall_floor": candidate["task_disjoint"]["unseen_positive_recall"] >= thresholds["minimum_task_unseen_recall"],
        "task_unseen_recall_gain_vs_id_control": (
            candidate["task_disjoint"]["unseen_positive_recall"]
            - control["task_disjoint"]["unseen_positive_recall"]
            >= thresholds["minimum_task_unseen_recall_gain_vs_independent_control"]
        ),
        "task_unseen_nll_gain_vs_id_control": (
            control["task_disjoint"]["unseen_positive_nll"]
            - candidate["task_disjoint"]["unseen_positive_nll"]
            >= thresholds["minimum_task_unseen_nll_gain_vs_independent_control"]
        ),
        "tool_family_unseen_recall_floor": candidate["tool_family_heldout"]["unseen_positive_recall"] >= thresholds["minimum_tool_family_unseen_recall"],
        "source_unseen_recall_floor": candidate["source_heldout"]["unseen_positive_recall"] >= thresholds["minimum_source_unseen_recall"],
        "seen_recall_noninferiority": candidate["task_disjoint"]["seen_positive_recall"] >= baseline["task_disjoint"]["seen_positive_recall"] - thresholds["seen_recall_noninferiority_margin"],
        "one_step_bce_noninferiority": candidate["task_disjoint"]["task_macro_bce"] <= baseline["task_disjoint"]["task_macro_bce"] + thresholds["one_step_bce_noninferiority_margin"],
        "rollout_bce_noninferiority": candidate["task_disjoint"]["rollout_bce"] <= baseline["task_disjoint"]["rollout_bce"] + thresholds["rollout_bce_noninferiority_margin"],
        "query_read_recall_noninferiority": candidate["tool_family_heldout"]["query_read_positive_recall"] >= baseline["tool_family_heldout"]["query_read_positive_recall"] - thresholds["query_read_recall_noninferiority_margin"],
        "calibration_improves_unseen_nll": raw["task_disjoint"]["unseen_positive_nll"] - candidate["task_disjoint"]["unseen_positive_nll"] >= thresholds["minimum_calibrated_unseen_nll_gain_vs_raw"],
        "calibration_preserves_unseen_recall": candidate["task_disjoint"]["unseen_positive_recall"] >= raw["task_disjoint"]["unseen_positive_recall"] - thresholds["calibrated_unseen_recall_noninferiority_margin"],
    }
    open_decision = "GO_PRETRAINED_SEMANTIC_HYBRID_V23" if all(clauses.values()) else "NO_GO_PRETRAINED_SEMANTIC_HYBRID_V23"
    direct_long = {
        key: value for key, value in long_gate["clauses"].items()
        if key != "v19_effect_rollout_noninferiority"
    }
    recomposed_long = all(direct_long.values()) and clauses["rollout_bce_noninferiority"]
    data_ready = data_gate["decision"].startswith("GO_DATA_GENERATION_PROTOCOL_READY_V22")
    overall_go = open_decision.startswith("GO_") and recomposed_long and data_ready
    decision = "GO_RUN_FROZEN_96_EPISODE_DATA_SMOKE_V23" if overall_go else "NO_GO_96_EPISODE_DATA_SMOKE_V23"
    payload = {
        "schema_version": "wmagentattack.pretrained_semantic_hybrid_gate.v23",
        "decision": decision,
        "open_vocabulary_decision": open_decision,
        "clauses": clauses,
        "passed": int(sum(bool(value) for value in clauses.values())),
        "total": len(clauses),
        "reproduction_max_absolute_error": reproduction_error,
        "panels": panels,
        "external_independent_control": control,
        "recomposed_long_gate": {
            "direct_v22_action_clauses": direct_long,
            "v23_effect_rollout_clause": clauses["rollout_bce_noninferiority"],
            "passed": recomposed_long,
        },
        "data_design_ready": data_ready,
        "authorization": {
            "run_frozen_96_episode_data_smoke": overall_go,
            "medium_scale_generation": false,
            "large_scale_generation": false,
            "large_world_model_training": false,
            "attack_generation": false,
        },
        "hashes": {
            "protocol": sha256(args.protocol), "metrics": sha256(args.metrics),
            "cache_audit": sha256(args.cache_audit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "open": open_decision, "passed": payload["passed"], "total": payload["total"]}))


if __name__ == "__main__":
    main()
