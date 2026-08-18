"""Apply the frozen relational-router E1 gate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.multisource_suitability import file_sha256


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


common = load("common", ROOT / "scripts/211_summarize_interface_affordance_c1.py")


def domain_means(tasks):
    values = defaultdict(list)
    for task, value in tasks.items(): values[task.split("|", 1)[0]].append(value)
    return {key: float(np.mean(rows)) for key, rows in sorted(values.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    rows = common.read(args.predictions); metrics = json.loads(args.run_metrics.read_text())
    external = common.read(protocol["external_v6_control"]["predictions"])
    if file_sha256(Path(protocol["external_v6_control"]["predictions"])) != protocol["external_v6_control"]["sha256"]:
        raise ValueError("external v6 mismatch")
    v9 = common.read(protocol["external_v9_negative_control"]["predictions"])
    if file_sha256(Path(protocol["external_v9_negative_control"]["predictions"])) != protocol["external_v9_negative_control"]["sha256"]:
        raise ValueError("external v9 mismatch")
    base = [row for row in external if row["arm"] == "structured_residual_v6"]
    dense = [row for row in rows if row["arm"] == "dense_relation_control"]
    sparse = [row for row in rows if row["arm"] == "sparse_relation_e1"]
    domain = [row for row in v9 if row["arm"] == "domain_expert_d1"]
    select = lambda source, h1: [row for row in source if (row["horizon"] == 1) == h1]
    h1_base, h1_sparse = select(base, True), select(sparse, True)
    multi_base, multi_dense = select(base, False), select(dense, False)
    multi_sparse, multi_domain = select(sparse, False), select(domain, False)
    joint_base = [row for row in multi_base if row["joint_trainable"]]
    joint_sparse = [row for row in multi_sparse if row["joint_trainable"]]
    effects = {
        "h1_nll_vs_v6": common.effect(h1_base, h1_sparse, "action_nll"),
        "h1_accuracy_vs_v6": common.effect(h1_base, h1_sparse, "action_correct", True),
        "h2_h5_nll_vs_v6": common.effect(multi_base, multi_sparse, "action_nll"),
        "h2_h5_nll_vs_dense": common.effect(multi_dense, multi_sparse, "action_nll"),
        "h2_h5_nll_vs_v9_domain": common.effect(multi_domain, multi_sparse, "action_nll"),
        "future_joint_ce_vs_v6": common.effect(joint_base, joint_sparse, "joint_ce"),
    }
    for name in ("h2_h5_nll_vs_v6", "h2_h5_nll_vs_dense"):
        effects[name]["domains"] = domain_means(effects[name]["tasks"])
    gate = protocol["stage_e1"]["gate"]; domains = effects["h2_h5_nll_vs_v6"]["domains"]
    routing = metrics["routing"]; audit = metrics["signature_audit"]
    checks = {
        "h1_nll_noninferiority": effects["h1_nll_vs_v6"]["mean"] >= -gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority": effects["h1_accuracy_vs_v6"]["mean"] >= -gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain_vs_v6": effects["h2_h5_nll_vs_v6"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_gain_vs_dense": effects["h2_h5_nll_vs_dense"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_dense"],
        "h2_h5_gain_vs_v9": effects["h2_h5_nll_vs_v9_domain"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_v9"],
        "task_breadth": effects["h2_h5_nll_vs_v6"]["positive"] >= gate["minimum_positive_task_fraction"],
        "seed_replication": sum(value >= gate["minimum_h2_h5_nll_gain_vs_v6"] for value in effects["h2_h5_nll_vs_v6"]["seeds"].values()) >= gate["minimum_threshold_positive_seeds"],
        "domain_breadth": sum(value > 0 for value in domains.values()) >= gate["minimum_positive_domains"],
        "slack_nonnegative": domains["slack"] >= 0,
        "travel_nonnegative": domains["travel"] >= 0,
        "future_joint_noninferiority": effects["future_joint_ce_vs_v6"]["mean"] >= -gate["maximum_future_joint_ce_degradation"],
        "parameter_matched": metrics["parameter_diagnostics"]["gap_fraction"] <= gate["maximum_parameter_gap_fraction"],
        "router_integrity": not routing["task_id_used"] and not routing["track_used"] and not routing["label_used"] and all(value > 0 for value in routing["hard_counts"]),
        "router_not_collapsed": routing["maximum_soft_load"] <= gate["maximum_expert_soft_load"] and routing["mean_normalized_topk_entropy"] >= gate["minimum_topk_entropy"] and abs(routing["mean_active_experts"] - 2) < 1e-6,
        "signature_integrity": not audit["lexical_hash_coordinates_used"] and not audit["raw_values_encoded"] and audit["training_only_standardization"] and audit["unmatched_text_tokens_encoded"] == audit["truncated_rows"] == audit["concept_truncated_rows"] == 0,
        "all_legal": all(row["legal_prediction"] == 1 for row in sparse),
        "complete_budget": metrics["teacher_fits"] == metrics["dense_control_fits"] == metrics["sparse_relation_fits"] == 15 and metrics["runtime_failures"] == 0,
    }
    decision = "GO_RELATIONAL_ROUTER_E1" if all(checks.values()) else "NO_GO_RELATIONAL_ROUTER_E1"
    summary = {
        "protocol_id": protocol["protocol_id"], "decision": decision,
        "checks": checks, "effects": effects,
        "absolute": {
            "v6_h1_nll": common.macro(h1_base, "action_nll"),
            "sparse_h1_nll": common.macro(h1_sparse, "action_nll"),
            "v6_h1_accuracy": common.macro(h1_base, "action_correct"),
            "sparse_h1_accuracy": common.macro(h1_sparse, "action_correct"),
            "v6_h2_h5_nll": common.macro(multi_base, "action_nll"),
            "dense_h2_h5_nll": common.macro(multi_dense, "action_nll"),
            "sparse_h2_h5_nll": common.macro(multi_sparse, "action_nll"),
            "v9_domain_h2_h5_nll": common.macro(multi_domain, "action_nll"),
            "v6_future_joint_ce": common.macro(joint_base, "joint_ce"),
            "sparse_future_joint_ce": common.macro(joint_sparse, "joint_ce"),
        },
        "parameter_diagnostics": metrics["parameter_diagnostics"], "routing": routing,
        "predictions_sha256": file_sha256(args.predictions),
        "run_metrics_sha256": file_sha256(args.run_metrics),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(
        "# Relational router E1\n\nDecision: `" + decision + "`\n\n" +
        "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + "\n"
    )


if __name__ == "__main__":
    main()
