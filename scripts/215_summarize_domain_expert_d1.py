"""Apply the frozen domain-expert D1 gate."""
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


c1sum = load("c1sum", ROOT / "scripts/211_summarize_interface_affordance_c1.py")


def domain_means(task_effects: dict[str, float]) -> dict[str, float]:
    values = defaultdict(list)
    for task, value in task_effects.items():
        values[task.split("|", 1)[0]].append(value)
    return {domain: float(np.mean(rows)) for domain, rows in sorted(values.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    rows = c1sum.read(args.predictions); metrics = json.loads(args.run_metrics.read_text())
    external = c1sum.read(protocol["external_v6_control"]["predictions"])
    if file_sha256(Path(protocol["external_v6_control"]["predictions"])) != protocol["external_v6_control"]["sha256"]:
        raise ValueError("external v6 mismatch")
    base = [row for row in external if row["arm"] == "structured_residual_v6"]
    dense = [row for row in rows if row["arm"] == "dense_capacity_control"]
    expert = [row for row in rows if row["arm"] == "domain_expert_d1"]
    h1_base = [row for row in base if row["horizon"] == 1]
    h1_expert = [row for row in expert if row["horizon"] == 1]
    multi_base = [row for row in base if row["horizon"] >= 2]
    multi_dense = [row for row in dense if row["horizon"] >= 2]
    multi_expert = [row for row in expert if row["horizon"] >= 2]
    joint_base = [row for row in multi_base if row["joint_trainable"]]
    joint_expert = [row for row in multi_expert if row["joint_trainable"]]
    effects = {
        "h1_nll_vs_v6": c1sum.effect(h1_base, h1_expert, "action_nll"),
        "h1_accuracy_vs_v6": c1sum.effect(h1_base, h1_expert, "action_correct", True),
        "h2_h5_nll_vs_v6": c1sum.effect(multi_base, multi_expert, "action_nll"),
        "h2_h5_nll_vs_dense": c1sum.effect(multi_dense, multi_expert, "action_nll"),
        "future_joint_ce_vs_v6": c1sum.effect(joint_base, joint_expert, "joint_ce"),
    }
    effects["h2_h5_nll_vs_v6"]["domains"] = domain_means(
        effects["h2_h5_nll_vs_v6"]["tasks"]
    )
    effects["h2_h5_nll_vs_dense"]["domains"] = domain_means(
        effects["h2_h5_nll_vs_dense"]["tasks"]
    )
    gate = protocol["stage_d1"]["gate"]
    domain_effects = effects["h2_h5_nll_vs_v6"]["domains"]
    audit = metrics["slot_audit"]; routing = metrics["routing"]
    checks = {
        "h1_nll_noninferiority": effects["h1_nll_vs_v6"]["mean"] >= -gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority": effects["h1_accuracy_vs_v6"]["mean"] >= -gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain_vs_v6": effects["h2_h5_nll_vs_v6"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_gain_vs_dense": effects["h2_h5_nll_vs_dense"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_dense"],
        "h2_h5_task_breadth": effects["h2_h5_nll_vs_v6"]["positive"] >= gate["minimum_h2_h5_positive_task_fraction"],
        "h2_h5_seed_replication": sum(
            value >= gate["minimum_h2_h5_nll_gain_vs_v6"]
            for value in effects["h2_h5_nll_vs_v6"]["seeds"].values()
        ) >= gate["minimum_threshold_positive_seeds"],
        "domain_breadth": sum(value > 0 for value in domain_effects.values()) >= gate["minimum_positive_domains"],
        "slack_nonnegative": domain_effects["slack"] >= gate["minimum_slack_h2_h5_gain_vs_v6"],
        "travel_nonnegative": domain_effects["travel"] >= gate["minimum_travel_h2_h5_gain_vs_v6"],
        "future_joint_noninferiority": effects["future_joint_ce_vs_v6"]["mean"] >= -gate["maximum_future_joint_ce_degradation"],
        "parameter_matched": metrics["parameter_diagnostics"]["gap_fraction"] <= gate["maximum_parameter_gap_fraction"],
        "router_integrity": routing["source"] == "causal_model_input.track" and not routing["task_id_used"] and all(value > 0 for value in routing["counts"].values()),
        "bottleneck_integrity": not audit["raw_values_encoded"] and audit["interface_only_lexical_encoding"] and audit["unmatched_text_tokens_encoded"] == audit["truncated_rows"] == audit["concept_truncated_rows"] == 0,
        "all_legal": all(row["legal_prediction"] == 1 for row in expert),
        "complete_budget": metrics["teacher_fits"] == metrics["dense_control_fits"] == metrics["domain_expert_fits"] == 15 and metrics["runtime_failures"] == 0,
    }
    decision = "GO_DOMAIN_EXPERT_D1" if all(checks.values()) else "NO_GO_DOMAIN_EXPERT_D1"
    summary = {
        "protocol_id": protocol["protocol_id"], "decision": decision,
        "checks": checks, "effects": effects,
        "absolute": {
            "v6_h1_nll": c1sum.macro(h1_base, "action_nll"),
            "expert_h1_nll": c1sum.macro(h1_expert, "action_nll"),
            "v6_h1_accuracy": c1sum.macro(h1_base, "action_correct"),
            "expert_h1_accuracy": c1sum.macro(h1_expert, "action_correct"),
            "v6_h2_h5_nll": c1sum.macro(multi_base, "action_nll"),
            "dense_h2_h5_nll": c1sum.macro(multi_dense, "action_nll"),
            "expert_h2_h5_nll": c1sum.macro(multi_expert, "action_nll"),
            "v6_future_joint_ce": c1sum.macro(joint_base, "joint_ce"),
            "expert_future_joint_ce": c1sum.macro(joint_expert, "joint_ce"),
        },
        "parameter_diagnostics": metrics["parameter_diagnostics"],
        "routing": routing,
        "predictions_sha256": file_sha256(args.predictions),
        "run_metrics_sha256": file_sha256(args.run_metrics),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(
        "# Domain expert D1\n\nDecision: `" + decision + "`\n\n"
        + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
        + "\n"
    )


if __name__ == "__main__":
    main()
