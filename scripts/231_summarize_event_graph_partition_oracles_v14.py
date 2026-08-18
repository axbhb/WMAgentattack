"""Apply the frozen v14 exact/evidence oracle attribution gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module); return module


v12_summary = _load("v12_summary", "225_summarize_action_event_graph_oracle_v12.py")


def _read(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _arm_effect(v6, arm, name, *, horizon_one=False, accuracy=False, seed=81400):
    if horizon_one:
        left = [row for row in v6 if row["horizon"] == 1]
        right = [row for row in arm if row["horizon"] == 1]
    else:
        left = [row for row in v6 if row["horizon"] >= 2]
        right = [row for row in arm if row["horizon"] >= 2]
    return v12_summary._effect(left, right, name, higher=accuracy, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    current = _read(args.predictions); metrics = json.loads(args.run_metrics.read_text())
    v6_path = Path(protocol["sources"]["v6_predictions"])
    if file_sha256(v6_path) != protocol["sources"]["v6_predictions_sha256"]: raise ValueError("v6 hash mismatch")
    v6 = [row for row in _read(v6_path) if row["arm"] == "structured_residual_v6"]
    arms = {name: [row for row in current if row["arm"] == name] for name in protocol["oracle_attribution_stage"]["arms"]}
    if any(len(rows) != len(v6) for rows in arms.values()): raise ValueError("paired row mismatch")
    effects = {}
    for offset, (name, rows) in enumerate(arms.items()):
        effects[name] = {
            "h1_nll_vs_v6": _arm_effect(v6, rows, "action_nll", horizon_one=True, seed=81401 + offset * 10),
            "h1_accuracy_vs_v6": _arm_effect(v6, rows, "action_correct", horizon_one=True, accuracy=True, seed=81402 + offset * 10),
            "h2_h5_nll_vs_v6": _arm_effect(v6, rows, "action_nll", seed=81403 + offset * 10),
        }
    stage = protocol["oracle_attribution_stage"]
    full_name = "full_graph_modular_oracle_v14"
    full = effects[full_name]; full_gate = stage["full_graph_gate"]
    full_checks = {
        "h1_nll_noninferiority": full["h1_nll_vs_v6"]["mean"] >= -full_gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority": full["h1_accuracy_vs_v6"]["mean"] >= -full_gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain": full["h2_h5_nll_vs_v6"]["mean"] >= full_gate["minimum_h2_h5_nll_gain_vs_v6"],
        "task_breadth": full["h2_h5_nll_vs_v6"]["positive_task_fraction"] >= full_gate["minimum_positive_task_fraction"],
        "seed_replication": sum(value > 0 for value in full["h2_h5_nll_vs_v6"]["seeds"].values()) >= full_gate["minimum_positive_seeds"],
    }
    partition_checks = {}
    gate = stage["per_arm_gate"]
    full_gain = full["h2_h5_nll_vs_v6"]["mean"]
    for name in ("exact_protocol_oracle_v14", "stochastic_evidence_oracle_v14"):
        value = effects[name]
        gain = value["h2_h5_nll_vs_v6"]["mean"]
        partition_checks[name] = {
            "h1_nll_noninferiority": value["h1_nll_vs_v6"]["mean"] >= -gate["maximum_h1_nll_degradation_vs_v6"],
            "h1_accuracy_noninferiority": value["h1_accuracy_vs_v6"]["mean"] >= -gate["maximum_h1_accuracy_degradation_vs_v6"],
            "absolute_h2_h5_gain": gain >= gate["minimum_h2_h5_nll_gain_vs_v6"],
            "fraction_of_full_gain": full_gain > 0 and gain / full_gain >= gate["minimum_fraction_of_v12_full_oracle_gain"],
            "task_breadth": value["h2_h5_nll_vs_v6"]["positive_task_fraction"] >= gate["minimum_positive_task_fraction"],
            "seed_replication": sum(entry > 0 for entry in value["h2_h5_nll_vs_v6"]["seeds"].values()) >= gate["minimum_positive_seeds"],
        }
    integrity = {
        "complete_budget": metrics["training_units"] == 45 and metrics["teacher_fits"] == 15,
        "runtime_clean": metrics["runtime_failures"] == 0,
        "parameter_match": bool(metrics["parameter_match"]),
        "all_legal": all(row["legal_prediction"] == 1 for row in current),
    }
    full_pass = all(full_checks.values())
    exact_pass = full_pass and all(partition_checks["exact_protocol_oracle_v14"].values()) and all(integrity.values())
    evidence_pass = full_pass and all(partition_checks["stochastic_evidence_oracle_v14"].values()) and all(integrity.values())
    if exact_pass and evidence_pass: decision = "GO_BOTH_PARTITIONS_V14"
    elif exact_pass: decision = "GO_EXACT_PROTOCOL_ONLY_V14"
    elif evidence_pass: decision = "GO_EVIDENCE_RESIDUAL_V14"
    elif not full_pass: decision = "INVALIDATE_PARTITION_ATTRIBUTION_FULL_REPLAY_FAILED_V14"
    else: decision = "NO_GO_INDIVIDUAL_PARTITIONS_INTERACTION_REQUIRED_V14"
    summary = {
        "protocol_id": protocol["protocol_id"], "decision": decision,
        "full_checks": full_checks, "partition_checks": partition_checks, "integrity_checks": integrity,
        "effects": effects, "exact_pass": exact_pass, "evidence_pass": evidence_pass,
        "counts": {name: len(rows) for name, rows in arms.items()},
        "predictions_sha256": file_sha256(args.predictions), "run_metrics_sha256": file_sha256(args.run_metrics),
    }
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    lines = ["# Event-graph partition oracle attribution v14", "", f"Decision: `{decision}`", "", "## Multi-step NLL gain over v6", ""]
    for name, value in effects.items(): lines.append(f"- {name}: {value['h2_h5_nll_vs_v6']['mean']:.6f}")
    lines.extend(["", "## Integrity", ""]); lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in integrity.items())
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
