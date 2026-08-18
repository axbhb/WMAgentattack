"""Execute and gate the frozen v18 parameter intervention pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.counterfactual_evidence import build_tool_binding_specs
from wmagentattack.counterfactual_execution import (
    apply_label_blind_adapter_repair,
    execute_frozen_manifest,
)
from wmagentattack.panel_v2_architecture_probe import load_panel_v2_adapter_registry
from wmagentattack.parameter_intervention import audit_parameter_interventions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--semantic-dataset", type=Path, required=True)
    parser.add_argument("--adapter-extension", type=Path, required=True)
    parser.add_argument("--adapter-repair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_parameter_execution":
        raise ValueError("v18 parameter manifest is not frozen")
    for path, expected in (
        (args.raw_dataset, protocol["source"]["raw_dataset_sha256"]),
        (args.semantic_dataset, protocol["source"]["semantic_dataset_sha256"]),
        (args.adapter_extension, protocol["source"]["adapter_extension_sha256"]),
        (args.adapter_repair, protocol["source"]["adapter_repair_sha256"]),
        (args.manifest, protocol["manifest"]["sha256"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    raw = json.loads(args.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suites = {
        name: get_suite(panel.BENCHMARK_VERSION, name)
        for name in protocol["selection"]["suites"]
    }
    specs = build_tool_binding_specs(suites, mutating_tools=set(panel.MUTATING_TOOLS))
    registry = load_panel_v2_adapter_registry(args.adapter_extension)
    registry = apply_label_blind_adapter_repair(
        registry, json.loads(args.adapter_repair.read_text(encoding="utf-8"))
    )
    rows = int(protocol["fixed_budget"]["manifest_rows"])
    dataset, collector = execute_frozen_manifest(
        raw,
        semantic,
        manifest,
        suites=suites,
        registry=registry,
        tool_specs=specs,
        selected_task_ids=protocol["selection"]["task_ids"],
        replicas=2,
        readiness_gate=protocol["legacy_evidence_diagnostics"],
        logical_clock_iso=protocol["execution"]["frozen_logical_clock_iso"],
        expected_rows=rows,
        expected_per_class={"mutating": rows},
    )
    paired = audit_parameter_interventions(dataset, manifest)
    gates = {
        "collector_pass": collector["collector_pass"],
        "exact_complete_pairs": paired["complete_pairs"]
        == int(protocol["parameter_gate"]["required_complete_pairs"]),
        "minimum_control_successes": paired["control_successes"]
        >= int(protocol["parameter_gate"]["minimum_control_successes"]),
        "minimum_corrupted_errors": paired["corrupted_errors"]
        >= int(protocol["parameter_gate"]["minimum_corrupted_errors"]),
        "minimum_paired_status_flips": paired["paired_status_flips"]
        >= int(protocol["parameter_gate"]["minimum_paired_status_flips"]),
        "minimum_effect_changes": paired["pairs_with_effect_change"]
        >= int(protocol["parameter_gate"]["minimum_effect_changes"]),
        "all_suites_have_status_flip": paired["suites_with_status_flip"] == 4,
    }
    budget = {
        "branch_calls": collector["counterfactual_executions"]
        == int(protocol["fixed_budget"]["branch_tool_executions"]),
        "prefix_replay_calls": collector["prior_observed_replay_executions"]
        == int(protocol["fixed_budget"]["observed_prefix_replay_tool_executions"]),
        "total_calls": collector["total_sandbox_tool_executions"]
        == int(protocol["fixed_budget"]["total_sandbox_tool_executions"]),
    }
    passed = all(gates.values()) and all(budget.values())
    audit = {
        "decision": (
            "GO_PARAMETER_INTERVENTION_DATA_DIRECTION_V18"
            if passed
            else "NO_GO_PARAMETER_INTERVENTION_DATA_DIRECTION_V18"
        ),
        "collector": collector,
        "paired": paired,
        "gate_checks": gates,
        "budget_checks": budget,
        "protocol_sha256": _sha256(args.protocol),
        "manifest_sha256": _sha256(args.manifest),
    }
    _write(args.output, dataset)
    audit["output_sha256"] = _sha256(args.output)
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not collector["collector_pass"] or not all(budget.values()):
        raise SystemExit("v18 collection integrity failed")


if __name__ == "__main__":
    main()
