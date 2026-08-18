"""Execute and gate the frozen same-prefix paired branch pilot."""

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
from wmagentattack.branching_identifiability import audit_paired_branch_effects
from wmagentattack.counterfactual_evidence import build_tool_binding_specs
from wmagentattack.counterfactual_execution import (
    apply_label_blind_adapter_repair,
    execute_frozen_manifest,
)
from wmagentattack.panel_v2_architecture_probe import (
    load_panel_v2_adapter_registry,
)


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
    if protocol["status"] != "manifest_frozen_before_branch_execution":
        raise ValueError("v17 branch manifest is not frozen")
    frozen = (
        (args.raw_dataset, protocol["source"]["raw_dataset_sha256"]),
        (args.semantic_dataset, protocol["source"]["semantic_dataset_sha256"]),
        (args.adapter_extension, protocol["source"]["adapter_extension_sha256"]),
        (args.adapter_repair, protocol["source"]["adapter_repair_sha256"]),
        (args.manifest, protocol["manifest"]["sha256"]),
    )
    for path, expected in frozen:
        if _sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")

    raw = json.loads(args.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suites = {
        name: get_suite(panel.BENCHMARK_VERSION, name)
        for name in protocol["selection"]["suites"]
    }
    specs = build_tool_binding_specs(
        suites, mutating_tools=set(panel.MUTATING_TOOLS)
    )
    registry = load_panel_v2_adapter_registry(args.adapter_extension)
    repair = json.loads(args.adapter_repair.read_text(encoding="utf-8"))
    registry = apply_label_blind_adapter_repair(registry, repair)
    expected_rows = int(protocol["fixed_budget"]["unique_bound_queries"])
    dataset, collector = execute_frozen_manifest(
        raw,
        semantic,
        manifest,
        suites=suites,
        registry=registry,
        tool_specs=specs,
        selected_task_ids=protocol["selection"]["task_ids"],
        replicas=int(protocol["execution"]["fresh_state_replicas_per_query"]),
        readiness_gate=protocol["legacy_evidence_diagnostics"],
        logical_clock_iso=protocol["execution"]["frozen_logical_clock_iso"],
        expected_rows=expected_rows,
        expected_per_class=protocol["fixed_budget"]["queries_per_class"],
    )
    effects = audit_paired_branch_effects(dataset, manifest)
    gates = {
        "collector_pass": collector["collector_pass"],
        "exact_complete_four_action_anchors": effects[
            "complete_four_action_anchors"
        ]
        == int(protocol["identifiability_gate"]["required_complete_anchors"]),
        "minimum_anchors_with_two_effects": effects["anchors_with_two_effects"]
        >= int(protocol["identifiability_gate"]["minimum_anchors_with_two_effects"]),
        "minimum_anchors_with_three_effects": effects[
            "anchors_with_three_effects"
        ]
        >= int(protocol["identifiability_gate"]["minimum_anchors_with_three_effects"]),
        "minimum_pairwise_effect_difference_fraction": effects[
            "pairwise_effect_difference_fraction"
        ]
        >= float(
            protocol["identifiability_gate"][
                "minimum_pairwise_effect_difference_fraction"
            ]
        ),
        "minimum_boundary_event_total": effects["boundary_event_total"]
        >= int(protocol["identifiability_gate"]["minimum_boundary_event_total"]),
        "minimum_boundary_event_types": effects["boundary_event_types_present"]
        >= int(protocol["identifiability_gate"]["minimum_boundary_event_types"]),
    }
    budget = {
        "counterfactual_tool_executions": collector["counterfactual_executions"]
        == int(protocol["fixed_budget"]["counterfactual_tool_executions"]),
        "observed_prefix_replay_tool_executions": collector[
            "prior_observed_replay_executions"
        ]
        == int(protocol["fixed_budget"]["observed_prefix_replay_tool_executions"]),
        "total_sandbox_tool_executions": collector["total_sandbox_tool_executions"]
        == int(protocol["fixed_budget"]["total_sandbox_tool_executions"]),
    }
    passed = all(gates.values()) and all(budget.values())
    audit = {
        "decision": (
            "GO_PAIRED_BRANCH_DATA_DIRECTION_V17"
            if passed
            else "NO_GO_PAIRED_BRANCH_DATA_DIRECTION_V17"
        ),
        "collector": collector,
        "identifiability": effects,
        "gate_checks": gates,
        "budget_checks": budget,
        "protocol_sha256": _sha256(args.protocol),
        "manifest_sha256": _sha256(args.manifest),
        "raw_dataset_sha256": _sha256(args.raw_dataset),
        "semantic_dataset_sha256": _sha256(args.semantic_dataset),
    }
    _write(args.output, dataset)
    audit["output_sha256"] = _sha256(args.output)
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not collector["collector_pass"] or not all(budget.values()):
        raise SystemExit("paired branch collector integrity failed")


if __name__ == "__main__":
    main()
