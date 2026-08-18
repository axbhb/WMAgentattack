"""Execute and gate the frozen v19 multi-step sequences."""

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
from wmagentattack.counterfactual_execution import apply_label_blind_adapter_repair
from wmagentattack.panel_v2_architecture_probe import load_panel_v2_adapter_registry
from wmagentattack.persistence_conflict import execute_persistence_conflict_manifest


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
    parser.add_argument("--persistence-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_sequence_execution":
        raise ValueError("v19 manifest is not frozen")
    for path, expected in (
        (args.manifest, protocol["manifest"]["sha256"]),
        (args.raw_dataset, protocol["source"]["raw_dataset_sha256"]),
        (args.semantic_dataset, protocol["source"]["semantic_dataset_sha256"]),
        (args.adapter_extension, protocol["source"]["adapter_extension_sha256"]),
        (args.adapter_repair, protocol["source"]["adapter_repair_sha256"]),
        (args.persistence_adapter, protocol["source"]["persistence_adapter_sha256"]),
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
    registry = load_panel_v2_adapter_registry(args.adapter_extension)
    registry = apply_label_blind_adapter_repair(
        registry, json.loads(args.adapter_repair.read_text(encoding="utf-8"))
    )
    registry = apply_label_blind_adapter_repair(
        registry, json.loads(args.persistence_adapter.read_text(encoding="utf-8"))
    )
    dataset, observed = execute_persistence_conflict_manifest(
        raw,
        semantic,
        manifest,
        suites=suites,
        registry=registry,
        replicas=2,
        logical_clock_iso=protocol["execution"]["frozen_logical_clock_iso"],
    )
    gate = protocol["persistence_conflict_gate"]
    gate_checks = {
        "exact_complete_sequences": observed["complete_sequences"] == gate["required_sequences"],
        "exact_complete_pairs": observed["complete_pairs"] == gate["required_pairs"],
        "all_steps_success": observed["all_steps_success"],
        "all_control_persistence_matches": observed["control_persistence_matches"]
        == gate["minimum_control_persistence_matches"],
        "all_conflict_readbacks_match": observed["conflict_readback_matches"]
        == gate["minimum_conflict_readback_matches"],
        "shared_first_write_identical": observed["shared_write_states_identical"]
        == gate["minimum_shared_write_states_identical"],
        "final_semantic_states_differ": observed["final_semantic_states_differ"]
        >= gate["minimum_final_semantic_states_differ"],
        "final_observations_differ": observed["final_observations_differ"]
        == gate["minimum_final_observations_differ"],
        "all_suites_represented": observed["suites_with_both_matches"] == 4,
        "replicas_identical": observed["replicas_identical"],
        "zero_prefix_replay_mismatches": observed["zero_prefix_replay_mismatches"],
        "zero_semantic_leakage": observed["zero_semantic_leakage"],
        "zero_runtime_failures": not observed["runtime_failures"],
    }
    budget = protocol["fixed_budget"]
    budget_checks = {
        "exact_step_executions": observed["step_executions"] == budget["sequence_step_executions"],
        "exact_prefix_replay_calls": observed["prefix_replay_calls"]
        == budget["observed_prefix_replay_tool_executions"],
        "exact_total_calls": observed["total_sandbox_calls"] == budget["total_sandbox_tool_executions"],
    }
    passed = all(gate_checks.values()) and all(budget_checks.values())
    audit = {
        "decision": (
            "GO_PERSISTENCE_CONFLICT_DATA_DIRECTION_V19"
            if passed
            else "NO_GO_PERSISTENCE_CONFLICT_DATA_DIRECTION_V19"
        ),
        "observed": observed,
        "gate_checks": gate_checks,
        "budget_checks": budget_checks,
        "protocol_sha256": _sha256(args.protocol),
        "manifest_sha256": _sha256(args.manifest),
    }
    _write(args.output, dataset)
    audit["output_sha256"] = _sha256(args.output)
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if observed["runtime_failures"] or not all(budget_checks.values()):
        raise SystemExit("v19 runtime or budget integrity failed")


if __name__ == "__main__":
    main()
