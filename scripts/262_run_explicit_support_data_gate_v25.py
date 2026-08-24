"""Execute the frozen support branches and apply the v25 coverage gate."""

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
from wmagentattack.counterfactual_execution import apply_label_blind_adapter_repair, execute_frozen_manifest
from wmagentattack.explicit_support_panel import build_atom_support_dataset_and_gate
from wmagentattack.panel_v2_architecture_probe import load_panel_v2_adapter_registry


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-dataset", type=Path, required=True)
    p.add_argument("--semantic-dataset", type=Path, required=True)
    p.add_argument("--hard-dataset", type=Path, required=True)
    p.add_argument("--adapter-extension", type=Path, required=True)
    p.add_argument("--adapter-repair", type=Path, required=True)
    p.add_argument("--execution-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_execution":
        raise ValueError("v25 manifest is not frozen")
    frozen = (
        (a.raw_dataset, protocol["sources"]["raw_dataset_sha256"]),
        (a.semantic_dataset, protocol["sources"]["semantic_dataset_sha256"]),
        (a.hard_dataset, protocol["sources"]["hard_dataset_sha256"]),
        (a.adapter_extension, protocol["sources"]["adapter_extension_sha256"]),
        (a.adapter_repair, protocol["sources"]["adapter_repair_sha256"]),
        (a.manifest, protocol["manifest"]["sha256"]),
    )
    for path, expected in frozen:
        if sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    raw = json.loads(a.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(a.semantic_dataset.read_text(encoding="utf-8"))
    hard = json.loads(a.hard_dataset.read_text(encoding="utf-8"))
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    suites = {name: get_suite(panel.BENCHMARK_VERSION, name) for name in protocol["selection"]["suites"]}
    specs = build_tool_binding_specs(suites, mutating_tools=set(panel.MUTATING_TOOLS))
    registry = load_panel_v2_adapter_registry(a.adapter_extension)
    registry = apply_label_blind_adapter_repair(registry, json.loads(a.adapter_repair.read_text(encoding="utf-8")))
    execution, collector = execute_frozen_manifest(
        raw,
        semantic,
        manifest,
        suites=suites,
        registry=registry,
        tool_specs=specs,
        selected_task_ids=sorted(protocol["selection"]["target_tools_by_task"]),
        replicas=2,
        readiness_gate=protocol["legacy_evidence_diagnostics"],
        logical_clock_iso=protocol["execution"]["frozen_logical_clock_iso"],
        expected_rows=int(protocol["fixed_budget"]["manifest_rows"]),
        expected_per_class=protocol["fixed_budget"]["rows_per_class"],
    )
    infrastructure_checks = {
        key: value
        for key, value in collector["collector_checks"].items()
        if key != "all_12_suite_difficulty_cells"
    }
    dataset, gate = build_atom_support_dataset_and_gate(
        execution,
        manifest,
        hard,
        confirmation_task_ids=protocol["selection"]["confirmation_task_ids"],
        thresholds=protocol["coverage_gate"],
    )
    budget_checks = {
        "branch_executions": collector["counterfactual_executions"] == int(protocol["fixed_budget"]["branch_tool_executions"]),
        "prefix_replay_executions": collector["prior_observed_replay_executions"] == int(protocol["fixed_budget"]["observed_prefix_replay_tool_executions"]),
        "total_sandbox_executions": collector["total_sandbox_tool_executions"] == int(protocol["fixed_budget"]["total_sandbox_tool_executions"]),
    }
    gate["collector"] = collector
    gate["infrastructure_checks"] = infrastructure_checks
    gate["budget_checks"] = budget_checks
    gate["protocol_sha256"] = sha256(a.protocol)
    gate["manifest_sha256"] = sha256(a.manifest)
    scientific_pass = (
        all(infrastructure_checks.values())
        and all(budget_checks.values())
        and all(gate["gate_checks"].values())
    )
    gate["decision"] = "GO_SUPPORT_CONDITIONED_MODEL_V25" if scientific_pass else "NO_GO_SUPPORT_DATA_V25"
    write(a.execution_output, execution)
    gate["execution_output_sha256"] = sha256(a.execution_output)
    write(a.output, dataset)
    gate["support_dataset_sha256"] = sha256(a.output)
    write(a.gate, gate)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    if not all(infrastructure_checks.values()) or not all(budget_checks.values()):
        raise SystemExit("v25 runtime integrity failed")


if __name__ == "__main__":
    main()
