"""Execute the exact frozen clean-sandbox counterfactual manifest."""

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
from wmagentattack.panel_v2_architecture_probe import (
    load_panel_v2_adapter_registry,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--adapter-repair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_counterfactual_execution":
        raise ValueError("counterfactual manifest is not frozen")
    for path, expected in (
        (args.raw_dataset, protocol["source"]["raw_dataset_sha256"]),
        (args.semantic_dataset, protocol["source"]["semantic_dataset_sha256"]),
        (args.adapter_extension, protocol["source"]["adapter_extension_sha256"]),
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
    specs = build_tool_binding_specs(
        suites, mutating_tools=set(panel.MUTATING_TOOLS)
    )
    registry = load_panel_v2_adapter_registry(args.adapter_extension)
    if args.adapter_repair is not None:
        if _sha256(args.adapter_repair) != protocol["source"][
            "adapter_repair_sha256"
        ]:
            raise ValueError(
                f"frozen input hash mismatch: {args.adapter_repair}"
            )
        repair = json.loads(args.adapter_repair.read_text(encoding="utf-8"))
        if repair.get("base_adapter_extension_sha256") != _sha256(
            args.adapter_extension
        ):
            raise ValueError("adapter repair base-extension hash mismatch")
        registry = apply_label_blind_adapter_repair(registry, repair)
    elif "adapter_repair_sha256" in protocol["source"]:
        raise ValueError("frozen protocol requires --adapter-repair")
    dataset, audit = execute_frozen_manifest(
        raw,
        semantic,
        manifest,
        suites=suites,
        registry=registry,
        tool_specs=specs,
        selected_task_ids=protocol["selection"]["task_ids"],
        replicas=int(protocol["execution"]["fresh_state_replicas_per_query"]),
        readiness_gate=protocol["training_readiness_gate"],
        logical_clock_iso=protocol["execution"].get("frozen_logical_clock_iso"),
    )
    budget_checks = {
        "counterfactual_tool_executions": audit["counterfactual_executions"]
        == int(protocol["fixed_budget"]["counterfactual_tool_executions"]),
        "observed_prefix_replay_tool_executions": audit[
            "prior_observed_replay_executions"
        ]
        == int(
            protocol["fixed_budget"]["observed_prefix_replay_tool_executions"]
        ),
        "total_sandbox_tool_executions": audit["total_sandbox_tool_executions"]
        == int(protocol["fixed_budget"]["total_sandbox_tool_executions"]),
    }
    audit["budget_checks"] = budget_checks
    if not all(budget_checks.values()):
        audit["collector_pass"] = False
        audit["decision"] = "NO_GO__COUNTERFACTUAL_COLLECTION_BUDGET_MISMATCH"
    audit["manifest_sha256"] = _sha256(args.manifest)
    audit["protocol_sha256"] = _sha256(args.protocol)
    audit["raw_dataset_sha256"] = _sha256(args.raw_dataset)
    audit["semantic_dataset_sha256"] = _sha256(args.semantic_dataset)
    _write(args.output, dataset)
    audit["output_sha256"] = _sha256(args.output)
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["collector_pass"]:
        raise SystemExit("counterfactual collector gate failed")


if __name__ == "__main__":
    main()
