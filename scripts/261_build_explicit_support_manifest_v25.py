"""Build the frozen outcome-blind v25 rare-mechanism support manifest."""

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
from wmagentattack.explicit_support_panel import build_explicit_support_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--raw-dataset", type=Path, required=True)
    p.add_argument("--semantic-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {"preregistered_before_manifest_build", "manifest_frozen_before_execution"}:
        raise ValueError("v25 protocol is not buildable")
    for path, expected in (
        (a.raw_dataset, protocol["sources"]["raw_dataset_sha256"]),
        (a.semantic_dataset, protocol["sources"]["semantic_dataset_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    raw = json.loads(a.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(a.semantic_dataset.read_text(encoding="utf-8"))
    suites = {name: get_suite(panel.BENCHMARK_VERSION, name) for name in protocol["selection"]["suites"]}
    specs = build_tool_binding_specs(suites, mutating_tools=set(panel.MUTATING_TOOLS))
    manifest, audit = build_explicit_support_manifest(
        raw,
        semantic,
        target_tools_by_task=protocol["selection"]["target_tools_by_task"],
        tool_specs=specs,
        seed=protocol["selection"]["seed"],
        anchors_per_task=int(protocol["selection"]["anchors_per_task"]),
        confirmation_task_ids=protocol["selection"]["confirmation_task_ids"],
    )
    expected = protocol["expected_manifest_preview"]
    checks = {key: audit[key] == value for key, value in expected.items()}
    audit["expected_manifest_checks"] = checks
    audit["passed"] = all(checks.values())
    write(a.output, manifest)
    audit["manifest_sha256"] = sha256(a.output)
    frozen = protocol.get("manifest", {}).get("sha256")
    if frozen and audit["manifest_sha256"] != frozen:
        raise ValueError("frozen manifest hash mismatch")
    write(a.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("v25 manifest preflight failed")


if __name__ == "__main__":
    main()
