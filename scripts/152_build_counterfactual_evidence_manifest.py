"""Build the frozen outcome-blind counterfactual execution manifest."""

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
from wmagentattack.counterfactual_evidence import (
    build_counterfactual_manifest,
    build_tool_binding_specs,
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
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--semantic-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {
        "preregistered_before_manifest_build",
        "manifest_frozen_before_counterfactual_execution",
    }:
        raise ValueError("counterfactual protocol is not in a buildable state")
    for path, expected in (
        (args.raw_dataset, protocol["source"]["raw_dataset_sha256"]),
        (args.semantic_dataset, protocol["source"]["semantic_dataset_sha256"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    raw = json.loads(args.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    suites = {
        name: get_suite(panel.BENCHMARK_VERSION, name)
        for name in protocol["selection"]["suites"]
    }
    specs = build_tool_binding_specs(
        suites, mutating_tools=set(panel.MUTATING_TOOLS)
    )
    manifest, audit = build_counterfactual_manifest(
        raw,
        semantic,
        selected_task_ids=protocol["selection"]["task_ids"],
        suites=protocol["selection"]["suites"],
        difficulties=protocol["selection"]["difficulties"],
        tool_specs=specs,
        seed=protocol["selection"]["seed"],
    )
    expected = protocol["expected_manifest_preview"]
    exact = {
        key: audit[key] == value
        for key, value in expected.items()
    }
    audit["expected_manifest_checks"] = exact
    audit["passed"] = all(exact.values()) and not audit["stop_selected"]
    _write(args.output, manifest)
    audit["manifest_sha256"] = _sha256(args.output)
    frozen_hash = protocol["manifest"].get("sha256")
    if frozen_hash is not None and audit["manifest_sha256"] != frozen_hash:
        raise ValueError("frozen manifest hash mismatch")
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("counterfactual manifest preflight failed")


if __name__ == "__main__":
    main()
