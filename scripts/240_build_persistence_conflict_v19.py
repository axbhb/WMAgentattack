"""Build the frozen v19 persistence/conflict sequence manifest."""

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
from wmagentattack.persistence_conflict import build_persistence_conflict_manifest


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
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {
        "preregistered_before_manifest_build",
        "manifest_frozen_before_sequence_execution",
    }:
        raise ValueError("v19 protocol is not buildable")
    if _sha256(args.base_manifest) != protocol["source"]["base_manifest_sha256"]:
        raise ValueError("v18 base manifest hash mismatch")
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    suites = {
        name: get_suite(panel.BENCHMARK_VERSION, name)
        for name in protocol["selection"]["suites"]
    }
    manifest, audit = build_persistence_conflict_manifest(
        base,
        suites=suites,
        selected_task_ids=protocol["selection"]["task_ids"],
        seed=protocol["selection"]["seed"],
    )
    checks = {
        key: audit[key] == value
        for key, value in protocol["expected_manifest_preview"].items()
    }
    audit["expected_manifest_checks"] = checks
    audit["passed"] = all(checks.values())
    _write(args.output, manifest)
    audit["manifest_sha256"] = _sha256(args.output)
    frozen = protocol.get("manifest", {}).get("sha256")
    if frozen is not None and frozen != audit["manifest_sha256"]:
        raise ValueError("frozen v19 manifest hash mismatch")
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("v19 manifest preflight failed")


if __name__ == "__main__":
    main()
