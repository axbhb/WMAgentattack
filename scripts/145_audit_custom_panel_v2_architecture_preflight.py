"""Fail-closed preflight for the frozen panel-v2 architecture ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    protocol_path: Path,
    dataset_path: Path,
    dataset_audit_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
    root = protocol_path.resolve().parents[1]
    implementation = {
        relative: {
            "expected": expected,
            "actual": _sha256(root / relative),
        }
        for relative, expected in protocol["implementation_sha256"].items()
    }
    argument_vocab = set(dataset["argument_key_vocab"])
    unknown_argument_target_keys = sorted(
        {
            key
            for episode in dataset["episodes"]
            for prefix in episode["prefixes"]
            for key in prefix["targets"]["argument_keys"]
            if key not in argument_vocab
        }
    )
    gates = {
        "protocol_preregistered_before_training": protocol.get("status")
        == "preregistered_before_training",
        "dataset_hash_frozen": _sha256(dataset_path)
        == protocol["source"]["dataset_sha256"],
        "dataset_audit_hash_frozen": _sha256(dataset_audit_path)
        == protocol["source"]["dataset_audit_sha256"],
        "dataset_audit_passed": dataset_audit.get("passed") is True
        and all(dataset_audit.get("gates", {}).values()),
        "dataset_schema_v2": dataset.get("schema_version")
        == "wmagentattack.custom_clean_panel_v2_architecture_dataset.v2"
        and dataset_audit.get("schema_version")
        == "wmagentattack.custom_clean_panel_v2_architecture_dataset.v2",
        "exact_budget_144_467_411": len(dataset["episodes"]) == 144
        and dataset_audit["prefixes"] == 467
        and sum(dataset_audit["evidence_statuses"].values()) == 411,
        "all_argument_targets_in_declared_schema_vocab": not unknown_argument_target_keys
        and dataset_audit.get("unknown_argument_target_keys") == [],
        "exact_three_nested_variants": protocol["frozen_variants"]
        == [
            "semantic_markov",
            "observable_execution",
            "observable_execution_ledger_v2",
        ]
        and protocol["representation_contract"]["strict_nesting"] is True,
        "implementation_hashes_match": all(
            row["expected"] == row["actual"] for row in implementation.values()
        ),
        "fixed_training_budget": protocol["training"]["training_seeds"]
        == [7, 17, 29]
        and protocol["training"]["hyperparameter_grid"] is False
        and protocol["fixed_budget"]["training_runs"] == 9,
        "completion_reporting_excluded": protocol["scope"][
            "completion_or_reporting_training"
        ]
        is False,
        "attack_h2_dreamer_blocked": protocol["scope"]["attack_data"] is False
        and protocol["scope"]["h2_attack_planning"] is False
        and protocol["scope"]["dreamer_training"] is False,
        "no_new_victim_calls": protocol["fixed_budget"]["new_victim_model_calls"]
        == 0,
    }
    passed = all(gates.values())
    return {
        "decision": (
            "CUSTOM_PANEL_V2_ARCHITECTURE_PREFLIGHT_PASS"
            if passed
            else "CUSTOM_PANEL_V2_ARCHITECTURE_PREFLIGHT_FAIL"
        ),
        "protocol_sha256": _sha256(protocol_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_audit_sha256": _sha256(dataset_audit_path),
        "implementation": implementation,
        "unknown_argument_target_keys": unknown_argument_target_keys,
        "gates": gates,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.protocol, args.dataset, args.dataset_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
