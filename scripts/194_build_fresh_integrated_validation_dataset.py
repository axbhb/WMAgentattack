"""Build the frozen historical-training / fresh-confirmation validation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.fresh_integrated_validation import (
    INTEGRATED_VALIDATION_SCHEMA_VERSION,
    build_fresh_action_and_transition_rows,
)
from wmagentattack.multisource_suitability import file_sha256, stable_hash


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--historical-actions", type=Path, required=True)
    parser.add_argument("--historical-action-audit", type=Path, required=True)
    parser.add_argument("--historical-transitions", type=Path, required=True)
    parser.add_argument("--historical-transition-audit", type=Path, required=True)
    parser.add_argument("--clean-gate", type=Path, required=True)
    parser.add_argument("--fresh-steps", type=Path, required=True)
    parser.add_argument("--fresh-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen = protocol["historical_training_sources"]
    source_hashes = {
        "three_source_dataset_sha256": file_sha256(args.historical_actions),
        "three_source_audit_sha256": file_sha256(args.historical_action_audit),
        "adjacent_transition_dataset_sha256": file_sha256(args.historical_transitions),
        "adjacent_transition_audit_sha256": file_sha256(args.historical_transition_audit),
    }
    for key, value in source_hashes.items():
        if value != frozen[key]:
            raise ValueError(f"historical source hash mismatch: {key}")
    action_audit = json.loads(args.historical_action_audit.read_text(encoding="utf-8"))
    transition_audit = json.loads(
        args.historical_transition_audit.read_text(encoding="utf-8")
    )
    if not action_audit["passed"] or not transition_audit["passed"]:
        raise ValueError("a historical preflight audit is not passing")
    clean_gate = json.loads(args.clean_gate.read_text(encoding="utf-8"))
    if clean_gate["decision"] != "GO_FRESH_CUSTOM_CONFIRMATION_V3_CLEAN_ELIGIBLE":
        raise ValueError("fresh clean gate did not authorize dataset construction")

    historical_actions = json.loads(args.historical_actions.read_text(encoding="utf-8"))
    historical_transitions = json.loads(
        args.historical_transitions.read_text(encoding="utf-8")
    )
    fresh_actions, fresh_transitions, catalog, fresh_audit = (
        build_fresh_action_and_transition_rows(
            steps=_read_jsonl(args.fresh_steps),
            metadata=_read_jsonl(args.fresh_metadata),
            historical_catalog=historical_actions["candidate_catalog"],
        )
    )
    historical_catalog_overlap_ok = all(
        key not in historical_transitions["candidate_catalog"]
        or stable_hash(historical_transitions["candidate_catalog"][key])
        == stable_hash(value)
        for key, value in historical_actions["candidate_catalog"].items()
    )
    dataset = {
        "schema_version": INTEGRATED_VALIDATION_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "historical_action_rows": historical_actions["rows"],
        "historical_transition_events": historical_transitions["events"],
        "fresh_action_rows": fresh_actions,
        "fresh_transition_events": fresh_transitions,
        "candidate_catalog": catalog,
    }
    checks = {
        "fresh_clean_gate_passed": clean_gate["passed"] is True,
        "historical_action_audit_passed": action_audit["passed"] is True,
        "historical_transition_audit_passed": transition_audit["passed"] is True,
        "historical_catalog_overlap_consistent": historical_catalog_overlap_ok,
        "fresh_builder_passed": fresh_audit["passed"] is True,
        "expected_historical_action_rows": len(dataset["historical_action_rows"]) == 13372,
        "expected_historical_transition_events": len(dataset["historical_transition_events"]) == 6763,
        "expected_fresh_trajectories": fresh_audit["fresh_trajectories"] == 36,
        "expected_fresh_tasks": fresh_audit["fresh_tasks"] == 12,
        "fresh_action_rows_nonempty": len(fresh_actions) > 36,
        "fresh_adjacent_transitions_nonempty": fresh_audit["fresh_adjacent_transitions"] > 0,
        "forbidden_causal_keys_absent": not fresh_audit["forbidden_causal_keys"],
    }
    audit = {
        "schema_version": INTEGRATED_VALIDATION_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "source_sha256": source_hashes,
        "clean_gate_sha256": file_sha256(args.clean_gate),
        "fresh_steps_sha256": file_sha256(args.fresh_steps),
        "fresh_metadata_sha256": file_sha256(args.fresh_metadata),
        "historical_action_rows": len(dataset["historical_action_rows"]),
        "historical_transition_events": len(dataset["historical_transition_events"]),
        "fresh_action_rows": len(fresh_actions),
        "fresh_transition_events": len(fresh_transitions),
        "fresh_audit": fresh_audit,
        "candidate_count": len(catalog),
        "dataset_content_sha256": stable_hash(dataset),
        "real_external_endpoint_calls": 0,
        "attack_episodes": 0,
        "dreamer_runs": 0,
    }
    _write(args.output, dataset)
    audit["dataset_file_sha256"] = file_sha256(args.output)
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("FRESH_INTEGRATED_DATASET_PREFLIGHT_NO_GO")


if __name__ == "__main__":
    main()
