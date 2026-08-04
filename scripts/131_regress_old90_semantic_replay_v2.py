"""Repair old-90 replay determinism with registered episode-local metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.decision_state import stable_fingerprint
from wmagentattack.runtime_metadata_v2 import (
    EpisodeLocalMetadataNormalizer,
    load_runtime_metadata_registry,
)
from wmagentattack.state_storage_v2 import (
    ContentAddressedStateStore,
    ModelTower,
    StateBlobReference,
)
from wmagentattack.structured_ledger_v2 import load_adapter_registry


def _load_v1_module():
    path = ROOT / "scripts" / "130_regress_old90_extraction_state_v2.py"
    specification = importlib.util.spec_from_file_location("old90_v1_regression", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load v1 regression helpers from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


V1 = _load_v1_module()


def _verify_source_markers(registry: Any) -> list[dict[str, Any]]:
    rows = []
    for evidence in registry.source_evidence:
        path = ROOT / evidence["path"]
        text = path.read_text(encoding="utf-8")
        matches = evidence["required_marker"] in text
        rows.append(
            {
                "path": evidence["path"],
                "symbol": evidence["symbol"],
                "required_marker": evidence["required_marker"],
                "matches": matches,
                "sha256": V1._sha256(path),
            }
        )
    return rows


def _semantic_projection(
    episode: dict[str, Any],
    store: ContentAddressedStateStore,
    metadata_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalizer = EpisodeLocalMetadataNormalizer(metadata_registry)
    transitions = []
    for transition in episode["transitions"]:
        state = transition["state_transition"]
        before_ref = StateBlobReference.model_validate(state["state_before_ref"])
        after_ref = StateBlobReference.model_validate(state["state_after_ref"])
        initial_ref = StateBlobReference.model_validate(state["initial_state_ref"])
        initial_state = store.get(initial_ref, requesting_tower=ModelTower.SIMULATOR)
        state_before = store.get(before_ref, requesting_tower=ModelTower.SIMULATOR)
        state_after = store.get(after_ref, requesting_tower=ModelTower.SIMULATOR)
        semantic_initial = normalizer.semantic_fingerprint(initial_state)
        semantic_before = normalizer.semantic_fingerprint(state_before)
        newly_bound = normalizer.observe_transition(
            tool_name=transition["tool_name"],
            call_index=int(transition["call_index"]),
            runtime_output=transition["structured_records"],
            exact_delta=state["exact_delta"],
        )
        semantic_after = normalizer.semantic_fingerprint(state_after)
        normalized_delta = normalizer.normalize(state["exact_delta"])
        normalized_records = normalizer.normalize(transition["structured_records"])
        normalized_conflicts = normalizer.normalize(transition["new_conflicts"])
        transitions.append(
            {
                "call_index": transition["call_index"],
                "tool_name": transition["tool_name"],
                "execution_status": transition["execution_status"],
                "semantic_initial_state_fingerprint": semantic_initial,
                "semantic_state_before_fingerprint": semantic_before,
                "semantic_state_after_fingerprint": semantic_after,
                "semantic_delta": normalized_delta,
                "semantic_structured_records": normalized_records,
                "semantic_new_conflicts": normalized_conflicts,
                "new_runtime_metadata_tokens": list(newly_bound),
            }
        )
    projection = {
        "episode_id": episode["episode_id"],
        "panel": episode["panel"],
        "seed": episode["seed"],
        "user_task_id": episode["user_task_id"],
        "source_trace_sha256": episode["source_trace_sha256"],
        "pairing": episode["pairing"],
        "transitions": transitions,
        "checks": episode["checks"],
        "observed_tools": episode["observed_tools"],
        "runtime_metadata_manifest": normalizer.public_manifest(),
    }
    return projection, normalizer.public_manifest()


def _run_pass(
    descriptors: list[dict[str, Any]],
    *,
    suite: Any,
    functions_runtime_type: Any,
    adapter_registry: Any,
    metadata_registry: Any,
    store: ContentAddressedStateStore,
) -> tuple[list[dict], list[dict], list[dict]]:
    raw_episodes = []
    semantic_episodes = []
    failures = []
    for descriptor in descriptors:
        episode_id = (
            f"{descriptor['panel']}::{descriptor['seed']}::{descriptor['user_task_id']}"
        )
        try:
            raw = V1._build_episode(
                descriptor,
                suite=suite,
                functions_runtime_type=functions_runtime_type,
                registry=adapter_registry,
                store=store,
            )
            semantic, _ = _semantic_projection(raw, store, metadata_registry)
            raw_episodes.append(raw)
            semantic_episodes.append(semantic)
        except Exception as error:  # preserve repair counterevidence
            failures.append(
                {
                    "episode_id": episode_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    return raw_episodes, semantic_episodes, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--metadata-registry", type=Path, required=True)
    parser.add_argument("--test-junit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "preregistered_after_v1_diagnosis_before_repair_execution"
    ):
        raise ValueError("semantic replay repair protocol was not preregistered")
    adapter_registry = load_adapter_registry(args.registry)
    metadata_registry = load_runtime_metadata_registry(args.metadata_registry)
    if metadata_registry.status != (
        "frozen_after_v1_nondeterminism_diagnosis_before_repair_execution"
    ):
        raise ValueError("runtime metadata registry is not frozen")
    source_markers = _verify_source_markers(metadata_registry)
    descriptors, source_chunks = V1._load_source_descriptors(protocol)
    test_summary = V1._junit_summary(args.test_junit)

    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(protocol["benchmark_version"], protocol["suite"])
    if set(adapter_registry.adapters) != {tool.name for tool in suite.tools}:
        raise ValueError("adapter registry differs from suite tool set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ContentAddressedStateStore(args.output_dir / "state_store")
    first_raw, first_semantic, first_failures = _run_pass(
        descriptors,
        suite=suite,
        functions_runtime_type=FunctionsRuntime,
        adapter_registry=adapter_registry,
        metadata_registry=metadata_registry,
        store=store,
    )
    second_raw, second_semantic, second_failures = _run_pass(
        descriptors,
        suite=suite,
        functions_runtime_type=FunctionsRuntime,
        adapter_registry=adapter_registry,
        metadata_registry=metadata_registry,
        store=store,
    )

    first_raw_by_id = {row["episode_id"]: row for row in first_raw}
    second_raw_by_id = {row["episode_id"]: row for row in second_raw}
    first_semantic_by_id = {row["episode_id"]: row for row in first_semantic}
    second_semantic_by_id = {row["episode_id"]: row for row in second_semantic}
    common_ids = set(first_raw_by_id) & set(second_raw_by_id)
    raw_mismatch_ids = {
        episode_id
        for episode_id in common_ids
        if stable_fingerprint(first_raw_by_id[episode_id])
        != stable_fingerprint(second_raw_by_id[episode_id])
    }
    semantic_mismatch_ids = {
        episode_id
        for episode_id in common_ids
        if stable_fingerprint(first_semantic_by_id[episode_id])
        != stable_fingerprint(second_semantic_by_id[episode_id])
    }
    metadata_episode_ids = {
        row["episode_id"]
        for row in first_semantic
        if row["runtime_metadata_manifest"]["binding_count"] > 0
    }
    first_binding_count = sum(
        row["runtime_metadata_manifest"]["binding_count"] for row in first_semantic
    )
    second_binding_count = sum(
        row["runtime_metadata_manifest"]["binding_count"] for row in second_semantic
    )
    expected = protocol["expected_counts"]
    proposals = sum(row["pairing"]["proposals"] for row in first_raw)
    executed = sum(row["pairing"]["executed_calls"] for row in first_raw)
    terminal = sum(row["pairing"]["terminal_unexecuted"] for row in first_raw)
    tasks = {row["user_task_id"] for row in first_raw}
    checks = [row["checks"] for row in first_raw]
    records = [
        record
        for episode in first_raw
        for record in episode["final_ledger"]["records"]
    ]
    conflicts = [
        conflict
        for episode in first_raw
        for conflict in episode["final_ledger"]["conflicts"]
    ]
    unlinked_conflicts = sum(
        conflict["entity_id"].startswith("PROVISIONAL::") for conflict in conflicts
    )
    forbidden_paths = V1._forbidden_paths(first_raw)
    gates = {
        "source_chunks_and_episode_counts_exact": (
            len(source_chunks) == int(expected["source_chunks"])
            and len(descriptors) == int(expected["episodes"])
            and len(first_raw) == int(expected["episodes"])
            and len(second_raw) == int(expected["episodes"])
        ),
        "task_count_exact": len(tasks) == int(expected["independent_tasks"]),
        "pairing_counts_exact": (
            proposals == int(expected["proposals"])
            and executed == int(expected["executed_calls"])
            and terminal == int(expected["terminal_unexecuted"])
        ),
        "zero_runtime_or_adapter_failures": not first_failures and not second_failures,
        "logged_execution_status_exact": all(
            row["logged_execution_status_exact"] for row in checks
        ),
        "runtime_metadata_source_markers_exact": all(
            row["matches"] for row in source_markers
        ),
        "raw_mismatch_count_matches_diagnosis": len(raw_mismatch_ids)
        == int(expected["raw_mismatched_episodes"]),
        "registered_metadata_binding_count_exact": (
            first_binding_count
            == int(expected["registered_metadata_bindings_per_pass"])
            and second_binding_count
            == int(expected["registered_metadata_bindings_per_pass"])
        ),
        "raw_differences_explained_by_registered_metadata": (
            raw_mismatch_ids == metadata_episode_ids and not semantic_mismatch_ids
        ),
        "deterministic_two_pass_semantic_replay": (
            set(first_semantic_by_id) == set(second_semantic_by_id)
            and not semantic_mismatch_ids
        ),
        "state_fingerprint_roundtrip_exact": all(
            row["state_fingerprint_roundtrip_exact"] for row in checks
        ),
        "ledger_replay_idempotent": all(
            row["ledger_replay_idempotent"] for row in checks
        ),
        "terminal_unexecuted_updates_zero": all(
            row["terminal_unexecuted_updates_zero"] for row in checks
        ),
        "unlinked_derived_conflicts_zero": unlinked_conflicts == 0,
        "all_records_victim_observed": all(
            row["observation_scope"] == "VICTIM_OBSERVED" for row in records
        ),
        "future_or_outcome_fields_zero": not forbidden_paths,
        "tests_pass": (
            test_summary["tests"] > 0
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
        ),
    }
    if protocol["frozen_gates"] != {name: True for name in gates}:
        raise ValueError("runtime gate names differ from frozen repair protocol")
    decision = protocol["pass_decision"] if all(gates.values()) else protocol["failure_decision"]

    with (args.output_dir / "raw_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in first_raw:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    with (args.output_dir / "semantic_episodes.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for episode in first_semantic:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    V1._write_json(args.output_dir / "source_manifest.json", source_chunks)
    audit = {
        "protocol_id": protocol["protocol_id"],
        "prior_decision": protocol["prior_decision"],
        "decision": decision,
        "gates": gates,
        "counts": {
            "source_chunks": len(source_chunks),
            "episodes_per_pass": len(first_raw),
            "independent_tasks": len(tasks),
            "proposals": proposals,
            "executed_calls": executed,
            "terminal_unexecuted": terminal,
            "structured_records": len(records),
            "conflicts": len(conflicts),
            "raw_mismatched_episodes": len(raw_mismatch_ids),
            "semantic_mismatched_episodes": len(semantic_mismatch_ids),
            "registered_metadata_bindings_pass1": first_binding_count,
            "registered_metadata_bindings_pass2": second_binding_count,
            "state_blobs": len(
                list((args.output_dir / "state_store" / "blobs").rglob("*.json"))
            ),
        },
        "raw_mismatch_episode_ids": sorted(raw_mismatch_ids),
        "semantic_mismatch_episode_ids": sorted(semantic_mismatch_ids),
        "registered_metadata_episode_ids": sorted(metadata_episode_ids),
        "source_marker_audit": source_markers,
        "test_summary": test_summary,
        "failures": {"pass1": first_failures, "pass2": second_failures},
        "claim_boundary": protocol["claim_boundary"],
        "safety": {
            "raw_exact_state_retained": True,
            "outcome_fields_accessed": False,
            "expert_trajectory_accessed": False,
            "model_training": False,
            "model_comparison": False,
            "victim_model_calls": 0,
            "attacks": 0,
            "dreamer": False,
        },
    }
    V1._write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
