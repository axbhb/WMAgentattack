"""Run ledger/state engineering regression on old clean traces without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import instrument_function_call
from wmagentattack.decision_state import canonical_json_value, stable_fingerprint
from wmagentattack.state_storage_v2 import (
    ContentAddressedStateStore,
    ModelTower,
    build_exact_state_transition,
)
from wmagentattack.structured_ledger_v2 import (
    ExecutionChannelStatus,
    StructuredEvidenceLedgerV2,
    load_adapter_registry,
    update_structured_ledger,
)
from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


FORBIDDEN_OUTPUT_KEYS = {
    "utility",
    "security",
    "task_success",
    "attack_success",
    "expert_slot_coverage",
    "ground_truth",
    "final_answer",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _junit_summary(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit report contains no test suites")
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = (*path, str(key))
            if str(key).lower() in FORBIDDEN_OUTPUT_KEYS:
                found.append(".".join(child))
            found.extend(_forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_paths(item, (*path, str(index))))
    return found


def _load_source_descriptors(protocol: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    descriptors = []
    chunks = []
    for panel in protocol["source_panels"]:
        panel_count = 0
        archive_root = Path(panel["archive_root"])
        for seed in panel["seeds"]:
            paths = sorted((archive_root / f"seed{seed}").glob("chunk*.json"))
            if len(paths) != int(panel["chunks_per_seed"]):
                raise ValueError(
                    f"unexpected source chunk count for {panel['name']} seed {seed}"
                )
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload["run_seed"]) != int(seed):
                    raise ValueError(f"source seed mismatch in {path}")
                chunks.append(
                    {
                        "panel": panel["name"],
                        "seed": int(seed),
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                )
                for result in payload["results"]:
                    if result.get("status") != "completed":
                        raise ValueError(f"incomplete source result in {path}")
                    descriptors.append(
                        {
                            "panel": panel["name"],
                            "seed": int(seed),
                            "archive_root": str(archive_root),
                            "user_task_id": str(result["user_task_id"]),
                            "raw_trace": str(result["raw_trace"]),
                        }
                    )
                    panel_count += 1
        if panel_count != int(panel["expected_episodes"]):
            raise ValueError(f"unexpected source episode count for {panel['name']}")
    return descriptors, chunks


def _clean_trace_contract(trace: Mapping[str, Any]) -> None:
    if trace.get("suite_name") != "travel":
        raise ValueError("source trace is not Travel")
    if trace.get("attack_type") not in (None, "none"):
        raise ValueError("source trace is not clean")
    if trace.get("injection_task_id") not in (None, "none"):
        raise ValueError("source trace has an injection task")
    if trace.get("injections") not in (None, {}, []):
        raise ValueError("source trace has injections")


def _trace_path(descriptor: Mapping[str, Any]) -> Path:
    path = Path(descriptor["raw_trace"])
    return path if path.is_absolute() else Path(descriptor["archive_root"]) / path


def _build_episode(
    descriptor: Mapping[str, Any],
    *,
    suite: Any,
    functions_runtime_type: Any,
    registry: Any,
    store: ContentAddressedStateStore,
) -> dict[str, Any]:
    episode_id = (
        f"{descriptor['panel']}::{descriptor['seed']}::{descriptor['user_task_id']}"
    )
    trace = json.loads(_trace_path(descriptor).read_text(encoding="utf-8"))
    _clean_trace_contract(trace)
    if str(trace.get("user_task_id")) != descriptor["user_task_id"]:
        raise ValueError("source task ID mismatch")
    pairing, _ = pair_executed_clean_tool_calls(trace["messages"])
    if not pairing.executed_alignment_ok:
        raise ValueError("executed-call pairing defect")

    task = suite.get_user_task_by_id(descriptor["user_task_id"])
    environment = task.init_environment(suite.load_and_inject_default_environment({}))
    initial_state = canonical_json_value(environment)
    runtime = functions_runtime_type(suite.tools)
    ledger = StructuredEvidenceLedgerV2()
    transitions = []
    status_matches = []
    state_roundtrips = []
    ledger_idempotence = []
    observed_tools = set()

    for call_index, pair in enumerate(pairing.executed_pairs):
        proposal = pair.proposal
        before_record_count = len(ledger.records)
        before_conflict_count = len(ledger.conflicts)
        transition, runtime_output = instrument_function_call(
            runtime,
            environment,
            event_index=call_index,
            function=proposal.function,
            arguments=proposal.arguments,
        )
        replay_error = transition.tool_execution_status == "error"
        status_matches.append(replay_error == pair.logged_error)
        state_record = build_exact_state_transition(
            store,
            episode_id=episode_id,
            call_index=call_index,
            initial_state=initial_state,
            state_before=transition.canonical_state_before,
            state_after=transition.canonical_state_after,
            exact_delta=transition.canonical_state_delta,
            execution_status=transition.tool_execution_status,
            error_type=transition.tool_error_type,
        )
        state_roundtrips.append(
            store.get(
                state_record.state_before_ref, requesting_tower=ModelTower.SIMULATOR
            )
            == transition.canonical_state_before
            and store.get(
                state_record.state_after_ref, requesting_tower=ModelTower.SIMULATOR
            )
            == transition.canonical_state_after
        )
        channel = (
            ExecutionChannelStatus.EXECUTED_ERROR
            if replay_error
            else ExecutionChannelStatus.EXECUTED_SUCCESS
        )
        result = update_structured_ledger(
            ledger,
            registry,
            episode_id=episode_id,
            call_index=call_index,
            channel_status=channel,
            tool_name=proposal.function,
            arguments=proposal.arguments,
            runtime_output=runtime_output,
            error_type=transition.tool_error_type,
            state_changed=transition.state_changed,
            proposal_signature=proposal.call_signature,
        )
        replay = update_structured_ledger(
            result.ledger,
            registry,
            episode_id=episode_id,
            call_index=call_index,
            channel_status=channel,
            tool_name=proposal.function,
            arguments=proposal.arguments,
            runtime_output=runtime_output,
            error_type=transition.tool_error_type,
            state_changed=transition.state_changed,
            proposal_signature=proposal.call_signature,
        )
        ledger_idempotence.append(replay.ledger == result.ledger)
        added_records = result.ledger.records[before_record_count:]
        added_conflicts = result.ledger.conflicts[before_conflict_count:]
        transitions.append(
            {
                "call_index": call_index,
                "tool_name": proposal.function,
                "execution_status": transition.tool_execution_status,
                "state_transition": state_record.model_dump(mode="json"),
                "structured_records": [
                    record.model_dump(mode="json") for record in added_records
                ],
                "new_conflicts": [
                    conflict.model_dump(mode="json") for conflict in added_conflicts
                ],
            }
        )
        observed_tools.add(proposal.function)
        ledger = result.ledger

    terminal_update_counts = []
    for offset, proposal in enumerate(pairing.terminal_unexecuted_proposals):
        result = update_structured_ledger(
            ledger,
            registry,
            episode_id=episode_id,
            call_index=len(pairing.executed_pairs) + offset,
            channel_status=ExecutionChannelStatus.TERMINAL_UNEXECUTED,
            tool_name=proposal.function,
            arguments=proposal.arguments,
            proposal_signature=proposal.call_signature,
        )
        terminal_update_counts.append(len(result.ledger.records) - len(ledger.records))
        if result.ledger != ledger:
            raise ValueError("terminal-unexecuted proposal changed the ledger")

    return {
        "episode_id": episode_id,
        "panel": descriptor["panel"],
        "seed": descriptor["seed"],
        "user_task_id": descriptor["user_task_id"],
        "source_trace_sha256": _sha256(_trace_path(descriptor)),
        "pairing": {
            "proposals": pairing.proposal_count,
            "executed_calls": len(pairing.executed_pairs),
            "terminal_unexecuted": len(pairing.terminal_unexecuted_proposals),
            "alignment_ok": pairing.executed_alignment_ok,
        },
        "transitions": transitions,
        "final_ledger": ledger.model_dump(mode="json"),
        "checks": {
            "logged_execution_status_exact": all(status_matches),
            "state_fingerprint_roundtrip_exact": all(state_roundtrips),
            "ledger_replay_idempotent": all(ledger_idempotence),
            "terminal_unexecuted_updates_zero": sum(terminal_update_counts) == 0,
        },
        "observed_tools": sorted(observed_tools),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--test-junit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("old-90 regression protocol was not preregistered")
    if int(protocol["determinism_passes"]) != 2:
        raise ValueError("this audit requires exactly two deterministic passes")
    registry = load_adapter_registry(args.registry)
    descriptors, source_chunks = _load_source_descriptors(protocol)
    test_summary = _junit_summary(args.test_junit)

    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(protocol["benchmark_version"], protocol["suite"])
    if set(registry.adapters) != {tool.name for tool in suite.tools}:
        raise ValueError("adapter registry differs from suite tool set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ContentAddressedStateStore(args.output_dir / "state_store")
    pass_outputs = []
    pass_failures = []
    for pass_index in range(2):
        episodes = []
        failures = []
        for descriptor in descriptors:
            try:
                episodes.append(
                    _build_episode(
                        descriptor,
                        suite=suite,
                        functions_runtime_type=FunctionsRuntime,
                        registry=registry,
                        store=store,
                    )
                )
            except Exception as error:  # preserve counterevidence in the audit
                failures.append(
                    {
                        "episode_id": (
                            f"{descriptor['panel']}::{descriptor['seed']}::"
                            f"{descriptor['user_task_id']}"
                        ),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
        pass_outputs.append(episodes)
        pass_failures.append(failures)

    first, second = pass_outputs
    first_by_id = {episode["episode_id"]: episode for episode in first}
    second_by_id = {episode["episode_id"]: episode for episode in second}
    deterministic_ids = set(first_by_id) == set(second_by_id)
    deterministic_replay = deterministic_ids and all(
        stable_fingerprint(first_by_id[episode_id])
        == stable_fingerprint(second_by_id[episode_id])
        for episode_id in first_by_id
    )
    expected = protocol["expected_counts"]
    proposals = sum(row["pairing"]["proposals"] for row in first)
    executed_calls = sum(row["pairing"]["executed_calls"] for row in first)
    terminal = sum(row["pairing"]["terminal_unexecuted"] for row in first)
    tasks = {row["user_task_id"] for row in first}
    all_checks = [row["checks"] for row in first]
    records = [
        record
        for episode in first
        for record in episode["final_ledger"]["records"]
    ]
    conflicts = [
        conflict
        for episode in first
        for conflict in episode["final_ledger"]["conflicts"]
    ]
    unlinked_conflicts = sum(
        conflict["entity_id"].startswith("PROVISIONAL::") for conflict in conflicts
    )
    forbidden_paths = _forbidden_paths(first)
    gates = {
        "source_chunks_and_episode_counts_exact": (
            len(source_chunks) == int(expected["source_chunks"])
            and len(descriptors) == int(expected["episodes"])
            and len(first) == int(expected["episodes"])
        ),
        "task_count_exact": len(tasks) == int(expected["independent_tasks"]),
        "pairing_counts_exact": (
            proposals == int(expected["proposals"])
            and executed_calls == int(expected["executed_calls"])
            and terminal == int(expected["terminal_unexecuted"])
        ),
        "zero_runtime_or_adapter_failures": not any(pass_failures),
        "logged_execution_status_exact": all(
            row["logged_execution_status_exact"] for row in all_checks
        ),
        "deterministic_two_pass_replay": deterministic_replay,
        "state_fingerprint_roundtrip_exact": all(
            row["state_fingerprint_roundtrip_exact"] for row in all_checks
        ),
        "ledger_replay_idempotent": all(
            row["ledger_replay_idempotent"] for row in all_checks
        ),
        "terminal_unexecuted_updates_zero": all(
            row["terminal_unexecuted_updates_zero"] for row in all_checks
        ),
        "unlinked_derived_conflicts_zero": unlinked_conflicts == 0,
        "all_records_victim_observed": all(
            record["observation_scope"] == "VICTIM_OBSERVED" for record in records
        ),
        "future_or_outcome_fields_zero": not forbidden_paths,
        "tests_pass": (
            test_summary["tests"] > 0
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
        ),
    }
    if protocol["frozen_gates"] != {name: True for name in gates}:
        raise ValueError("runtime gate names differ from frozen old-90 protocol")
    decision = protocol["pass_decision"] if all(gates.values()) else protocol["failure_decision"]
    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in first:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    _write_json(args.output_dir / "source_manifest.json", source_chunks)
    audit = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "gates": gates,
        "counts": {
            "source_chunks": len(source_chunks),
            "episodes": len(first),
            "independent_tasks": len(tasks),
            "proposals": proposals,
            "executed_calls": executed_calls,
            "terminal_unexecuted": terminal,
            "structured_records": len(records),
            "conflicts": len(conflicts),
            "unlinked_derived_conflicts": unlinked_conflicts,
            "state_blobs": len(
                list((args.output_dir / "state_store" / "blobs").rglob("*.json"))
            ),
            "pass1_failures": len(pass_failures[0]),
            "pass2_failures": len(pass_failures[1]),
        },
        "observed_tools": sorted(
            {tool for episode in first for tool in episode["observed_tools"]}
        ),
        "test_summary": test_summary,
        "failures": {"pass1": pass_failures[0], "pass2": pass_failures[1]},
        "claim_boundary": protocol["claim_boundary"],
        "safety": {
            "outcome_fields_accessed": False,
            "expert_trajectory_accessed": False,
            "model_training": False,
            "model_comparison": False,
            "victim_model_calls": 0,
            "attacks": 0,
            "dreamer": False,
        },
    }
    _write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
