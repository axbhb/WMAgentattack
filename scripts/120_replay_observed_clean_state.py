"""Replay frozen clean victim calls through exact AgentDojo state transitions.

The audit consumes existing clean traces only. It does not load a victim model,
construct attacks, contact external endpoints, or emit training examples. Raw
tool outputs and assistant text are intentionally replaced by fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import (
    build_ground_truth_goal_slots,
    canonical_call_signature,
    instrument_function_call,
    match_completed_goal_slots,
)
from wmagentattack.decision_state import canonical_json_value, stable_fingerprint
from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


SCOPE = "frozen existing clean victim trace exact-state replay"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        value = block.get("content", block.get("text", ""))
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def _clean_contract(trace: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures = []
    if trace.get("attack_type") not in (None, "none"):
        failures.append("attack_type")
    if trace.get("injection_task_id") not in (None, "none"):
        failures.append("injection_task_id")
    if trace.get("injections") not in (None, {}, []):
        failures.append("injections")
    if not isinstance(trace.get("messages"), list):
        failures.append("messages")
    return not failures, tuple(failures)


def _extract_calls_and_output(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    assistant_calls = []
    tool_messages = []
    final_output = ""
    multi_call_messages = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            calls = list(message.get("tool_calls") or [])
            if len(calls) > 1:
                multi_call_messages += 1
            for call in calls:
                assistant_calls.append(
                    {
                        "function": str(call["function"]),
                        "args": canonical_json_value(call.get("args") or {}),
                    }
                )
            final_output = _message_text(message)
        elif role == "tool":
            call = message.get("tool_call")
            if isinstance(call, dict):
                tool_messages.append(
                    {
                        "function": str(call["function"]),
                        "args": canonical_json_value(call.get("args") or {}),
                        "has_error": bool(message.get("error")),
                    }
                )
    return assistant_calls, tool_messages, final_output, multi_call_messages


def _calls_align(
    assistant_calls: list[dict[str, Any]], tool_messages: list[dict[str, Any]]
) -> bool:
    if len(assistant_calls) != len(tool_messages):
        return False
    return all(
        canonical_call_signature(call["function"], call["args"])
        == canonical_call_signature(tool["function"], tool["args"])
        for call, tool in zip(assistant_calls, tool_messages, strict=True)
    )


def _load_source_rows(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    source_rows = []
    input_chunks = []
    panel_counts = {}
    for panel in protocol["panels"]:
        name = str(panel["name"])
        root = Path(panel["archive_root"])
        panel_rows = []
        chunk_paths = []
        for seed in panel["seeds"]:
            seed_paths = sorted((root / f"seed{seed}").glob("chunk*.json"))
            if len(seed_paths) != int(panel["chunks_per_seed"]):
                raise ValueError(
                    f"{name} seed {seed} has {len(seed_paths)} chunks, expected "
                    f"{panel['chunks_per_seed']}"
                )
            for chunk_path in seed_paths:
                payload = json.loads(chunk_path.read_text(encoding="utf-8"))
                if int(payload["run_seed"]) != int(seed):
                    raise ValueError(f"seed mismatch in {chunk_path}")
                chunk_paths.append(chunk_path)
                input_chunks.append(
                    {
                        "panel": name,
                        "seed": int(seed),
                        "path": str(chunk_path),
                        "sha256": _sha256(chunk_path),
                    }
                )
                for result in payload["results"]:
                    if result.get("status") != "completed":
                        raise ValueError(f"incomplete source result in {chunk_path}")
                    panel_rows.append(
                        {
                            "panel": name,
                            "seed": int(seed),
                            "chunk_path": str(chunk_path),
                            "result": result,
                        }
                    )
        if len(panel_rows) != int(panel["expected_episodes"]):
            raise ValueError(
                f"{name} has {len(panel_rows)} episodes, expected "
                f"{panel['expected_episodes']}"
            )
        panel_counts[name] = {
            "episodes": len(panel_rows),
            "utility_successes": sum(
                bool(row["result"]["utility"]) for row in panel_rows
            ),
            "expected_utility_successes": int(panel["expected_utility_successes"]),
            "chunk_count": len(chunk_paths),
        }
        source_rows.extend(panel_rows)
    return source_rows, {
        "input_chunks": input_chunks,
        "panel_counts": panel_counts,
    }


def _utility(task, output, pre_environment, environment, calls) -> tuple[bool, str]:
    trace_result = task.utility_from_traces(
        output, pre_environment, environment, calls
    )
    if trace_result is not None:
        return bool(trace_result), "utility_from_traces"
    return bool(task.utility(output, pre_environment, environment)), "utility"


def _resolution_counts(links) -> dict[str, int]:
    counts = Counter(link.resolution for link in links)
    return {
        key: int(counts.get(key, 0))
        for key in ("no_match", "unique", "ambiguous")
    }


def _delta_roots(delta: tuple[dict[str, Any], ...]) -> dict[str, int]:
    roots = Counter()
    for change in delta:
        path = str(change.get("path", ""))
        root = "/" + path.lstrip("/").split("/", 1)[0] if path else "<root>"
        roots[root] += 1
    return dict(sorted(roots.items()))


def _replay_episode(
    source: dict[str, Any],
    *,
    suite,
    functions_runtime_type,
    function_call_type,
) -> dict[str, Any]:
    result = source["result"]
    trace_path = Path(result["raw_trace"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    clean, contract_failures = _clean_contract(trace)
    if not clean:
        raise ValueError(
            f"non-clean trace {trace_path}: {','.join(contract_failures)}"
        )
    if trace.get("benchmark_version") != "v1.2.2":
        raise ValueError(f"benchmark version mismatch in {trace_path}")
    if trace.get("suite_name") != "travel" or result["suite"] != "travel":
        raise ValueError(f"suite mismatch in {trace_path}")
    if trace.get("user_task_id") != result["user_task_id"]:
        raise ValueError(f"task mismatch in {trace_path}")
    if bool(trace.get("utility")) != bool(result["utility"]):
        raise ValueError(f"source utility mismatch in {trace_path}")

    pairing, final_output = pair_executed_clean_tool_calls(trace["messages"])
    raw_calls = [
        {
            "function": pair.proposal.function,
            "args": pair.proposal.arguments,
        }
        for pair in pairing.executed_pairs
    ]
    aligned = pairing.executed_alignment_ok
    calls = [
        function_call_type(function=call["function"], args=call["args"])
        for call in raw_calls
    ]
    task = suite.get_user_task_by_id(result["user_task_id"])
    environment = suite.load_and_inject_default_environment({})
    environment = task.init_environment(environment)
    pre_environment = environment.model_copy(deep=True)
    initial_fingerprint = stable_fingerprint(
        pre_environment.model_dump(mode="json")
    )
    runtime = functions_runtime_type(suite.tools)
    transitions = []
    runtime_error_alignment_failures = 0
    transition_outcome_labels_present = 0
    for index, call in enumerate(calls):
        transition, _ = instrument_function_call(
            runtime,
            environment,
            event_index=index,
            function=call.function,
            arguments=call.args,
        )
        if transition.outcome_labels_present:
            transition_outcome_labels_present += 1
        logged_error = pairing.executed_pairs[index].logged_error
        runtime_error = transition.tool_execution_status == "error"
        if bool(logged_error) != runtime_error:
            runtime_error_alignment_failures += 1
        transitions.append(
            {
                "event_index": index,
                "tool_schema_id": transition.tool_schema_id,
                "tool_execution_status": transition.tool_execution_status,
                "tool_error_type": transition.tool_error_type,
                "normalized_argument_slot_count": len(
                    transition.normalized_argument_slots
                ),
                "argument_entity_link_resolution": _resolution_counts(
                    transition.argument_entity_links
                ),
                "state_before_fingerprint": transition.state_before_fingerprint,
                "state_after_fingerprint": transition.state_after_fingerprint,
                "state_changed": transition.state_changed,
                "state_delta_operation_count": len(
                    transition.canonical_state_delta
                ),
                "state_delta_roots": _delta_roots(
                    transition.canonical_state_delta
                ),
                "task_progress_delta": None,
                "irreversible_effect": None,
                "outcome_labels_present": False,
            }
        )

    replay_utility, utility_source = _utility(
        task, final_output, pre_environment, environment, calls
    )
    expert_calls = list(task.ground_truth(pre_environment.model_copy(deep=True)))
    goal_slots = build_ground_truth_goal_slots(expert_calls, pre_environment)
    completed_slots, remaining_slots = match_completed_goal_slots(calls, goal_slots)
    final_fingerprint = stable_fingerprint(environment.model_dump(mode="json"))
    call_signatures = [
        canonical_call_signature(call.function, call.args) for call in calls
    ]
    return {
        "status": "replayed",
        "panel": source["panel"],
        "seed": source["seed"],
        "row_id": result["row_id"],
        "suite": result["suite"],
        "user_task_id": result["user_task_id"],
        "source_trace_sha256": _sha256(trace_path),
        "source_clean_contract": True,
        "assistant_tool_proposal_count": pairing.proposal_count,
        "executed_tool_call_count": len(calls),
        "tool_message_count": pairing.tool_message_count,
        "tool_message_alignment": aligned,
        "terminal_unexecuted_proposal_count": len(
            pairing.terminal_unexecuted_proposals
        ),
        "terminal_unexecuted_tool_names": [
            proposal.function
            for proposal in pairing.terminal_unexecuted_proposals
        ],
        "midtrajectory_unexecuted_proposal_count": len(
            pairing.midtrajectory_unexecuted_proposals
        ),
        "orphan_tool_message_count": len(
            pairing.orphan_tool_message_indices
        ),
        "signature_mismatch_tool_message_count": len(
            pairing.signature_mismatch_tool_message_indices
        ),
        "assistant_multi_call_message_count": (
            pairing.assistant_multi_call_message_count
        ),
        "runtime_error_alignment_failures": runtime_error_alignment_failures,
        "transition_outcome_labels_present_count": transition_outcome_labels_present,
        "archived_utility": bool(result["utility"]),
        "recomputed_utility": replay_utility,
        "utility_matches_archive": replay_utility == bool(result["utility"]),
        "utility_source": utility_source,
        "model_output_fingerprint": stable_fingerprint(final_output),
        "initial_state_fingerprint": initial_fingerprint,
        "final_state_fingerprint": final_fingerprint,
        "state_changed_from_initial": initial_fingerprint != final_fingerprint,
        "call_path_tools": [call.function for call in calls],
        "exact_call_path_fingerprint": stable_fingerprint(call_signatures),
        "expert_goal_slot_count": len(goal_slots),
        "completed_exact_expert_goal_slot_count": len(completed_slots),
        "remaining_exact_expert_goal_slot_count": len(remaining_slots),
        "exact_expert_goal_slot_fraction": (
            len(completed_slots) / len(goal_slots) if goal_slots else 1.0
        ),
        "transitions": transitions,
    }


def _mixed_groups(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups[key].append(row)
    mixed = []
    for key, members in groups.items():
        outcomes = {bool(row["archived_utility"]) for row in members}
        if len(outcomes) > 1:
            mixed.append(
                {
                    "key": list(key),
                    "episodes": len(members),
                    "successes": sum(bool(row["archived_utility"]) for row in members),
                    "panels_and_seeds": sorted(
                        f"{row['panel']}:{row['seed']}" for row in members
                    ),
                }
            )
    return sorted(mixed, key=lambda row: (-row["episodes"], row["key"]))


def _aggregate(rows: list[dict], source_audit: dict) -> dict[str, Any]:
    replayed = [row for row in rows if row.get("status") == "replayed"]
    transitions = [
        transition for row in replayed for transition in row["transitions"]
    ]
    entity = Counter()
    delta_roots = Counter()
    for transition in transitions:
        entity.update(transition["argument_entity_link_resolution"])
        delta_roots.update(transition["state_delta_roots"])
    successes = [row for row in replayed if row["archived_utility"]]
    failures = [row for row in replayed if not row["archived_utility"]]

    per_task = {}
    for task_id in sorted({row["user_task_id"] for row in replayed}):
        task_rows = [row for row in replayed if row["user_task_id"] == task_id]
        per_task[task_id] = {
            "episodes": len(task_rows),
            "utility_successes": sum(
                bool(row["archived_utility"]) for row in task_rows
            ),
            "unique_exact_call_paths": len(
                {row["exact_call_path_fingerprint"] for row in task_rows}
            ),
            "unique_final_states": len(
                {row["final_state_fingerprint"] for row in task_rows}
            ),
            "episodes_with_state_change": sum(
                row["state_changed_from_initial"] for row in task_rows
            ),
            "mean_exact_expert_goal_slot_fraction": statistics.fmean(
                row["exact_expert_goal_slot_fraction"] for row in task_rows
            ),
        }

    mixed_final = _mixed_groups(
        replayed, ("user_task_id", "final_state_fingerprint")
    )
    mixed_path = _mixed_groups(
        replayed, ("user_task_id", "exact_call_path_fingerprint")
    )
    mixed_path_state = _mixed_groups(
        replayed,
        (
            "user_task_id",
            "exact_call_path_fingerprint",
            "final_state_fingerprint",
        ),
    )
    return {
        "source_episode_count": len(rows),
        "replayed_episode_count": len(replayed),
        "replay_infrastructure_failure_count": len(rows) - len(replayed),
        "source_panel_counts": source_audit["panel_counts"],
        "task_count": len({row["user_task_id"] for row in replayed}),
        "seed_count": len({row["seed"] for row in replayed}),
        "archived_utility_successes": len(successes),
        "archived_utility_failures": len(failures),
        "utility_recomputation_mismatches": sum(
            not row["utility_matches_archive"] for row in replayed
        ),
        "tool_message_alignment_failures": sum(
            not row["tool_message_alignment"] for row in replayed
        ),
        "runtime_error_alignment_failures": sum(
            row["runtime_error_alignment_failures"] for row in replayed
        ),
        "assistant_multi_call_message_count": sum(
            row["assistant_multi_call_message_count"] for row in replayed
        ),
        "assistant_tool_proposal_count": sum(
            row["assistant_tool_proposal_count"] for row in replayed
        ),
        "terminal_unexecuted_proposal_count": sum(
            row["terminal_unexecuted_proposal_count"] for row in replayed
        ),
        "midtrajectory_unexecuted_proposal_count": sum(
            row["midtrajectory_unexecuted_proposal_count"] for row in replayed
        ),
        "orphan_tool_message_count": sum(
            row["orphan_tool_message_count"] for row in replayed
        ),
        "signature_mismatch_tool_message_count": sum(
            row["signature_mismatch_tool_message_count"] for row in replayed
        ),
        "transition_outcome_labels_present_count": sum(
            row["transition_outcome_labels_present_count"] for row in replayed
        ),
        "tool_call_count": len(transitions),
        "episodes_without_tool_calls": sum(
            row["executed_tool_call_count"] == 0 for row in replayed
        ),
        "runtime_tool_error_call_count": sum(
            transition["tool_execution_status"] == "error"
            for transition in transitions
        ),
        "state_mutating_call_count": sum(
            transition["state_changed"] for transition in transitions
        ),
        "read_only_call_count": sum(
            not transition["state_changed"] for transition in transitions
        ),
        "episodes_with_state_change": sum(
            row["state_changed_from_initial"] for row in replayed
        ),
        "successful_episodes_with_state_change": sum(
            row["state_changed_from_initial"] for row in successes
        ),
        "failed_episodes_with_state_change": sum(
            row["state_changed_from_initial"] for row in failures
        ),
        "mean_exact_expert_goal_slot_fraction": statistics.fmean(
            (row["exact_expert_goal_slot_fraction"] for row in replayed)
        )
        if replayed
        else 0.0,
        "mean_exact_expert_goal_slot_fraction_success": statistics.fmean(
            (row["exact_expert_goal_slot_fraction"] for row in successes)
        )
        if successes
        else 0.0,
        "mean_exact_expert_goal_slot_fraction_failure": statistics.fmean(
            (row["exact_expert_goal_slot_fraction"] for row in failures)
        )
        if failures
        else 0.0,
        "argument_entity_link_resolution": {
            key: int(entity.get(key, 0))
            for key in ("no_match", "unique", "ambiguous")
        },
        "state_delta_roots": dict(delta_roots.most_common()),
        "mixed_utility_same_final_state_groups": mixed_final,
        "mixed_utility_same_exact_call_path_groups": mixed_path,
        "mixed_utility_same_exact_call_path_and_final_state_groups": mixed_path_state,
        "per_task": per_task,
    }


def _decision(protocol: dict[str, Any], overall: dict[str, Any]) -> dict:
    expected_episodes = int(protocol["fixed_budget"]["episodes"])
    expected_tasks = set(protocol["expected_task_ids"])
    observed_tasks = set(overall["per_task"])
    panels = overall["source_panel_counts"]
    panel_integrity = all(
        values["episodes"] == next(
            int(panel["expected_episodes"])
            for panel in protocol["panels"]
            if panel["name"] == name
        )
        and values["utility_successes"] == values["expected_utility_successes"]
        for name, values in panels.items()
    )
    gates = {
        "exactly_90_complete_source_episodes": (
            overall["source_episode_count"] == expected_episodes
            and overall["replayed_episode_count"] == expected_episodes
        ),
        "all_sources_are_clean_sandbox_traces": (
            overall["replay_infrastructure_failure_count"] == 0
        ),
        "all_logged_calls_align_with_tool_messages": (
            overall["tool_message_alignment_failures"] == 0
            and overall["runtime_error_alignment_failures"] == 0
            and overall["midtrajectory_unexecuted_proposal_count"] == 0
            and overall["orphan_tool_message_count"] == 0
            and overall["signature_mismatch_tool_message_count"] == 0
        ),
        "zero_replay_infrastructure_failures": (
            overall["replay_infrastructure_failure_count"] == 0
        ),
        "all_recomputed_final_utilities_match_archived_utilities": (
            overall["utility_recomputation_mismatches"] == 0
        ),
        "zero_outcome_labels_inside_transition_records": (
            overall["transition_outcome_labels_present_count"] == 0
        ),
        "all_expected_tasks_and_panels_present": (
            observed_tasks == expected_tasks and panel_integrity
        ),
    }
    ready = all(gates.values())
    repaired_pairing = "pairing_repair" in str(protocol["protocol_id"])
    return {
        "gates": gates,
        "observed_clean_state_replay_ready": ready,
        "clean_data_gate": "BLOCKED",
        "attack_data_permitted": False,
        "dreamer_training_permitted": False,
        "decision": (
            (
                "OBSERVED_CLEAN_EXECUTED_CALL_PAIRING_READY_CLEAN_GATE_BLOCKED"
                if repaired_pairing
                else "OBSERVED_CLEAN_STATE_REPLAY_READY_CLEAN_GATE_BLOCKED"
            )
            if ready
            else (
                "OBSERVED_CLEAN_EXECUTED_CALL_PAIRING_NOT_READY_CLEAN_GATE_BLOCKED"
                if repaired_pairing
                else "OBSERVED_CLEAN_STATE_REPLAY_NOT_READY_CLEAN_GATE_BLOCKED"
            )
        ),
        "next_admissible_step": (
            "instrument a separately frozen stronger clean victim/task panel"
            if ready
            else "repair replay integrity without new victim calls"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("protocol was not preregistered")

    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    sources, source_audit = _load_source_rows(protocol)
    suite = get_suite(protocol["scope"]["benchmark_version"], "travel")
    rows = []
    for source in sources:
        try:
            rows.append(
                _replay_episode(
                    source,
                    suite=suite,
                    functions_runtime_type=FunctionsRuntime,
                    function_call_type=FunctionCall,
                )
            )
        except Exception as error:
            rows.append(
                {
                    "status": "replay_failed",
                    "panel": source["panel"],
                    "seed": source["seed"],
                    "row_id": source["result"].get("row_id"),
                    "user_task_id": source["result"].get("user_task_id"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    overall = _aggregate(rows, source_audit)
    decision = _decision(protocol, overall)
    report = {
        "protocol_id": protocol["protocol_id"],
        "scope": SCOPE,
        "safety_contract": {
            "existing_clean_traces_only": True,
            "llm_loaded": False,
            "attacks_constructed": False,
            "external_endpoints": False,
            "training_examples_created": False,
            "raw_tool_outputs_stored": False,
            "assistant_text_stored": False,
        },
        "source_audit": source_audit,
        "overall": overall,
        "decision": decision,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"overall": overall, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
