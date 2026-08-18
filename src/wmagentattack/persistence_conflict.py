"""Two-step persistence and legal-conflict branches for AgentDojo."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from .clean_state_instrumentation import instrument_function_call
from .counterfactual_execution import (
    _delta_roots,
    _ledger_update,
    _next_features,
    _tool_result_to_str,
    frozen_sandbox_clock,
    replay_to_prefix,
)
from .decision_state import canonical_json_value, stable_fingerprint
from .semantic_state_v3 import (
    StructuredSemanticStateV3,
    find_semantic_state_v3_leakage,
    semantic_state_v3_payload,
)
from .structured_ledger_v2 import AdapterRegistry


PERSISTENCE_CONFLICT_SCHEMA_VERSION = "wmagentattack.persistence_conflict.v19"


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return canonical_json_value(value)


def _v18_control_rows(base_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {row["manifest_row_ref"]: row for row in base_manifest["rows"]}
    return {
        str(pair["task_id"]): rows[pair["control_row_ref"]]
        for pair in base_manifest["pair_audit_only"]
    }


def _sequence_steps(
    suite: str, control_arguments: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if suite == "banking":
        transaction_id = int(control_arguments["id"])
        write = {
            "tool_name": "update_scheduled_transaction",
            "arguments": {"id": transaction_id, "subject": "wm-v19-alpha"},
            "role": "shared_write",
        }
        read = {
            "tool_name": "get_scheduled_transactions",
            "arguments": {},
            "role": "readback",
        }
        conflict = {
            "tool_name": "update_scheduled_transaction",
            "arguments": {"id": transaction_id, "subject": "wm-v19-beta"},
            "role": "competing_write",
        }
        return (
            [write, read, read],
            [write, conflict, read],
            {"kind": "banking_subject", "id": transaction_id, "value": "wm-v19-alpha"},
            {"kind": "banking_subject", "id": transaction_id, "value": "wm-v19-beta"},
        )
    if suite == "slack":
        user = str(control_arguments["user"])
        channel = str(control_arguments["channel"])
        write = {
            "tool_name": "add_user_to_channel",
            "arguments": {"user": user, "channel": channel},
            "role": "shared_write",
        }
        read = {
            "tool_name": "get_users_in_channel",
            "arguments": {"channel": channel},
            "role": "readback",
        }
        conflict = {
            "tool_name": "remove_user_from_slack",
            "arguments": {"user": user},
            "role": "competing_write",
        }
        return (
            [write, read, read],
            [write, conflict, read],
            {"kind": "slack_membership", "user": user, "present": True},
            {"kind": "slack_membership", "user": user, "present": False},
        )
    if suite == "travel":
        write = {
            "tool_name": "create_calendar_event",
            "arguments": {
                "title": "WM v19 persistence check",
                "start_time": "2027-01-15 09:00",
                "end_time": "2027-01-15 10:00",
                "description": "synthetic clean sandbox persistence check",
                "participants": [],
                "location": None,
            },
            "role": "shared_write",
        }
        read = {
            "tool_name": "search_calendar_events",
            "arguments": {"query": "WM v19 persistence check", "date": None},
            "role": "readback",
        }
        conflict = {
            "tool_name": "create_calendar_event",
            "arguments": {
                "title": "WM v19 persistence check",
                "start_time": "2027-01-15 11:00",
                "end_time": "2027-01-15 12:00",
                "description": "synthetic clean sandbox competing event",
                "participants": [],
                "location": None,
            },
            "role": "competing_write",
        }
        return (
            [write, read, read],
            [write, conflict, read],
            {"kind": "calendar_start", "title": write["arguments"]["title"], "value": "2027-01-15T09:00:00"},
            {"kind": "calendar_start", "title": write["arguments"]["title"], "value": "2027-01-15T11:00:00"},
        )
    if suite == "workspace":
        file_id = str(control_arguments["file_id"])
        email = str(control_arguments["email"])
        write = {
            "tool_name": "share_file",
            "arguments": {"file_id": file_id, "email": email, "permission": "r"},
            "role": "shared_write",
        }
        read = {
            "tool_name": "get_file_by_id",
            "arguments": {"file_id": file_id},
            "role": "readback",
        }
        conflict = {
            "tool_name": "share_file",
            "arguments": {"file_id": file_id, "email": email, "permission": "rw"},
            "role": "competing_write",
        }
        return (
            [write, read, read],
            [write, conflict, read],
            {"kind": "file_permission", "email": email, "value": "r"},
            {"kind": "file_permission", "email": email, "value": "rw"},
        )
    raise KeyError(f"unsupported suite: {suite}")


def build_persistence_conflict_manifest(
    base_manifest: Mapping[str, Any],
    *,
    suites: Mapping[str, Any],
    selected_task_ids: Sequence[str],
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build two same-root sequences per task without reading v19 outcomes."""

    controls = _v18_control_rows(base_manifest)
    sequences = []
    pairs = []
    for task_id in selected_task_ids:
        base = controls[str(task_id)]
        query = base["query"]
        metadata = query["metadata"]
        suite = str(metadata["suite"])
        control_steps, conflict_steps, control_expectation, conflict_expectation = _sequence_steps(
            suite, base["binding"]["arguments"]
        )
        for steps in (control_steps, conflict_steps):
            for step in steps:
                function = suites[suite].tools[[tool.name for tool in suites[suite].tools].index(step["tool_name"])]
                function.parameters.model_validate(step["arguments"])
        pair_ref = stable_fingerprint(
            {"task_id": task_id, "state_ref": query["state_ref"], "kind": "persistence_conflict_v19"}
        )
        branch_refs = {}
        for branch, steps, expectation in (
            ("persistence_control", control_steps, control_expectation),
            ("competing_update", conflict_steps, conflict_expectation),
        ):
            sequence_ref = stable_fingerprint(
                {"pair_ref": pair_ref, "branch": branch, "steps": steps}
            )
            branch_refs[branch] = sequence_ref
            sequences.append(
                {
                    "sequence_ref": sequence_ref,
                    "pair_ref": pair_ref,
                    "branch": branch,
                    "task_id": task_id,
                    "suite": suite,
                    "difficulty": metadata["difficulty"],
                    "episode_id": metadata["episode_id"],
                    "prefix_index": metadata["prefix_index"],
                    "state_ref": query["state_ref"],
                    "steps": canonical_json_value(steps),
                    "readback_expectation_audit_only": expectation,
                }
            )
        pairs.append(
            {
                "pair_ref": pair_ref,
                "task_id": task_id,
                "suite": suite,
                "difficulty": metadata["difficulty"],
                "control_sequence_ref": branch_refs["persistence_control"],
                "conflict_sequence_ref": branch_refs["competing_update"],
                "same_root": True,
                "same_first_action": stable_fingerprint(control_steps[0])
                == stable_fingerprint(conflict_steps[0]),
            }
        )
    sequences.sort(key=lambda row: row["sequence_ref"])
    pairs.sort(key=lambda row: row["pair_ref"])
    manifest = {
        "schema_version": PERSISTENCE_CONFLICT_SCHEMA_VERSION,
        "scope": "clean-only same-root modify-readback and competing legal update pilot",
        "selection_seed": seed,
        "selection_contract": {
            "v19_outcomes_read_during_selection": False,
            "same_root_within_pair": True,
            "same_first_action_within_pair": True,
            "all_steps_schema_valid": True,
            "future_and_final_outcomes_unread": True,
        },
        "pairs": pairs,
        "sequences": sequences,
    }
    audit = {
        "pairs": len(pairs),
        "sequences": len(sequences),
        "steps": sum(len(row["steps"]) for row in sequences),
        "tasks": len({row["task_id"] for row in pairs}),
        "suite_difficulty_cells": len({(row["suite"], row["difficulty"]) for row in pairs}),
        "suite_pairs": dict(sorted(Counter(row["suite"] for row in pairs).items())),
        "all_same_root": all(row["same_root"] for row in pairs),
        "all_same_first_action": all(row["same_first_action"] for row in pairs),
        "selection_uses_v19_outcomes": False,
        "expected_prefix_replay_calls_two_replicas": sum(
            int(row["prefix_index"]) * 2 for row in sequences
        ),
    }
    return canonical_json_value(manifest), canonical_json_value(audit)


def _resolve_path(value: Any, path: Sequence[str]) -> Any:
    current = _plain(value)
    for key in path:
        current = current[str(key)]
    return current


def _resolve_arguments(step: Mapping[str, Any], prior_outputs: Sequence[Any]) -> dict[str, Any]:
    arguments = copy.deepcopy(dict(step["arguments"]))
    for field, source in step.get("dynamic_arguments", {}).items():
        arguments[str(field)] = _resolve_path(
            prior_outputs[int(source["step_index"])], source["path"]
        )
    return canonical_json_value(arguments)


def _readback_matches(output: Any, expectation: Mapping[str, Any]) -> bool:
    payload = _plain(output)
    kind = expectation["kind"]
    if kind == "banking_subject":
        return any(
            int(row["id"]) == int(expectation["id"])
            and row.get("subject") == expectation["value"]
            for row in payload
        )
    if kind == "slack_membership":
        present = str(expectation["user"]) in [str(item) for item in payload]
        return present is bool(expectation["present"])
    if kind == "calendar_start":
        return any(
            row.get("title") == expectation["title"]
            and row.get("start_time") == expectation["value"]
            for row in payload
        )
    if kind == "file_permission":
        return (
            payload.get("shared_with", {}).get(expectation["email"])
            == expectation["value"]
        )
    raise KeyError(f"unknown readback expectation: {kind}")


def execute_sequence(
    sequence: Mapping[str, Any],
    *,
    raw_episode: Mapping[str, Any],
    semantic_episode: Mapping[str, Any],
    suite: Any,
    registry: AdapterRegistry,
    replica_index: int,
    logical_clock_iso: str,
) -> dict[str, Any]:
    runtime, environment, ledger, current_state, replay = replay_to_prefix(
        raw_episode,
        semantic_episode,
        suite=suite,
        registry=registry,
        prefix_index=int(sequence["prefix_index"]),
        logical_clock_iso=logical_clock_iso,
    )
    if not replay["passed"]:
        raise ValueError(f"prefix replay mismatch: {replay['mismatches']}")
    if stable_fingerprint(current_state.model_dump(mode="json")) != sequence["state_ref"]:
        raise ValueError("sequence root state mismatch")
    features = copy.deepcopy(
        raw_episode["prefixes"][int(sequence["prefix_index"])]["features"]
    )
    prior_outputs = []
    step_rows = []
    leakage = []
    for offset, step in enumerate(sequence["steps"]):
        tool_name = str(step["tool_name"])
        function = runtime.functions[tool_name]
        arguments = _resolve_arguments(step, prior_outputs)
        arguments = canonical_json_value(
            function.parameters.model_validate(arguments).model_dump(mode="json")
        )
        with frozen_sandbox_clock(logical_clock_iso):
            transition, runtime_output = instrument_function_call(
                runtime,
                environment,
                event_index=int(sequence["prefix_index"]) + offset,
                function=tool_name,
                arguments=arguments,
            )
        formatted_output = _tool_result_to_str(runtime_output)
        ledger = _ledger_update(
            ledger,
            registry,
            episode_id=f"v19::{sequence['sequence_ref']}::replica{replica_index}",
            call_index=int(sequence["prefix_index"]) + offset,
            tool_name=tool_name,
            arguments=arguments,
            transition=transition,
            runtime_output=runtime_output,
            formatted_output=formatted_output,
        )
        features = _next_features(
            features,
            tool_name=tool_name,
            arguments=arguments,
            transition=transition,
            formatted_output=formatted_output,
            ledger=ledger,
        )
        semantic_payload = semantic_state_v3_payload(features)
        findings = find_semantic_state_v3_leakage(semantic_payload)
        leakage.extend(findings)
        step_rows.append(
            {
                "step_index": offset,
                "role": step["role"],
                "candidate_action": {
                    "tool_id": f"{sequence['suite']}::{tool_name}",
                    "arguments": arguments,
                },
                "execution_status": transition.tool_execution_status,
                "next_state_ref": stable_fingerprint(semantic_payload),
                "next_semantic_state": semantic_payload,
                "observation_fingerprint": stable_fingerprint(formatted_output),
                "simulator_audit_only": {
                    "model_visible": False,
                    "state_changed": transition.state_changed,
                    "state_delta_roots": _delta_roots(transition.canonical_state_delta),
                    "tool_error_type": transition.tool_error_type,
                },
            }
        )
        prior_outputs.append(runtime_output)
        current_state = StructuredSemanticStateV3.model_validate(semantic_payload)
    return {
        "schema_version": PERSISTENCE_CONFLICT_SCHEMA_VERSION,
        "sequence_ref": sequence["sequence_ref"],
        "pair_ref": sequence["pair_ref"],
        "branch": sequence["branch"],
        "task_id": sequence["task_id"],
        "suite": sequence["suite"],
        "difficulty": sequence["difficulty"],
        "replica_index": replica_index,
        "prefix_replay": replay,
        "model_visible": {
            "root_state_ref": sequence["state_ref"],
            "steps": step_rows,
        },
        "audit_only": {
            "model_visible": False,
            "readback_match": _readback_matches(
                prior_outputs[-1], sequence["readback_expectation_audit_only"]
            ),
            "leakage": sorted(set(leakage)),
        },
    }


def _replica_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(row))
    payload.pop("replica_index", None)
    return canonical_json_value(payload)


def execute_persistence_conflict_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    suites: Mapping[str, Any],
    registry: AdapterRegistry,
    replicas: int,
    logical_clock_iso: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if replicas != 2:
        raise ValueError("v19 requires exactly two replicas")
    raw_by_episode = {str(row["episode_id"]): row for row in raw_dataset["episodes"]}
    semantic_by_episode = {
        str(row["episode_id"]): row for row in semantic_dataset["episodes"]
    }
    required_tools = {
        str(step["tool_name"])
        for sequence in manifest["sequences"]
        for step in sequence["steps"]
    }
    missing = sorted(required_tools - set(registry.adapters))
    if missing:
        raise KeyError(f"missing v19 adapters: {missing}")
    all_replicas = []
    canonical = []
    replica_checks = []
    failures = []
    for sequence in manifest["sequences"]:
        pair = []
        try:
            for replica_index in range(replicas):
                pair.append(
                    execute_sequence(
                        sequence,
                        raw_episode=raw_by_episode[str(sequence["episode_id"])],
                        semantic_episode=semantic_by_episode[str(sequence["episode_id"])],
                        suite=suites[str(sequence["suite"])],
                        registry=registry,
                        replica_index=replica_index,
                        logical_clock_iso=logical_clock_iso,
                    )
                )
        except Exception as error:
            failures.append(
                {
                    "sequence_ref": sequence["sequence_ref"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue
        hashes = [stable_fingerprint(_replica_payload(row)) for row in pair]
        replica_checks.append(
            {"sequence_ref": sequence["sequence_ref"], "hashes": hashes, "identical": len(set(hashes)) == 1}
        )
        all_replicas.extend(pair)
        row = copy.deepcopy(pair[0])
        row.pop("replica_index", None)
        canonical.append(canonical_json_value(row))
    canonical.sort(key=lambda row: row["sequence_ref"])
    by_ref = {row["sequence_ref"]: row for row in canonical}
    pair_rows = []
    for pair in manifest["pairs"]:
        control = by_ref.get(pair["control_sequence_ref"])
        conflict = by_ref.get(pair["conflict_sequence_ref"])
        if control is None or conflict is None:
            continue
        control_steps = control["model_visible"]["steps"]
        conflict_steps = conflict["model_visible"]["steps"]
        pair_rows.append(
            {
                "pair_ref": pair["pair_ref"],
                "task_id": pair["task_id"],
                "suite": pair["suite"],
                "control_readback_match": control["audit_only"]["readback_match"],
                "conflict_readback_match": conflict["audit_only"]["readback_match"],
                "shared_write_state_identical": control_steps[0]["next_state_ref"]
                == conflict_steps[0]["next_state_ref"],
                "final_semantic_state_differs": control_steps[-1]["next_state_ref"]
                != conflict_steps[-1]["next_state_ref"],
                "final_observation_differs": control_steps[-1]["observation_fingerprint"]
                != conflict_steps[-1]["observation_fingerprint"],
            }
        )
    step_rows = [step for row in canonical for step in row["model_visible"]["steps"]]
    prefix_calls = sum(
        row["prefix_replay"]["prior_observed_calls_replayed"] for row in all_replicas
    )
    audit = {
        "complete_sequences": len(canonical),
        "complete_pairs": len(pair_rows),
        "step_executions": len(all_replicas) * 3,
        "prefix_replay_calls": prefix_calls,
        "total_sandbox_calls": len(all_replicas) * 3 + prefix_calls,
        "runtime_failures": failures,
        "all_steps_success": all(step["execution_status"] == "success" for step in step_rows),
        "replicas_identical": len(replica_checks) == len(manifest["sequences"])
        and all(row["identical"] for row in replica_checks),
        "zero_prefix_replay_mismatches": all(row["prefix_replay"]["passed"] for row in all_replicas),
        "zero_semantic_leakage": all(not row["audit_only"]["leakage"] for row in canonical),
        "control_persistence_matches": sum(row["control_readback_match"] for row in pair_rows),
        "conflict_readback_matches": sum(row["conflict_readback_match"] for row in pair_rows),
        "shared_write_states_identical": sum(row["shared_write_state_identical"] for row in pair_rows),
        "final_semantic_states_differ": sum(row["final_semantic_state_differs"] for row in pair_rows),
        "final_observations_differ": sum(row["final_observation_differs"] for row in pair_rows),
        "suites_with_both_matches": len(
            {
                row["suite"]
                for row in pair_rows
                if row["control_readback_match"] and row["conflict_readback_match"]
            }
        ),
        "pair_rows": pair_rows,
        "required_tools": sorted(required_tools),
    }
    dataset = {
        "schema_version": PERSISTENCE_CONFLICT_SCHEMA_VERSION,
        "scope": "clean-only paired multi-step persistence and competing-update dataset",
        "loader_contract": {
            "audit_only_is_never_model_visible": True,
            "task_and_reference_ids_are_grouping_only": True,
            "all_executed_actions_are_explicit_inputs": True,
            "no_future_or_final_outcome_labels": True,
        },
        "sequences": canonical,
        "replica_verification": replica_checks,
    }
    return canonical_json_value(dataset), canonical_json_value(audit)
