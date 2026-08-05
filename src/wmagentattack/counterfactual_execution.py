"""Exact clean-sandbox execution of a frozen counterfactual manifest."""

from __future__ import annotations

import ast
import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from pydantic import BaseModel

from .clean_state_instrumentation import instrument_function_call
from .counterfactual_evidence import build_query_universe
from .decision_state import canonical_json_value, stable_fingerprint
from .semantic_state_v3 import (
    StructuredSemanticStateV3,
    find_semantic_state_v3_leakage,
    semantic_state_v3_payload,
)
from .structured_ledger_v2 import (
    AdapterMode,
    AdapterRegistry,
    AdapterSpec,
    ExecutionChannelStatus,
    StructuredEvidenceLedgerV2,
    update_structured_ledger,
)


COUNTERFACTUAL_OUTCOME_SCHEMA_VERSION = (
    "wmagentattack.counterfactual_evidence_outcomes.v1"
)


def apply_label_blind_adapter_repair(
    registry: AdapterRegistry, repair: Mapping[str, Any]
) -> AdapterRegistry:
    """Apply a frozen, outcome-blind adapter-only repair without shadowing."""

    if repair.get("outcome_labels_present") is not False:
        raise ValueError("counterfactual adapter repair must remain outcome-label blind")
    additions = {
        str(name): AdapterSpec.model_validate(spec)
        for name, spec in repair.get("additional_adapters", {}).items()
    }
    overlap = sorted(set(registry.adapters) & set(additions))
    if overlap:
        raise ValueError(f"counterfactual adapter repair shadows tools: {overlap}")
    return AdapterRegistry(
        schema_version=str(repair["schema_version"]),
        benchmark_version=str(repair["benchmark_version"]),
        suite="counterfactual_clean_multi_suite",
        adapters={**registry.adapters, **additions},
    )


def _tool_result_to_str(tool_result: Any) -> str:
    """Mirror AgentDojo v1.2.2's default YAML tool-output formatter."""

    if isinstance(tool_result, BaseModel):
        return yaml.safe_dump(tool_result.model_dump()).strip()
    if isinstance(tool_result, list):
        rows = []
        for item in tool_result:
            if type(item) in (str, int):
                rows.append(str(item))
            elif isinstance(item, BaseModel):
                rows.append(item.model_dump())
            else:
                raise TypeError(f"unsupported AgentDojo tool result item: {type(item)}")
        return yaml.safe_dump(rows).strip()
    return str(tool_result)


def _without_runtime_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_ids(item)
            for key, item in value.items()
            if key
            not in {
                "record_id",
                "fact_id",
                "conflict_id",
                "episode_id",
                "arguments_fingerprint",
                "observation_fingerprint",
                "resolved_entity_id",
                "provisional_entity_id",
                "entity_id",
                "left_fact_id",
                "right_fact_id",
            }
        }
    if isinstance(value, list):
        return [_without_runtime_ids(item) for item in value]
    return value


def _ledger_feature_payload(ledger: StructuredEvidenceLedgerV2) -> dict[str, Any]:
    records = []
    for record in sorted(ledger.records, key=lambda row: (row.call_index, row.record_index)):
        records.append(
            {
                "entity_type": record.entity_type,
                "entity_key": canonical_json_value(record.entity_key),
                "entity_candidates": [
                    canonical_json_value(candidate.entity_key)
                    for candidate in record.entity_candidates
                ],
                "link_status": record.link_status,
                "attributes": [
                    {
                        "name": attribute.name,
                        "value": canonical_json_value(attribute.value),
                        "kind": attribute.kind.value,
                    }
                    for attribute in record.attributes
                ],
                "context": canonical_json_value(record.context),
                "source_tool": record.source_tool,
                "source_arguments": canonical_json_value(record.source_arguments),
                "call_index": record.call_index,
                "execution_status": record.execution_status,
                "state_provenance": record.state_provenance,
            }
        )
    payload = {
        "records": records,
        "conflicts": [
            {"attribute_name": row.attribute_name, "reason": row.reason}
            for row in ledger.conflicts
        ],
        "execution_receipts": [
            {
                "call_index": row.call_index,
                "tool_name": row.tool_name,
                "execution_status": row.execution_status,
            }
            for row in ledger.execution_receipts
        ],
    }
    return _without_runtime_ids(payload)


def _parse_tool_output(text: str, mode: AdapterMode) -> Any:
    if mode in {AdapterMode.NAME_LIST_TEXT, AdapterMode.FLIGHT_LINES}:
        return text
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = None
    if mode in {AdapterMode.VALUE, AdapterMode.MUTATION_ACK}:
        return text if parsed is None else parsed
    return parsed


def _delta_roots(delta: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    roots = Counter()
    for operation in delta:
        path = str(operation.get("path", ""))
        root = "/" + path.lstrip("/").split("/", 1)[0] if path else "<root>"
        roots[root] += 1
    return dict(sorted(roots.items()))


def _constraint_ref(goal: str, term: str) -> str:
    return stable_fingerprint(
        {"kind": "goal_fact_term", "trusted_goal": goal, "term": term}
    )


def _candidate_tool(candidate_id: str, suite: str) -> str:
    if candidate_id == "STOP":
        raise ValueError("STOP is not an executable environment tool")
    prefix, tool_name = candidate_id.split("::", 1)
    if prefix != suite:
        raise ValueError("candidate suite prefix mismatch")
    return tool_name


def adapter_coverage_for_manifest(
    raw_dataset: Mapping[str, Any], manifest: Mapping[str, Any], registry: AdapterRegistry
) -> dict[str, Any]:
    """Audit every candidate and observed replay tool before any execution."""

    raw_by_episode = {
        str(row["episode_id"]): row for row in raw_dataset["episodes"]
    }
    candidate_tools: set[str] = set()
    replay_tools: set[str] = set()
    for row in manifest["rows"]:
        query = row["query"]
        metadata = query["metadata"]
        suite = str(metadata["suite"])
        candidate_tools.add(_candidate_tool(str(query["candidate_id"]), suite))
        episode = raw_by_episode[str(metadata["episode_id"])]
        for prefix_index in range(int(metadata["prefix_index"])):
            replay_tools.add(
                _candidate_tool(
                    str(episode["prefixes"][prefix_index]["targets"]["next_action"]),
                    suite,
                )
            )
    required_tools = candidate_tools | replay_tools
    missing = sorted(required_tools - set(registry.adapters))
    return {
        "candidate_tools": sorted(candidate_tools),
        "replay_tools": sorted(replay_tools),
        "required_tools": sorted(required_tools),
        "missing_tools": missing,
        "complete": not missing,
    }


def _ledger_update(
    ledger: StructuredEvidenceLedgerV2,
    registry: AdapterRegistry,
    *,
    episode_id: str,
    call_index: int,
    tool_name: str,
    arguments: Mapping[str, Any],
    transition: Any,
    runtime_output: Any,
    formatted_output: str,
) -> StructuredEvidenceLedgerV2:
    success = transition.tool_execution_status == "success"
    adapted = (
        _parse_tool_output(formatted_output, registry.adapters[tool_name].mode)
        if success
        else runtime_output
    )
    return update_structured_ledger(
        ledger,
        registry,
        episode_id=episode_id,
        call_index=call_index,
        channel_status=(
            ExecutionChannelStatus.EXECUTED_SUCCESS
            if success
            else ExecutionChannelStatus.EXECUTED_ERROR
        ),
        tool_name=tool_name,
        arguments=arguments,
        runtime_output=adapted,
        error_type=transition.tool_error_type,
        state_changed=transition.state_changed,
    ).ledger


def _semantic_prefix_matches(
    raw_prefix: Mapping[str, Any], semantic_prefix: Mapping[str, Any]
) -> bool:
    return semantic_state_v3_payload(raw_prefix["features"]) == semantic_prefix[
        "features"
    ]["semantic_state_v3"]


def replay_to_prefix(
    raw_episode: Mapping[str, Any],
    semantic_episode: Mapping[str, Any],
    *,
    suite: Any,
    registry: AdapterRegistry,
    prefix_index: int,
) -> tuple[
    Any,
    Any,
    StructuredEvidenceLedgerV2,
    StructuredSemanticStateV3,
    dict[str, Any],
]:
    """Reconstruct a prefix from a new default environment and verify each step."""

    from agentdojo.functions_runtime import FunctionsRuntime

    raw_prefixes = raw_episode["prefixes"]
    semantic_prefixes = semantic_episode["prefixes"]
    if prefix_index < 0 or prefix_index >= len(raw_prefixes):
        raise ValueError("manifest prefix index is outside the episode")
    runtime = FunctionsRuntime(suite.tools)
    environment = suite.load_and_inject_default_environment({})
    ledger = StructuredEvidenceLedgerV2()
    mismatches = []
    if _ledger_feature_payload(ledger) != raw_prefixes[0]["features"]["ledger_v2"]:
        mismatches.append("initial_ledger")
    if not _semantic_prefix_matches(raw_prefixes[0], semantic_prefixes[0]):
        mismatches.append("initial_semantic")

    for index in range(prefix_index):
        current = raw_prefixes[index]
        following = raw_prefixes[index + 1]
        candidate_id = str(current["targets"]["next_action"])
        tool_name = _candidate_tool(candidate_id, str(raw_episode["suite"]))
        arguments = canonical_json_value(
            following["features"]["last_action"].get("arguments", {})
        )
        transition, runtime_output = instrument_function_call(
            runtime,
            environment,
            event_index=index,
            function=tool_name,
            arguments=arguments,
        )
        formatted_output = _tool_result_to_str(runtime_output)
        if following["features"]["last_action"].get("function") != tool_name:
            mismatches.append(f"call_{index}_action")
        if formatted_output != following["features"]["last_observation"]:
            mismatches.append(f"call_{index}_observation")
        if (
            transition.tool_execution_status
            != following["features"]["execution_receipt"]["status"]
        ):
            mismatches.append(f"call_{index}_status")
        ledger = _ledger_update(
            ledger,
            registry,
            episode_id=str(raw_episode["episode_id"]),
            call_index=index,
            tool_name=tool_name,
            arguments=arguments,
            transition=transition,
            runtime_output=runtime_output,
            formatted_output=formatted_output,
        )
        if _ledger_feature_payload(ledger) != following["features"]["ledger_v2"]:
            mismatches.append(f"call_{index}_ledger")
        if not _semantic_prefix_matches(following, semantic_prefixes[index + 1]):
            mismatches.append(f"call_{index}_semantic")

    current_state = StructuredSemanticStateV3.model_validate(
        semantic_prefixes[prefix_index]["features"]["semantic_state_v3"]
    )
    return runtime, environment, ledger, current_state, {
        "prior_observed_calls_replayed": prefix_index,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _next_features(
    current_features: Mapping[str, Any],
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    transition: Any,
    formatted_output: str,
    ledger: StructuredEvidenceLedgerV2,
) -> dict[str, Any]:
    features = copy.deepcopy(dict(current_features))
    summary = copy.deepcopy(dict(features.get("causal_state_summary", {})))
    roots = Counter(summary.get("delta_roots", {}))
    roots.update(_delta_roots(transition.canonical_state_delta))
    summary.update(
        {
            "last_state_changed": bool(transition.state_changed),
            "cumulative_state_changes": int(
                summary.get("cumulative_state_changes", 0)
            )
            + int(transition.state_changed),
            "cumulative_errors": int(summary.get("cumulative_errors", 0))
            + int(transition.tool_execution_status == "error"),
            "last_delta_count": len(transition.canonical_state_delta),
            "delta_roots": dict(sorted(roots.items())),
        }
    )
    features.update(
        {
            "prefix_index": int(features["prefix_index"]) + 1,
            "last_action": {
                "function": tool_name,
                "arguments": canonical_json_value(dict(arguments)),
            },
            "last_observation": formatted_output,
            "execution_receipt": {
                "status": transition.tool_execution_status,
                "error_type": transition.tool_error_type,
                "output_type": transition.tool_output_type,
            },
            "causal_state_summary": summary,
            "ledger_v2": _ledger_feature_payload(ledger),
        }
    )
    return features


def _constraint_progress(
    current: StructuredSemanticStateV3,
    following: StructuredSemanticStateV3,
) -> list[dict[str, Any]]:
    current_matched = set(current.goal_evidence.matched_fact_terms)
    next_matched = set(following.goal_evidence.matched_fact_terms)
    rows = []
    for term in current.goal.fact_terms:
        if term in current_matched:
            progress = "ALREADY_SUPPORTED"
            prior_status = "SUPPORTED"
            role = "STATE_CONSISTENCY_ONLY"
        elif term in next_matched:
            progress = "NEWLY_SUPPORTED"
            prior_status = "UNSUPPORTED"
            role = "PREDICTIVE"
        else:
            progress = "UNCHANGED_UNSUPPORTED"
            prior_status = "UNSUPPORTED"
            role = "PREDICTIVE"
        rows.append(
            {
                "constraint_ref": _constraint_ref(
                    current.goal.normalized_goal, term
                ),
                "term": term,
                "progress": progress,
                "prior_status": prior_status,
                "training_role": role,
                "label_source": "fresh_clean_sandbox_counterfactual",
            }
        )
    return sorted(rows, key=lambda row: row["constraint_ref"])


def execute_manifest_row(
    row: Mapping[str, Any],
    *,
    raw_episode: Mapping[str, Any],
    semantic_episode: Mapping[str, Any],
    suite: Any,
    registry: AdapterRegistry,
    replica_index: int,
) -> dict[str, Any]:
    query = row["query"]
    metadata = query["metadata"]
    runtime, environment, ledger, current_state, replay = replay_to_prefix(
        raw_episode,
        semantic_episode,
        suite=suite,
        registry=registry,
        prefix_index=int(metadata["prefix_index"]),
    )
    if not replay["passed"]:
        raise ValueError(f"prefix replay mismatch: {replay['mismatches']}")
    current_payload = current_state.model_dump(mode="json")
    if stable_fingerprint(current_payload) != query["state_ref"]:
        raise ValueError("manifest current-state reference mismatch")
    tool_name = _candidate_tool(str(query["candidate_id"]), str(metadata["suite"]))
    if query["candidate_id"] == query["current_victim_decision"]:
        raise ValueError("counterfactual candidate equals observed victim decision")
    function = runtime.functions[tool_name]
    arguments = canonical_json_value(
        function.parameters.model_validate(row["binding"]["arguments"]).model_dump(
            mode="json"
        )
    )
    transition, runtime_output = instrument_function_call(
        runtime,
        environment,
        event_index=int(metadata["prefix_index"]),
        function=tool_name,
        arguments=arguments,
    )
    formatted_output = _tool_result_to_str(runtime_output)
    ledger = _ledger_update(
        ledger,
        registry,
        episode_id=f"counterfactual::{row['manifest_row_ref']}::replica{replica_index}",
        call_index=int(metadata["prefix_index"]),
        tool_name=tool_name,
        arguments=arguments,
        transition=transition,
        runtime_output=runtime_output,
        formatted_output=formatted_output,
    )
    features = _next_features(
        raw_episode["prefixes"][int(metadata["prefix_index"])]["features"],
        tool_name=tool_name,
        arguments=arguments,
        transition=transition,
        formatted_output=formatted_output,
        ledger=ledger,
    )
    next_payload = semantic_state_v3_payload(features)
    following = StructuredSemanticStateV3.model_validate(next_payload)
    current_unresolved = (
        current_state.goal_evidence.ambiguous_entity_records
        + current_state.goal_evidence.unlinked_entity_records
    )
    next_unresolved = (
        following.goal_evidence.ambiguous_entity_records
        + following.goal_evidence.unlinked_entity_records
    )
    progress = _constraint_progress(current_state, following)
    if {item["constraint_ref"] for item in progress} != set(
        query["constraint_refs"]
    ):
        raise ValueError("manifest constraint references changed")
    link_resolution = Counter(
        link.resolution for link in transition.argument_entity_links
    )
    model_visible = {
        "current_state_ref": str(query["state_ref"]),
        "next_state_ref": stable_fingerprint(next_payload),
        "candidate_action": {
            "tool_id": str(query["candidate_id"]),
            "arguments": arguments,
            "argument_binding": "OBSERVED_COUNTERFACTUAL",
        },
        "next_semantic_state": next_payload,
        "constraint_progress": progress,
        "events": {
            "execution_status": transition.tool_execution_status,
            "record_count_delta": len(following.evidence_records)
            - len(current_state.evidence_records),
            "conflict_count_delta": len(following.conflicts)
            - len(current_state.conflicts),
            "unresolved_entity_count_delta": next_unresolved - current_unresolved,
            "newly_matched_goal_terms": sorted(
                set(following.goal_evidence.matched_fact_terms)
                - set(current_state.goal_evidence.matched_fact_terms)
            ),
        },
    }
    return {
        "schema_version": COUNTERFACTUAL_OUTCOME_SCHEMA_VERSION,
        "manifest_row_ref": str(row["manifest_row_ref"]),
        "query_ref": str(query["query_ref"]),
        "replica_index": replica_index,
        "metadata": canonical_json_value(metadata),
        "binding_provenance": canonical_json_value(row["binding"]),
        "prefix_replay": replay,
        "model_visible": model_visible,
        "simulator_audit_only": {
            "model_visible": False,
            "state_before_fingerprint": transition.state_before_fingerprint,
            "state_after_fingerprint": transition.state_after_fingerprint,
            "state_changed": transition.state_changed,
            "state_delta_operation_count": len(transition.canonical_state_delta),
            "state_delta_roots": _delta_roots(transition.canonical_state_delta),
            "tool_error_type": transition.tool_error_type,
            "tool_output_type": transition.tool_output_type,
            "tool_output_fingerprint": stable_fingerprint(runtime_output),
            "formatted_output_fingerprint": stable_fingerprint(formatted_output),
            "formatted_output_characters": len(formatted_output),
            "argument_entity_link_resolution": {
                key: int(link_resolution.get(key, 0))
                for key in ("no_match", "unique", "ambiguous")
            },
        },
    }


def replica_comparison_payload(outcome: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(outcome))
    payload.pop("replica_index", None)
    return canonical_json_value(payload)


def execute_frozen_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    suites: Mapping[str, Any],
    registry: AdapterRegistry,
    tool_specs: Mapping[str, Any],
    selected_task_ids: Sequence[str],
    replicas: int,
    readiness_gate: Mapping[str, float | int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if replicas != 2:
        raise ValueError("the frozen pilot requires exactly two replicas")
    adapter_coverage = adapter_coverage_for_manifest(raw_dataset, manifest, registry)
    if not adapter_coverage["complete"]:
        raise KeyError(
            "missing counterfactual ledger adapters: "
            + ", ".join(adapter_coverage["missing_tools"])
        )
    raw_by_episode = {str(row["episode_id"]): row for row in raw_dataset["episodes"]}
    semantic_by_episode = {
        str(row["episode_id"]): row for row in semantic_dataset["episodes"]
    }
    universe = build_query_universe(
        raw_dataset,
        semantic_dataset,
        selected_task_ids=selected_task_ids,
        tool_specs=tool_specs,
    )
    all_replicas = []
    canonical_outcomes = []
    replica_verification = []
    infrastructure_failures = []
    for row in manifest["rows"]:
        episode_id = str(row["query"]["metadata"]["episode_id"])
        pair = []
        try:
            for replica_index in range(replicas):
                pair.append(
                    execute_manifest_row(
                        row,
                        raw_episode=raw_by_episode[episode_id],
                        semantic_episode=semantic_by_episode[episode_id],
                        suite=suites[row["query"]["metadata"]["suite"]],
                        registry=registry,
                        replica_index=replica_index,
                    )
                )
        except Exception as error:
            infrastructure_failures.append(
                {
                    "manifest_row_ref": row["manifest_row_ref"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            continue
        hashes = [stable_fingerprint(replica_comparison_payload(item)) for item in pair]
        replica_verification.append(
            {
                "manifest_row_ref": row["manifest_row_ref"],
                "replica_hashes": hashes,
                "identical": len(set(hashes)) == 1,
            }
        )
        all_replicas.extend(pair)
        canonical = copy.deepcopy(pair[0])
        canonical.pop("replica_index", None)
        canonical_outcomes.append(canonical_json_value(canonical))

    canonical_outcomes.sort(key=lambda row: row["manifest_row_ref"])
    current_states = universe["state_catalog"]
    next_states = {
        stable_fingerprint(row["model_visible"]["next_semantic_state"]): row[
            "model_visible"
        ]["next_semantic_state"]
        for row in canonical_outcomes
    }
    leakage = {}
    for state_ref, state in {**current_states, **next_states}.items():
        findings = find_semantic_state_v3_leakage(state)
        if findings:
            leakage[state_ref] = list(findings)
    progress = Counter(
        item["progress"]
        for row in canonical_outcomes
        for item in row["model_visible"]["constraint_progress"]
    )
    execution_errors = sum(
        row["model_visible"]["events"]["execution_status"] == "error"
        for row in canonical_outcomes
    )
    conflicts = sum(
        row["model_visible"]["events"]["conflict_count_delta"] > 0
        for row in canonical_outcomes
    )
    ambiguity = sum(
        row["model_visible"]["events"]["unresolved_entity_count_delta"] > 0
        for row in canonical_outcomes
    )
    selected_relations = sum(
        len(row["model_visible"]["constraint_progress"])
        for row in canonical_outcomes
    )
    total_relations = (
        int(universe["audit"]["observed_constraint_rows"])
        + int(universe["audit"]["candidate_constraint_queries"])
    )
    observed_relations = int(universe["audit"]["observed_constraint_rows"]) + selected_relations
    observed_fraction = observed_relations / max(1, total_relations)
    collector_checks = {
        "exact_24_bound_queries": len(manifest["rows"]) == 24,
        "exact_48_fresh_state_executions": len(all_replicas) == 48,
        "all_12_suite_difficulty_cells": len(
            {
                (
                    row["query"]["metadata"]["suite"],
                    row["query"]["metadata"]["difficulty"],
                )
                for row in manifest["rows"]
            }
        )
        == 12,
        "exact_12_read_only_and_12_mutating": Counter(
            row["query"]["mutation_class"] for row in manifest["rows"]
        )
        == {"read_only": 12, "mutating": 12},
        "zero_stop_tool_executions": all(
            row["query"]["candidate_id"] != "STOP" for row in manifest["rows"]
        ),
        "zero_prefix_replay_mismatches": all(
            item["prefix_replay"]["passed"] for item in all_replicas
        ),
        "zero_infrastructure_failures": not infrastructure_failures,
        "all_argument_payloads_schema_valid": len(canonical_outcomes) == 24,
        "all_replica_pairs_identical": len(replica_verification) == 24
        and all(row["identical"] for row in replica_verification),
        "zero_semantic_state_leakage": not leakage,
    }
    readiness_checks = {
        "minimum_observed_bound_query_fraction": observed_fraction
        >= float(readiness_gate["minimum_observed_bound_query_fraction"]),
        "minimum_counterfactual_execution_errors": execution_errors
        >= int(readiness_gate["minimum_counterfactual_execution_errors"]),
        "minimum_counterfactual_conflicts": conflicts
        >= int(readiness_gate["minimum_counterfactual_conflicts"]),
        "minimum_counterfactual_ambiguity_events": ambiguity
        >= int(readiness_gate["minimum_counterfactual_ambiguity_events"]),
    }
    collector_pass = all(collector_checks.values())
    training_ready = all(readiness_checks.values())
    if collector_pass and training_ready:
        decision = "GO_COLLECTOR__GO_TINY_RELATIONAL_MODEL_PROBE"
    elif collector_pass:
        decision = "GO_COLLECTOR__NO_GO_TRAINING__TARGETED_DATA_ROUND_REQUIRED"
    else:
        decision = "NO_GO__COUNTERFACTUAL_COLLECTION_HARNESS_FAILED"
    dataset = {
        "schema_version": COUNTERFACTUAL_OUTCOME_SCHEMA_VERSION,
        "scope": "clean-only AgentDojo synthetic sandbox counterfactual delta dataset",
        "loader_contract": {
            "simulator_audit_only_is_never_model_visible": True,
            "reference_and_metadata_ids_are_grouping_only": True,
            "counterfactual_arguments_are_explicit_model_inputs": True,
            "unexecuted_actions_remain_unlabeled": True,
        },
        "universe_audit": universe["audit"],
        "current_state_catalog": current_states,
        "next_state_catalog": dict(sorted(next_states.items())),
        "constraint_catalog": universe["constraint_catalog"],
        "counterfactual_outcomes": canonical_outcomes,
        "replica_verification": replica_verification,
        "remaining_unlabeled_action_queries": int(
            universe["audit"]["executable_action_queries"]
        )
        - len(canonical_outcomes),
    }
    audit = {
        "adapter_coverage": adapter_coverage,
        "manifest_rows": len(manifest["rows"]),
        "counterfactual_executions": len(all_replicas),
        "prior_observed_replay_executions": sum(
            item["prefix_replay"]["prior_observed_calls_replayed"]
            for item in all_replicas
        ),
        "total_sandbox_tool_executions": len(all_replicas)
        + sum(
            item["prefix_replay"]["prior_observed_calls_replayed"]
            for item in all_replicas
        ),
        "canonical_outcomes": len(canonical_outcomes),
        "counterfactual_constraint_rows": selected_relations,
        "progress_rows": dict(sorted(progress.items())),
        "execution_error_outcomes": execution_errors,
        "conflict_positive_outcomes": conflicts,
        "ambiguity_positive_outcomes": ambiguity,
        "state_changed_outcomes": sum(
            row["simulator_audit_only"]["state_changed"]
            for row in canonical_outcomes
        ),
        "baseline_observed_constraint_rows": int(
            universe["audit"]["observed_constraint_rows"]
        ),
        "total_bound_query_relation_space": total_relations,
        "observed_bound_query_relations_after_pilot": observed_relations,
        "observed_bound_query_fraction_after_pilot": observed_fraction,
        "infrastructure_failures": infrastructure_failures,
        "semantic_state_leakage": leakage,
        "collector_checks": collector_checks,
        "training_readiness_checks": readiness_checks,
        "collector_pass": collector_pass,
        "training_ready": training_ready,
        "decision": decision,
    }
    return dataset, audit
