"""Label-blind selection and frozen gate helpers for the tau3 tail pilot."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

from wmagentattack.tau3_multistep import stable_hash


def effective_tail_protocol(
    protocol: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay only the preregistered 20/20/80 horizon onto the frozen base."""

    result = copy.deepcopy(dict(base))
    mechanism = protocol["single_mutable_mechanism"]
    result["protocol_id"] = protocol["protocol_id"]
    result["status"] = "manifest_frozen_before_interactive_outcomes"
    result["role_contracts"]["agent"][
        "maximum_generation_calls_per_episode"
    ] = int(mechanism["agent_generation_calls_per_episode_candidate"])
    result["role_contracts"]["user"][
        "maximum_generation_calls_per_episode"
    ] = int(mechanism["user_generation_calls_per_episode_candidate"])
    result["interaction"]["maximum_orchestrator_steps"] = int(
        mechanism["orchestrator_steps_candidate"]
    )
    result["fixed_budget"] = copy.deepcopy(protocol["fixed_budget"])
    return result


def _hamilton_allocation(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if population < total:
        raise ValueError("tail panel population is smaller than its budget")
    ideals = {key: total * value / population for key, value in counts.items()}
    allocation = {key: min(value, math.floor(ideals[key])) for key, value in counts.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
    )
    while remaining:
        advanced = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            raise ValueError("tail panel allocation cannot be completed")
    return allocation


def select_tail_panel(
    parent: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select four out-of-pilot tasks per domain without reading outcomes."""

    panel = protocol["pilot_panel"]
    allowed_ids = set(map(str, parent["out_of_pilot_episode_ids"]))
    pool = [row for row in parent["rows"] if str(row["episode_id"]) in allowed_ids]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_task[str(row["task_key"])].append(row)
    expected_seeds = set(map(int, panel["seeds"]))
    for rows in by_task.values():
        if {int(row["llm_seed"]) for row in rows} != expected_seeds:
            raise ValueError("tail candidate task does not contain both frozen seeds")

    selected_tasks: set[str] = set()
    allocation_audit: dict[str, Any] = {}
    for domain in panel["domains"]:
        tasks = {
            key: rows
            for key, rows in by_task.items()
            if str(rows[0]["domain"]) == str(domain)
        }
        strata = Counter(str(rows[0]["structural_stratum"]) for rows in tasks.values())
        allocation = _hamilton_allocation(strata, int(panel["tasks_per_domain"]))
        selected_by_stratum: Counter[str] = Counter()
        for stratum, count in allocation.items():
            candidates = [
                key
                for key, rows in tasks.items()
                if str(rows[0]["structural_stratum"]) == stratum
            ]
            candidates.sort(
                key=lambda key: stable_hash(
                    {
                        "protocol_id": protocol["protocol_id"],
                        "domain": domain,
                        "task_key": key,
                    }
                )
            )
            selected_tasks.update(candidates[:count])
            selected_by_stratum[stratum] += count
        allocation_audit[str(domain)] = {
            "population_by_stratum": dict(sorted(strata.items())),
            "selected_by_stratum": dict(sorted(selected_by_stratum.items())),
        }

    rows = [copy.deepcopy(row) for row in pool if str(row["task_key"]) in selected_tasks]
    rows.sort(key=lambda row: (str(row["domain"]), str(row["task_key"]), int(row["llm_seed"])))
    manifest = {
        "schema_version": "wmagentattack.tau3_tail_horizon_manifest.v1",
        "protocol_id": protocol["protocol_id"],
        "source_commit": parent["source_commit"],
        "parent_manifest_sha256": protocol["binding_parent_result"]["manifest_sha256"],
        "selection_contract_sha256": stable_hash(panel),
        "real_external_endpoint_calls": 0,
        "rows": rows,
    }
    task_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task_splits[str(row["experimental_split"])].add(str(row["task_key"]))
    overlaps = {
        f"{left}::{right}": sorted(task_splits[left] & task_splits[right])
        for left in task_splits
        for right in task_splits
        if left < right
    }
    audit = {
        "passed": (
            len(rows) == int(panel["episodes"])
            and len(selected_tasks) == int(panel["tasks"])
            and all(not value for value in overlaps.values())
        ),
        "episodes": len(rows),
        "tasks": len(selected_tasks),
        "domain_episode_counts": dict(sorted(Counter(str(row["domain"]) for row in rows).items())),
        "allocation": allocation_audit,
        "split_task_counts": {key: len(value) for key, value in sorted(task_splits.items())},
        "split_task_overlaps": overlaps,
        "selection_inputs_used": list(panel["selection_inputs_allowed"]),
        "forbidden_outcome_inputs_read": [],
        "manifest_content_sha256": stable_hash(manifest),
    }
    return manifest, audit


def evaluate_tail_gate(
    metrics: Mapping[str, Any], integrity: Mapping[str, bool], gate: Mapping[str, Any]
) -> dict[str, bool]:
    """Evaluate every tail-pilot clause exactly once."""

    checks = {
        "episodes_complete": metrics["episodes_complete"] == gate["expected_complete_episodes"],
        "runtime_failures": metrics["runtime_failures"] == 0,
        "private_scenario_exposures": metrics["agent_private_scenario_exposures"] == 0,
        "real_external_endpoints": metrics["real_external_endpoint_calls"] == 0,
        "communication_errors": metrics["communication_error_terminations"] == 0,
        "forced_stops": metrics["forced_budget_stop_episodes"] <= gate["maximum_forced_budget_stop_episodes"],
        "relative_stop_reduction": metrics["relative_forced_stop_reduction_vs_parent"] >= gate["minimum_relative_forced_stop_reduction_vs_paired_parent"],
        "natural_user_messages": metrics["natural_user_messages"] >= gate["minimum_natural_user_messages"],
        "adjacent_transitions": metrics["adjacent_assistant_tool_transitions"] >= gate["minimum_adjacent_assistant_tool_transitions"],
        "multi_transition_episodes": metrics["episodes_with_two_or_more_assistant_transitions"] >= gate["minimum_episodes_with_two_or_more_assistant_transitions"],
        "tasks_with_transition": metrics["tasks_with_at_least_one_assistant_transition"] >= gate["minimum_tasks_with_at_least_one_assistant_transition"],
        "unique_assistant_tools": metrics["unique_executed_assistant_tools"] >= gate["minimum_unique_executed_assistant_tools"],
        "agent_tool_rate_floor": metrics["agent_tool_decision_rate"] >= gate["minimum_agent_tool_decision_rate"],
        "agent_tool_rate_ceiling": metrics["agent_tool_decision_rate"] <= gate["maximum_agent_tool_decision_rate"],
        "dominant_action": metrics["dominant_agent_action_fraction"] <= gate["maximum_dominant_agent_action_fraction"],
        "changed_transitions": metrics["state_changed_assistant_transitions"] >= gate["minimum_state_changed_assistant_transitions"],
        "unchanged_transitions": metrics["state_unchanged_assistant_transitions"] >= gate["minimum_state_unchanged_assistant_transitions"],
        "changed_tasks": metrics["tasks_with_state_changed_assistant_transition"] >= gate["minimum_tasks_with_state_changed_assistant_transition"],
        "changed_domains": metrics["domains_with_state_changed_assistant_transition"] >= gate["minimum_domains_with_state_changed_assistant_transition"],
        "paired_change_gain": metrics["paired_state_changed_transition_gain"] >= gate["minimum_paired_state_changed_transition_gain"],
        "supported_targets": metrics["supported_transition_targets"] >= gate["minimum_supported_transition_targets"],
        "tool_error_non_regression": metrics["assistant_tool_error_rate_increase_over_parent"] <= gate["maximum_assistant_tool_error_rate_increase_over_paired_parent"],
    }
    checks.update({f"integrity::{key}": bool(value) for key, value in integrity.items()})
    return checks
