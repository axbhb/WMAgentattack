"""Audit exact clean-state instrumentation across frozen AgentDojo tasks.

This script executes only each user task's built-in ground-truth calls in the
synthetic in-memory sandbox.  It does not load an LLM, construct attacks, contact
external endpoints, or create training examples.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import (
    build_ground_truth_goal_slots,
    candidate_tool_manifest,
    canonical_call_signature,
    instrument_function_call,
    match_completed_goal_slots,
)


SCOPE = "clean-only AgentDojo ground-truth state-interface audit"


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _canonical_json_roundtrip_ok(value: Any) -> bool:
    """Return whether a value is finite canonical JSON and round-trips exactly."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded) == value
    except (TypeError, ValueError):
        return False


def _utility(task, pre_environment, environment, traces) -> tuple[bool, str]:
    output = task.GROUND_TRUTH_OUTPUT
    trace_result = task.utility_from_traces(
        output, pre_environment, environment, traces
    )
    if trace_result is not None:
        return bool(trace_result), "utility_from_traces"
    return bool(task.utility(output, pre_environment, environment)), "utility"


def _resolution_counts(links) -> dict[str, int]:
    counts = Counter(link.resolution for link in links)
    return {
        name: int(counts.get(name, 0))
        for name in ("no_match", "unique", "ambiguous")
    }


def _audit_task(suite, task, functions_runtime_type) -> dict:
    environment = suite.load_and_inject_default_environment({})
    environment = task.init_environment(environment)
    pre_environment = environment.model_copy(deep=True)
    ground_truth = list(task.ground_truth(pre_environment.model_copy(deep=True)))
    goal_slots = build_ground_truth_goal_slots(ground_truth, pre_environment)
    runtime = functions_runtime_type(suite.tools)
    traces = []
    transitions = []
    signatures = []
    utilities = []
    utility_sources = []
    delta_roots = Counter()
    canonical_snapshot_failures = int(
        not _canonical_json_roundtrip_ok(pre_environment.model_dump(mode="json"))
    )
    for event_index, call in enumerate(ground_truth):
        signature = canonical_call_signature(call.function, call.args)
        signatures.append(signature)
        transition, _ = instrument_function_call(
            runtime,
            environment,
            event_index=event_index,
            function=call.function,
            arguments=call.args,
        )
        canonical_snapshot_failures += int(
            not _canonical_json_roundtrip_ok(transition.canonical_state_before)
        )
        canonical_snapshot_failures += int(
            not _canonical_json_roundtrip_ok(transition.canonical_state_after)
        )
        traces.append(call)
        utility, source = _utility(task, pre_environment, environment, traces)
        utilities.append(utility)
        utility_sources.append(source)
        completed, remaining = match_completed_goal_slots(traces, goal_slots)
        for change in transition.canonical_state_delta:
            path = str(change.get("path", ""))
            root = "/" + path.lstrip("/").split("/", 1)[0] if path else "<root>"
            delta_roots[root] += 1
        transitions.append(
            {
                "event_index": event_index,
                "tool_schema_id": transition.tool_schema_id,
                "tool_execution_status": transition.tool_execution_status,
                "tool_error_type": transition.tool_error_type,
                "argument_slot_count": len(transition.normalized_argument_slots),
                "entity_links": _resolution_counts(
                    transition.argument_entity_links
                ),
                "state_changed": transition.state_changed,
                "state_delta_operation_count": len(
                    transition.canonical_state_delta
                ),
                "state_before_bytes": _json_size(
                    transition.canonical_state_before
                ),
                "state_after_bytes": _json_size(
                    transition.canonical_state_after
                ),
                "state_delta_bytes": _json_size(
                    transition.canonical_state_delta
                ),
                "completed_target_slots": len(completed),
                "remaining_target_slots": len(remaining),
                "target_slot_progress": (
                    len(completed) / len(goal_slots) if goal_slots else 1.0
                ),
                "official_utility": utility,
                "official_utility_source": source,
                "task_progress_delta": None,
                "irreversible_effect": None,
            }
        )
    if ground_truth:
        final_utility = utilities[-1]
        first_true = next(
            (index for index, value in enumerate(utilities) if value), None
        )
    else:
        final_utility, final_source = _utility(
            task, pre_environment, environment, []
        )
        utilities = [final_utility]
        utility_sources = [final_source]
        first_true = 0 if final_utility else None
    early_true = (
        first_true is not None and bool(ground_truth) and first_true < len(ground_truth) - 1
    )
    monotonic = not any(
        utilities[index] and not utilities[index + 1]
        for index in range(len(utilities) - 1)
    )
    entity_counts = Counter()
    for row in transitions:
        entity_counts.update(row["entity_links"])
    return {
        "task_id": task.ID,
        "difficulty": str(task.DIFFICULTY.name),
        "ground_truth_call_count": len(ground_truth),
        "unique_ground_truth_call_signatures": len(set(signatures)),
        "duplicate_ground_truth_call_signatures": len(signatures) - len(set(signatures)),
        "placeholder_argument_calls": sum(
            call.placeholder_args is not None for call in ground_truth
        ),
        "canonical_state_json_roundtrip_failures": canonical_snapshot_failures,
        "initial_state_bytes": _json_size(pre_environment.model_dump(mode="json")),
        "state_mutating_calls": sum(row["state_changed"] for row in transitions),
        "read_only_calls": sum(not row["state_changed"] for row in transitions),
        "execution_error_calls": sum(
            row["tool_execution_status"] == "error" for row in transitions
        ),
        "state_delta_operations": sum(
            row["state_delta_operation_count"] for row in transitions
        ),
        "state_delta_roots": dict(sorted(delta_roots.items())),
        "argument_entity_link_resolution": {
            name: int(entity_counts.get(name, 0))
            for name in ("no_match", "unique", "ambiguous")
        },
        "official_utility_series": utilities,
        "official_utility_sources": sorted(set(utility_sources)),
        "official_utility_first_true_event": first_true,
        "official_utility_true_before_final_call": early_true,
        "official_utility_monotonic": monotonic,
        "final_ground_truth_utility": bool(final_utility),
        "all_target_slots_match_expert_trace": (
            not goal_slots
            or transitions[-1]["remaining_target_slots"] == 0
        ),
        "transitions": transitions,
    }


def _aggregate(task_rows: list[dict], tool_manifest: tuple[dict, ...]) -> dict:
    calls = [transition for task in task_rows for transition in task["transitions"]]
    entity = Counter()
    for task in task_rows:
        entity.update(task["argument_entity_link_resolution"])
    state_sizes = [row["state_before_bytes"] for row in calls]
    delta_sizes = [row["state_delta_bytes"] for row in calls]
    return {
        "task_count": len(task_rows),
        "tool_count": len(tool_manifest),
        "ground_truth_call_count": len(calls),
        "tasks_with_no_ground_truth_calls": sum(
            task["ground_truth_call_count"] == 0 for task in task_rows
        ),
        "tasks_with_all_read_only_calls": sum(
            task["ground_truth_call_count"] > 0
            and task["state_mutating_calls"] == 0
            for task in task_rows
        ),
        "state_mutating_call_count": sum(row["state_changed"] for row in calls),
        "read_only_call_count": sum(not row["state_changed"] for row in calls),
        "execution_error_call_count": sum(
            row["tool_execution_status"] == "error" for row in calls
        ),
        "final_ground_truth_utility_successes": sum(
            task["final_ground_truth_utility"] for task in task_rows
        ),
        "early_official_utility_true_tasks": sum(
            task["official_utility_true_before_final_call"] for task in task_rows
        ),
        "non_monotonic_official_utility_tasks": sum(
            not task["official_utility_monotonic"] for task in task_rows
        ),
        "expert_trace_slot_match_failures": sum(
            not task["all_target_slots_match_expert_trace"] for task in task_rows
        ),
        "duplicate_goal_slot_signature_count": sum(
            task["duplicate_ground_truth_call_signatures"] for task in task_rows
        ),
        "placeholder_argument_call_count": sum(
            task["placeholder_argument_calls"] for task in task_rows
        ),
        "canonical_state_json_roundtrip_failures": sum(
            task["canonical_state_json_roundtrip_failures"] for task in task_rows
        ),
        "argument_entity_link_resolution": {
            name: int(entity.get(name, 0))
            for name in ("no_match", "unique", "ambiguous")
        },
        "state_snapshot_bytes": {
            "mean": statistics.fmean(state_sizes) if state_sizes else 0.0,
            "maximum": max(state_sizes, default=0),
        },
        "state_delta_bytes": {
            "mean": statistics.fmean(delta_sizes) if delta_sizes else 0.0,
            "maximum": max(delta_sizes, default=0),
        },
        "tool_metadata": {
            "all_have_json_schema": all(bool(row["parameters"]) for row in tool_manifest),
            "dynamic_preconditions_available": any(
                row["precondition_metadata_available"] for row in tool_manifest
            ),
            "irreversibility_annotations_available": any(
                row["irreversibility_metadata_available"] for row in tool_manifest
            ),
        },
    }


def _state_delta_adapter_ready(overall: dict[str, Any]) -> bool:
    """Apply the complete preregistered state-adapter gate."""

    return bool(
        overall["execution_error_call_count"] == 0
        and overall["final_ground_truth_utility_successes"]
        == overall["task_count"]
        and overall["expert_trace_slot_match_failures"] == 0
        and overall["canonical_state_json_roundtrip_failures"] == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["banking", "slack", "travel", "workspace"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suites = {}
    for suite_name in args.suites:
        suite = get_suite(args.benchmark_version, suite_name)
        manifest = candidate_tool_manifest(FunctionsRuntime(suite.tools))
        tasks = [
            _audit_task(suite, task, FunctionsRuntime)
            for _, task in sorted(suite.user_tasks.items())
        ]
        suites[suite_name] = {
            "aggregate": _aggregate(tasks, manifest),
            "candidate_tool_manifest": manifest,
            "tasks": tasks,
        }

    all_tasks = [task for suite in suites.values() for task in suite["tasks"]]
    all_tools = [
        tool for suite in suites.values() for tool in suite["candidate_tool_manifest"]
    ]
    overall = _aggregate(all_tasks, tuple(all_tools))
    entity = overall["argument_entity_link_resolution"]
    entity_total = sum(entity.values())
    report = {
        "scope": SCOPE,
        "benchmark_version": args.benchmark_version,
        "safety_contract": {
            "clean_user_tasks_only": True,
            "ground_truth_replay_only": True,
            "llm_loaded": False,
            "attacks_constructed": False,
            "external_endpoints": False,
            "training_examples_created": False,
        },
        "overall": overall,
        "field_readiness": {
            "canonical_state_snapshot": (
                overall["canonical_state_json_roundtrip_failures"] == 0
            ),
            "canonical_state_delta": _state_delta_adapter_ready(overall),
            "tool_execution_status": True,
            "normalized_argument_slots": True,
            "argument_entity_links": {
                "status": "exact-match heuristic with ambiguity retained",
                "unique_fraction": (
                    entity["unique"] / entity_total if entity_total else 0.0
                ),
                "not_validated_as_model_input": True,
            },
            "task_progress_delta": {
                "official_fractional_progress_available": False,
                "official_final_utility_available": True,
                "expert_call_slots_available_as_target_only": True,
                "warning": (
                    "Ground-truth call completion is an offline diagnostic target; "
                    "it is not a label-blind victim-dynamics input and may not cover "
                    "equivalent alternative plans."
                ),
            },
            "irreversible_effect": {
                "available": False,
                "reason": "AgentDojo Function metadata has no irreversibility annotation",
            },
            "dynamic_candidate_preconditions": {
                "available": False,
                "reason": (
                    "FunctionsRuntime exposes suite-wide functions and validates at "
                    "execution time; it has no current-state precondition interface"
                ),
            },
        },
        "suites": suites,
        "decision_rule": {
            "state_delta_adapter_ready": _state_delta_adapter_ready(overall),
            "progress_model_ready": False,
            "irreversibility_model_ready": False,
            "next_step": (
                "instrument observed clean victim tool execution; retain official "
                "utility only at episode end and ground-truth slots as target-only audit"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"overall": overall, "field_readiness": report["field_readiness"], "decision_rule": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
