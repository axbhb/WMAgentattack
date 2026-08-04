"""Build the frozen causal-prefix dataset for the panel-v2 architecture ablation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from wmagentattack import custom_agentdojo_panel_v2 as panel
from wmagentattack.clean_state_instrumentation import (
    candidate_tool_manifest,
    instrument_function_call,
)
from wmagentattack.decision_state import canonical_json_value
from wmagentattack.panel_v2_architecture_probe import (
    assess_obligation_progress,
    canonical_argument_key_target,
    canonical_executed_call,
    ledger_feature_payload,
    load_panel_v2_adapter_registry,
)
from wmagentattack.structured_ledger_v2 import (
    AdapterMode,
    ExecutionChannelStatus,
    StructuredEvidenceLedgerV2,
    update_structured_ledger,
)
from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


DATASET_SCHEMA_VERSION = "wmagentattack.custom_clean_panel_v2_architecture_dataset.v2"
STOP = "STOP"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("content", block.get("text", "")))
        for block in content
        if isinstance(block, Mapping)
    )


def _trusted_goal(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        _message_text(message)
        for message in messages
        if message.get("role") == "user"
    ).strip()


def _parse_logged_output(text: str, mode: AdapterMode) -> Any:
    if mode in {
        AdapterMode.NAME_LIST_TEXT,
        AdapterMode.FLIGHT_LINES,
    }:
        return text
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = None
    if mode == AdapterMode.VALUE:
        return text if parsed is None else parsed
    if mode == AdapterMode.MUTATION_ACK:
        return text if parsed is None else parsed
    return parsed


def _delta_roots(delta: tuple[dict[str, Any], ...]) -> dict[str, int]:
    roots = Counter()
    for change in delta:
        path = str(change.get("path", ""))
        root = "/" + path.lstrip("/").split("/", 1)[0] if path else "<root>"
        roots[root] += 1
    return dict(sorted(roots.items()))


def _candidate_id(suite: str, function: str) -> str:
    return STOP if function == STOP else f"{suite}::{function}"


def _build_catalog(suites: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    catalog: dict[str, Any] = {
        STOP: {
            "candidate_id": STOP,
            "suite": "all",
            "name": STOP,
            "description": "Stop using tools and answer the trusted user request.",
            "parameters": {},
            "dependencies": [],
        }
    }
    legal: dict[str, list[str]] = {}
    argument_keys = set()
    for suite_name, suite in sorted(suites.items()):
        runtime = FunctionsRuntime(suite.tools)
        rows = []
        for descriptor in candidate_tool_manifest(runtime):
            candidate_id = _candidate_id(suite_name, descriptor["name"])
            if candidate_id in catalog:
                raise ValueError(f"duplicate candidate ID: {candidate_id}")
            catalog[candidate_id] = {
                "candidate_id": candidate_id,
                "suite": suite_name,
                **descriptor,
            }
            rows.append(candidate_id)
            properties = descriptor.get("parameters", {}).get("properties", {})
            argument_keys.update(str(key) for key in properties)
        legal[suite_name] = [*sorted(rows), STOP]
    return catalog, legal, sorted(argument_keys)


def _prefix(
    *,
    prefix_index: int,
    goal: str,
    track: str,
    legal_tools: list[str],
    last_action: Mapping[str, Any],
    last_observation: str,
    execution_receipt: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    ledger: StructuredEvidenceLedgerV2,
    next_action: str,
    next_argument_keys: tuple[str, ...],
    obligations: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "prefix_index": prefix_index,
        "features": {
            "trusted_goal": goal,
            "track": track,
            "prefix_index": prefix_index,
            "legal_tools": legal_tools,
            "last_action": canonical_json_value(dict(last_action)),
            "last_observation": last_observation,
            "execution_receipt": canonical_json_value(dict(execution_receipt)),
            "causal_state_summary": canonical_json_value(dict(state_summary)),
            "ledger_v2": ledger_feature_payload(ledger),
        },
        "targets": {
            "next_action": next_action,
            "stop": next_action == STOP,
            "argument_keys": list(next_argument_keys),
            "evidence_obligations": obligations,
        },
    }


def _build_episode(
    row: Mapping[str, Any],
    *,
    spec: Any,
    suite: Any,
    registry: Any,
    legal_tools: list[str],
) -> tuple[dict[str, Any], set[str]]:
    trace_path = Path(str(row["source_trace"]))
    if _sha256(trace_path) != row["source_trace_sha256"]:
        raise ValueError(f"trace hash mismatch: {trace_path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("attack_type") not in (None, "none") or trace.get("injections") not in ({}, [], None):
        raise ValueError("architecture dataset must remain clean-only")
    if trace.get("suite_name") != spec.suite or trace.get("user_task_id") != spec.task_id:
        raise ValueError("trace/task metadata mismatch")
    pairing, _ = pair_executed_clean_tool_calls(trace["messages"])
    if not pairing.executed_alignment_ok or pairing.assistant_multi_call_message_count:
        raise ValueError("unsupported execution pairing")

    tool_map = {tool.name: tool for tool in suite.tools}
    runtime = FunctionsRuntime(suite.tools)
    environment = suite.load_and_inject_default_environment({})
    ledger = StructuredEvidenceLedgerV2()
    calls = []
    prefixes = []
    used_adapters: set[str] = set()
    cumulative_state_changes = 0
    cumulative_errors = 0
    cumulative_roots: Counter[str] = Counter()
    goal = _trusted_goal(trace["messages"])
    pairs = list(pairing.executed_pairs)

    first_action = (
        _candidate_id(spec.suite, pairs[0].proposal.function) if pairs else STOP
    )
    first_argument_keys = (
        canonical_argument_key_target(
            tool_map[pairs[0].proposal.function], pairs[0].proposal.arguments
        )
        if pairs
        else ()
    )
    prefixes.append(
        _prefix(
            prefix_index=0,
            goal=goal,
            track=str(row["track"]),
            legal_tools=legal_tools,
            last_action={"function": "<START>", "arguments": {}},
            last_observation="",
            execution_receipt={"status": "start", "error_type": None},
            state_summary={
                "last_state_changed": False,
                "cumulative_state_changes": 0,
                "cumulative_errors": 0,
                "last_delta_count": 0,
                "delta_roots": {},
            },
            ledger=ledger,
            next_action=first_action,
            next_argument_keys=first_argument_keys,
            obligations=assess_obligation_progress(calls, spec.contract, suite.tools),
        )
    )

    for call_index, pair in enumerate(pairs):
        proposal = pair.proposal
        if proposal.function not in tool_map:
            raise ValueError(f"unknown tool: {proposal.function}")
        transition, runtime_output = instrument_function_call(
            runtime,
            environment,
            event_index=call_index,
            function=proposal.function,
            arguments=proposal.arguments,
        )
        if (transition.tool_execution_status == "error") != bool(pair.logged_error):
            raise ValueError("runtime/logged error mismatch")
        tool_message = trace["messages"][pair.tool_message_index]
        recorded_error = (
            None if tool_message.get("error") is None else str(tool_message.get("error"))
        )
        call = canonical_executed_call(
            call_index=call_index,
            function=proposal.function,
            arguments=proposal.arguments,
            error=recorded_error,
            tools=tool_map,
            mutating_tools=set(panel.MUTATING_TOOLS),
        )
        if call.executed_successfully != (transition.tool_execution_status == "success"):
            raise ValueError("canonical execution status mismatch")
        calls.append(call)

        channel = (
            ExecutionChannelStatus.EXECUTED_ERROR
            if recorded_error is not None
            else ExecutionChannelStatus.EXECUTED_SUCCESS
        )
        spec_adapter = registry.adapters.get(proposal.function)
        if spec_adapter is None:
            raise KeyError(f"no panel-v2 ledger adapter for {proposal.function}")
        logged_output = _parse_logged_output(_message_text(tool_message), spec_adapter.mode)
        if recorded_error is None and logged_output is None and spec_adapter.mode not in {
            AdapterMode.MUTATION_ACK,
            AdapterMode.VALUE,
        }:
            raise ValueError(f"could not parse logged output for {proposal.function}")
        ledger = update_structured_ledger(
            ledger,
            registry,
            episode_id=f"{row['track']}::{row['run_seed']}::{row['row_id']}",
            call_index=call_index,
            channel_status=channel,
            tool_name=proposal.function,
            arguments=call.canonical_args,
            runtime_output=logged_output if recorded_error is None else runtime_output,
            error_type=transition.tool_error_type,
            state_changed=transition.state_changed,
        ).ledger
        used_adapters.add(proposal.function)

        cumulative_state_changes += int(transition.state_changed)
        cumulative_errors += int(recorded_error is not None)
        roots = _delta_roots(transition.canonical_state_delta)
        cumulative_roots.update(roots)
        next_pair = pairs[call_index + 1] if call_index + 1 < len(pairs) else None
        next_action = (
            _candidate_id(spec.suite, next_pair.proposal.function)
            if next_pair is not None
            else STOP
        )
        next_argument_keys = (
            canonical_argument_key_target(
                tool_map[next_pair.proposal.function], next_pair.proposal.arguments
            )
            if next_pair is not None
            else ()
        )
        prefixes.append(
            _prefix(
                prefix_index=call_index + 1,
                goal=goal,
                track=str(row["track"]),
                legal_tools=legal_tools,
                last_action={
                    "function": call.function,
                    "arguments": call.canonical_args,
                },
                last_observation=_message_text(tool_message),
                execution_receipt={
                    "status": transition.tool_execution_status,
                    "error_type": transition.tool_error_type,
                    "output_type": transition.tool_output_type,
                },
                state_summary={
                    "last_state_changed": transition.state_changed,
                    "cumulative_state_changes": cumulative_state_changes,
                    "cumulative_errors": cumulative_errors,
                    "last_delta_count": len(transition.canonical_state_delta),
                    "delta_roots": dict(sorted(cumulative_roots.items())),
                },
                ledger=ledger,
                next_action=next_action,
                next_argument_keys=next_argument_keys,
                obligations=assess_obligation_progress(
                    calls, spec.contract, suite.tools
                ),
            )
        )

    for prefix in prefixes:
        if prefix["targets"]["next_action"] not in legal_tools:
            raise ValueError("target action is outside the legal candidate set")
    return (
        {
            "episode_id": f"{row['track']}::{row['run_seed']}::{row['row_id']}",
            "task_id": str(row["row_id"]),
            "suite": spec.suite,
            "split": spec.split,
            "track": str(row["track"]),
            "run_seed": int(row["run_seed"]),
            "task_difficulty": spec.contract.task_difficulty.value,
            "task_archetype": spec.contract.task_archetype,
            "source_trace_sha256": str(row["source_trace_sha256"]),
            "prefixes": prefixes,
        },
        used_adapters,
    )


def build(summary_path: Path, extension_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["execution_integrity"]["passed"]:
        raise ValueError("source execution integrity did not pass")
    if not summary["gates"]["dynamics"]["passed"] or not summary["gates"]["evidence_progress"]["passed"]:
        raise ValueError("source dynamics/evidence gates did not pass")
    if not summary["permissions"]["three_backbone_dynamics_evidence_ablation"]:
        raise ValueError("architecture ablation is not permitted")
    if summary["permissions"]["attack_data"] or summary["permissions"]["dreamer_training"]:
        raise ValueError("source permissions unexpectedly opened attack/Dreamer work")

    specs = {spec.row_id: spec for spec in panel.TASK_SPECS}
    suite_names = sorted({spec.suite for spec in panel.TASK_SPECS})
    suites = {name: get_suite(panel.BENCHMARK_VERSION, name) for name in suite_names}
    registry = load_panel_v2_adapter_registry(extension_path)
    catalog, legal, argument_keys = _build_catalog(suites)
    catalog_tool_names = {
        descriptor["name"]
        for candidate_id, descriptor in catalog.items()
        if candidate_id != STOP
    }
    missing_adapters = sorted(catalog_tool_names - set(registry.adapters))

    episodes = []
    used_adapters = set()
    for row in summary["episodes"]:
        spec = specs[str(row["row_id"])]
        episode, used = _build_episode(
            row,
            spec=spec,
            suite=suites[spec.suite],
            registry=registry,
            legal_tools=legal[spec.suite],
        )
        episodes.append(episode)
        used_adapters.update(used)
    episodes.sort(key=lambda row: row["episode_id"])

    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_summary_sha256": _sha256(summary_path),
        "adapter_extension_sha256": _sha256(extension_path),
        "benchmark_version": panel.BENCHMARK_VERSION,
        "scope": "clean-only causal prefixes for dynamics/evidence architecture ablation",
        "tool_catalog": catalog,
        "suite_legal_tools": legal,
        "argument_key_vocab": argument_keys,
        "episodes": episodes,
    }
    prefixes = [prefix for episode in episodes for prefix in episode["prefixes"]]
    obligation_rows = [
        row
        for prefix in prefixes
        for row in prefix["targets"]["evidence_obligations"]
    ]
    argument_key_vocab = set(argument_keys)
    unknown_argument_target_keys = sorted(
        {
            key
            for prefix in prefixes
            for key in prefix["targets"]["argument_keys"]
            if key not in argument_key_vocab
        }
    )
    serialized_features = json.dumps(
        [prefix["features"] for prefix in prefixes], ensure_ascii=False
    ).casefold()
    forbidden_hits = [
        token
        for token in (
            '"utility"',
            '"security"',
            '"factorized"',
            '"final_output"',
            '"final_report"',
            '"expert_calls"',
            '"future_calls"',
        )
        if token in serialized_features
    ]
    audit = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_summary_sha256": _sha256(summary_path),
        "episodes": len(episodes),
        "independent_tasks": len({row["task_id"] for row in episodes}),
        "prefixes": len(prefixes),
        "executed_calls": sum(len(row["prefixes"]) - 1 for row in episodes),
        "episodes_by_split": dict(Counter(row["split"] for row in episodes)),
        "episodes_by_track": dict(Counter(row["track"] for row in episodes)),
        "tasks_by_split": {
            split: len({row["task_id"] for row in episodes if row["split"] == split})
            for split in ("training", "calibration", "confirmation")
        },
        "action_targets": dict(
            Counter(prefix["targets"]["next_action"] for prefix in prefixes)
        ),
        "evidence_statuses": dict(Counter(row["status"] for row in obligation_rows)),
        "tool_catalog_size": len(catalog),
        "argument_key_vocab_size": len(argument_keys),
        "unknown_argument_target_keys": unknown_argument_target_keys,
        "adapter_registry_size": len(registry.adapters),
        "used_adapters": sorted(used_adapters),
        "used_adapter_count": len(used_adapters),
        "missing_legal_tool_adapters": missing_adapters,
        "forbidden_feature_hits": forbidden_hits,
        "gates": {
            "exact_144_clean_episodes": len(episodes) == 144,
            "exact_48_independent_tasks": len({row["task_id"] for row in episodes}) == 48,
            "frozen_split_counts_72_36_36": Counter(row["split"] for row in episodes)
            == {"training": 72, "calibration": 36, "confirmation": 36},
            "task_split_counts_24_12_12": {
                split: len({row["task_id"] for row in episodes if row["split"] == split})
                for split in ("training", "calibration", "confirmation")
            }
            == {"training": 24, "calibration": 12, "confirmation": 12},
            "all_targets_legal": all(
                prefix["targets"]["next_action"]
                in dataset["suite_legal_tools"][episode["suite"]]
                for episode in episodes
                for prefix in episode["prefixes"]
            ),
            "all_argument_targets_in_declared_schema_vocab": not unknown_argument_target_keys,
            "all_executed_tools_have_adapters": used_adapters.issubset(
                registry.adapters
            ),
            "no_outcome_or_future_feature_leakage": not forbidden_hits,
            "attack_and_dreamer_remain_blocked": not summary["permissions"]["attack_data"]
            and not summary["permissions"]["dreamer_training"],
        },
    }
    audit["passed"] = all(audit["gates"].values())
    return dataset, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adapter-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    dataset, audit = build(args.summary, args.adapter_extension)
    _write_json(args.output, dataset)
    audit["dataset_sha256"] = _sha256(args.output)
    _write_json(args.audit_output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
