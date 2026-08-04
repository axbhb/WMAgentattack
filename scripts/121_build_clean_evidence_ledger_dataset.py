"""Build the frozen clean-only evidence-ledger architecture-probe dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import (
    build_ground_truth_goal_slots,
    instrument_function_call,
    match_completed_goal_slots,
)
from wmagentattack.decision_state import canonical_json_value, stable_fingerprint
from wmagentattack.evidence_ledger import (
    EvidenceLedger,
    evidence_item_text,
    ledger_length_features,
    ledger_text,
    update_evidence_ledger,
)
from wmagentattack.trace_execution_pairing import pair_executed_clean_tool_calls


SCOPE = "AgentDojo v1.2.2 existing Travel clean traces; evidence-ledger probe"


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


def _trusted_goal(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        _message_text(message)
        for message in messages
        if message.get("role") == "user"
    ).strip()


def _delta_roots(delta: tuple[dict[str, Any], ...]) -> dict[str, int]:
    roots = Counter()
    for change in delta:
        path = str(change.get("path", ""))
        root = "/" + path.lstrip("/").split("/", 1)[0] if path else "<root>"
        roots[root] += 1
    return dict(sorted(roots.items()))


def _resolution_counts(links) -> dict[str, int]:
    counts = Counter(link.resolution for link in links)
    return {
        key: int(counts.get(key, 0))
        for key in ("no_match", "unique", "ambiguous")
    }


def _assert_feature_contract(features: dict[str, Any]) -> None:
    serialized = json.dumps(features, ensure_ascii=False).lower()
    forbidden = (
        '"utility"',
        '"security"',
        '"expert_slot',
        '"coverage_target"',
        '"future_',
        '"attack_',
    )
    leaked = [token for token in forbidden if token in serialized]
    if leaked:
        raise ValueError(f"prefix feature leakage: {leaked}")


def _load_sources(protocol: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    rows = []
    chunks = []
    for panel in protocol["source_panels"]:
        root = Path(panel["archive_root"])
        panel_rows = []
        for seed in panel["seeds"]:
            paths = sorted((root / f"seed{seed}").glob("chunk*.json"))
            if len(paths) != int(panel["chunks_per_seed"]):
                raise ValueError(f"unexpected chunk count for {panel['name']} seed {seed}")
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload["run_seed"]) != int(seed):
                    raise ValueError(f"seed mismatch in {path}")
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
                        raise ValueError(f"incomplete clean result in {path}")
                    panel_rows.append(
                        {
                            "panel": panel["name"],
                            "seed": int(seed),
                            "result": result,
                        }
                    )
        if len(panel_rows) != int(panel["expected_episodes"]):
            raise ValueError(f"unexpected episode count for {panel['name']}")
        successes = sum(bool(row["result"]["utility"]) for row in panel_rows)
        if successes != int(panel["expected_utility_successes"]):
            raise ValueError(f"utility count mismatch for {panel['name']}")
        rows.extend(panel_rows)
    return rows, chunks


def _clean_trace_contract(trace: dict[str, Any]) -> None:
    if trace.get("attack_type") not in (None, "none"):
        raise ValueError("non-clean attack_type")
    if trace.get("injection_task_id") not in (None, "none"):
        raise ValueError("non-clean injection_task_id")
    if trace.get("injections") not in (None, {}, []):
        raise ValueError("non-clean injections")
    if trace.get("suite_name") != "travel":
        raise ValueError("non-Travel trace")


def _utility(task, output, pre_environment, environment, calls) -> bool:
    result = task.utility_from_traces(output, pre_environment, environment, calls)
    if result is not None:
        return bool(result)
    return bool(task.utility(output, pre_environment, environment))


def _build_episode(source, suite, functions_runtime_type, function_call_type) -> dict:
    result = source["result"]
    trace_path = Path(result["raw_trace"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    _clean_trace_contract(trace)
    if trace.get("user_task_id") != result["user_task_id"]:
        raise ValueError("task mismatch")
    pairing, final_output = pair_executed_clean_tool_calls(trace["messages"])
    if not pairing.executed_alignment_ok:
        raise ValueError("executed-call pairing defect")

    task = suite.get_user_task_by_id(result["user_task_id"])
    environment = task.init_environment(suite.load_and_inject_default_environment({}))
    pre_environment = environment.model_copy(deep=True)
    runtime = functions_runtime_type(suite.tools)
    expert_calls = list(task.ground_truth(pre_environment.model_copy(deep=True)))
    goal_slots = build_ground_truth_goal_slots(expert_calls, pre_environment)
    goal = _trusted_goal(trace["messages"])
    observed_calls = []
    ledger = EvidenceLedger()
    cumulative_state_changes = 0
    cumulative_errors = 0
    cumulative_roots = Counter()
    prefixes = []

    initial_features = {
        "trusted_goal": goal,
        "last_event": {
            "tool_name": "<START>",
            "arguments": {},
            "execution_status": "start",
            "error_type": None,
        },
        "state_summary": {
            "last_state_changed": False,
            "cumulative_state_changes": 0,
            "cumulative_errors": 0,
            "delta_roots": {},
        },
        "canonical_state": canonical_json_value(pre_environment),
        "evidence_text": "",
        "new_evidence_text": "",
        "evidence_length": ledger_length_features(ledger),
        "prefix_length": 0,
    }
    _assert_feature_contract(initial_features)
    prefixes.append(
        {
            "prefix_index": 0,
            "features": initial_features,
            "targets": {
                "expert_slot_coverage": 0.0 if goal_slots else 1.0,
                "is_final_prefix": not pairing.executed_pairs,
            },
        }
    )

    for step_index, pair in enumerate(pairing.executed_pairs):
        proposal = pair.proposal
        call = function_call_type(
            function=proposal.function, args=proposal.arguments
        )
        transition, _ = instrument_function_call(
            runtime,
            environment,
            event_index=step_index,
            function=call.function,
            arguments=call.args,
        )
        runtime_error = transition.tool_execution_status == "error"
        if runtime_error != pair.logged_error:
            raise ValueError("runtime/logged error mismatch")
        observed_calls.append(call)
        tool_message = trace["messages"][pair.tool_message_index]
        link_counts = _resolution_counts(transition.argument_entity_links)
        previous_evidence_count = len(ledger.items)
        ledger = update_evidence_ledger(
            ledger,
            goal=goal,
            tool_name=call.function,
            arguments=call.args,
            observation_text=_message_text(tool_message),
            step_index=step_index,
            execution_status=transition.tool_execution_status,
            error_type=transition.tool_error_type,
            argument_link_resolution=link_counts,
            state_changed=transition.state_changed,
        )
        cumulative_state_changes += int(transition.state_changed)
        cumulative_errors += int(runtime_error)
        roots = _delta_roots(transition.canonical_state_delta)
        cumulative_roots.update(roots)
        completed, _ = match_completed_goal_slots(observed_calls, goal_slots)
        coverage = len(completed) / len(goal_slots) if goal_slots else 1.0
        features = {
            "trusted_goal": goal,
            "last_event": {
                "tool_name": call.function,
                "arguments": call.args,
                "execution_status": transition.tool_execution_status,
                "error_type": transition.tool_error_type,
            },
            "state_summary": {
                "last_state_changed": transition.state_changed,
                "cumulative_state_changes": cumulative_state_changes,
                "cumulative_errors": cumulative_errors,
                "delta_roots": dict(sorted(cumulative_roots.items())),
            },
            "canonical_state": transition.canonical_state_after,
            "evidence_text": ledger_text(ledger),
            "new_evidence_text": "\n".join(
                evidence_item_text(item)
                for item in ledger.items[previous_evidence_count:]
            ),
            "evidence_length": ledger_length_features(ledger),
            "prefix_length": step_index + 1,
        }
        _assert_feature_contract(features)
        prefixes.append(
            {
                "prefix_index": step_index + 1,
                "features": features,
                "targets": {
                    "expert_slot_coverage": coverage,
                    "is_final_prefix": step_index + 1 == len(pairing.executed_pairs),
                },
            }
        )

    replay_utility = _utility(
        task, final_output, pre_environment, environment, observed_calls
    )
    if replay_utility != bool(result["utility"]):
        raise ValueError("recomputed utility mismatch")
    return {
        "episode_id": f"{source['panel']}::{source['seed']}::{result['row_id']}",
        "panel": source["panel"],
        "seed": source["seed"],
        "suite": "travel",
        "task_id": result["user_task_id"],
        "source_trace_sha256": _sha256(trace_path),
        "pairing": {
            "proposal_count": pairing.proposal_count,
            "executed_call_count": len(pairing.executed_pairs),
            "terminal_unexecuted_count": len(
                pairing.terminal_unexecuted_proposals
            ),
            "terminal_unexecuted_tools": [
                proposal.function
                for proposal in pairing.terminal_unexecuted_proposals
            ],
        },
        "targets": {
            "final_utility": replay_utility,
            "expert_slot_count": len(goal_slots),
        },
        "final_output_fingerprint": stable_fingerprint(final_output),
        "prefixes": prefixes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("protocol is not preregistered")

    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    sources, chunks = _load_sources(protocol)
    suite = get_suite(protocol["scope"]["benchmark_version"], "travel")
    episodes = [
        _build_episode(source, suite, FunctionsRuntime, FunctionCall)
        for source in sources
    ]
    expected_tasks = {
        task
        for tasks in protocol["task_folds"].values()
        for task in tasks
    }
    observed_tasks = {episode["task_id"] for episode in episodes}
    if observed_tasks != expected_tasks:
        raise ValueError("task fold coverage mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "episodes.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    prefix_count = sum(len(episode["prefixes"]) for episode in episodes)
    executed_calls = sum(
        episode["pairing"]["executed_call_count"] for episode in episodes
    )
    audit = {
        "scope": SCOPE,
        "protocol_id": protocol["protocol_id"],
        "safety_contract": {
            "clean_traces_only": True,
            "llm_loaded": False,
            "attacks_constructed": False,
            "external_endpoints": False,
            "expert_coverage_in_features": False,
            "utility_in_prefix_features": False,
            "terminal_unexecuted_replayed": False,
        },
        "counts": {
            "episodes": len(episodes),
            "tasks": len(observed_tasks),
            "prefixes": prefix_count,
            "executed_calls": executed_calls,
            "proposals": sum(
                episode["pairing"]["proposal_count"] for episode in episodes
            ),
            "terminal_unexecuted": sum(
                episode["pairing"]["terminal_unexecuted_count"]
                for episode in episodes
            ),
            "utility_successes": sum(
                episode["targets"]["final_utility"] for episode in episodes
            ),
            "evidence_items": sum(
                prefix["features"]["evidence_length"]["item_count"]
                for episode in episodes
                for prefix in episode["prefixes"][-1:]
            ),
        },
        "task_folds": protocol["task_folds"],
        "input_chunks": chunks,
        "dataset_sha256": _sha256(dataset_path),
    }
    _write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
