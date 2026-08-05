"""Merge interaction-faithful tau3 chunks and apply the frozen data gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_interactive import build_interactive_dataset
from wmagentattack.tau3_multistep import file_sha256


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _transition_support(
    dataset: Mapping[str, Any], target_names: Sequence[str]
) -> dict[str, Any]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for episode in dataset["episodes"]:
        for transition in episode["transitions"]:
            for name in target_names:
                value = str(int(float(transition["target"][name]) >= 0.5))
                counts[name][str(episode["split"])][value] += 1
    return {
        name: {
            split: {
                "negative": int(counts[name][split]["0"]),
                "positive": int(counts[name][split]["1"]),
            }
            for split in ("training", "calibration", "confirmation")
        }
        for name in target_names
    }


def _supported_targets(
    support: Mapping[str, Any], *, minimum_each_class: int
) -> list[str]:
    return sorted(
        name
        for name, splits in support.items()
        if all(
            int(splits[split][label]) >= minimum_each_class
            for split in ("training", "confirmation")
            for label in ("negative", "positive")
        )
    )


def evaluate_interactive_data_gate(
    *,
    metrics: Mapping[str, Any],
    integrity_checks: Mapping[str, bool],
    gate: Mapping[str, Any],
    minimum_supported_targets: int,
) -> dict[str, bool]:
    """Return every preregistered clause without adapting any threshold."""

    checks = {
        "episodes_complete": int(metrics["episodes_complete"])
        == int(gate["expected_episodes_complete"]),
        "runtime_failures": int(metrics["runtime_failures"]) == 0,
        "agent_private_scenario_exposures": int(
            metrics["agent_private_scenario_exposures"]
        )
        == 0,
        "agent_and_user_completions_nonempty": bool(
            integrity_checks["all_agent_and_user_completions_nonempty"]
        ),
        "tool_names_in_role_presented_schema": bool(
            integrity_checks["all_tool_names_in_role_presented_schema"]
        ),
        "complete_tool_sequences_replica_identical": bool(
            integrity_checks[
                "all_complete_tool_sequences_live_and_replicas_identical"
            ]
        ),
        "real_external_endpoint_calls": int(
            metrics["real_external_endpoint_calls"]
        )
        == 0,
        "communication_error_terminations": int(
            metrics["communication_error_terminations"]
        )
        == 0,
        "forced_budget_stop_episodes": int(metrics["forced_budget_stop_episodes"])
        <= int(gate["maximum_forced_budget_stop_episodes"]),
        "natural_user_messages": int(metrics["natural_user_messages"])
        >= int(gate["minimum_natural_user_messages"]),
        "adjacent_assistant_tool_transitions": int(
            metrics["adjacent_assistant_tool_transitions"]
        )
        >= int(gate["minimum_adjacent_assistant_tool_transitions"]),
        "episodes_with_two_or_more_assistant_transitions": int(
            metrics["episodes_with_two_or_more_assistant_transitions"]
        )
        >= int(gate["minimum_episodes_with_two_or_more_assistant_transitions"]),
        "tasks_with_at_least_one_assistant_transition": int(
            metrics["tasks_with_at_least_one_assistant_transition"]
        )
        >= int(gate["minimum_tasks_with_at_least_one_assistant_transition"]),
        "unique_executed_assistant_tools": int(
            metrics["unique_executed_assistant_tools"]
        )
        >= int(gate["minimum_unique_executed_assistant_tools"]),
        "agent_tool_decision_rate": float(gate["minimum_agent_tool_decision_rate"])
        <= float(metrics["agent_tool_decision_rate"])
        <= float(gate["maximum_agent_tool_decision_rate"]),
        "dominant_agent_action_fraction": float(
            metrics["dominant_agent_action_fraction"]
        )
        <= float(gate["maximum_dominant_agent_action_fraction"]),
        "state_changed_assistant_transitions": int(
            metrics["state_changed_assistant_transitions"]
        )
        >= int(gate["minimum_state_changed_assistant_transitions"]),
        "state_unchanged_assistant_transitions": int(
            metrics["state_unchanged_assistant_transitions"]
        )
        >= int(gate["minimum_state_unchanged_assistant_transitions"]),
        "tasks_with_state_changed_assistant_transition": int(
            metrics["tasks_with_state_changed_assistant_transition"]
        )
        >= int(gate["minimum_tasks_with_state_changed_assistant_transition"]),
        "domains_with_state_changed_assistant_transition": int(
            metrics["domains_with_state_changed_assistant_transition"]
        )
        >= int(gate["minimum_domains_with_state_changed_assistant_transition"]),
        "paired_state_changed_transition_gain_over_parent": int(
            metrics["paired_state_changed_transition_gain_over_parent"]
        )
        >= int(gate["minimum_paired_state_changed_transition_gain_over_parent"]),
        "supported_transition_targets": int(metrics["supported_transition_targets"])
        >= int(minimum_supported_targets),
    }
    checks.update({f"integrity::{name}": bool(value) for name, value in integrity_checks.items()})
    return checks


def _paired_parent_metrics(
    episodes: Sequence[Mapping[str, Any]], parent_dataset: Mapping[str, Any]
) -> dict[str, Any]:
    parents = {row["episode_id"]: row for row in parent_dataset["episodes"]}
    if {row["parent_episode_id"] for row in episodes} != set(parents):
        raise ValueError("interactive and parent paired episode surfaces differ")
    episode_rows = []
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    parent_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_catalog = parent_dataset["candidate_catalog"]
    for episode in episodes:
        parent = parents[episode["parent_episode_id"]]
        if str(parent["task_id"]) != str(episode["task_key"]):
            raise ValueError("paired task differs")
        if int(parent["run_seed"]) != int(episode["llm_seed"]):
            raise ValueError("paired agent seed differs")
        new_changed = sum(
            event["requestor"] == "assistant" and bool(event["state_changed"])
            for event in episode["combined_tool_events"]
        )
        parent_changed = sum(
            float(row["target"]["state_changed"]) >= 0.5
            for row in parent["transitions"]
        )
        domain = str(episode["domain"])
        task = str(episode["task_key"])
        by_domain[domain].update(new=new_changed, parent=parent_changed)
        by_task[task].update(new=new_changed, parent=parent_changed)
        for transition in parent["transitions"]:
            descriptor = candidate_catalog[str(transition["action"])]
            name = str(descriptor["function"]["name"])
            parent_by_tool[name].update(
                transitions=1,
                changed=int(
                    float(transition["target"]["state_changed"]) >= 0.5
                ),
            )
        episode_rows.append(
            {
                "episode_id": episode["episode_id"],
                "parent_episode_id": episode["parent_episode_id"],
                "task_key": task,
                "domain": domain,
                "llm_seed": int(episode["llm_seed"]),
                "new_state_changed": new_changed,
                "parent_state_changed": parent_changed,
                "gain": new_changed - parent_changed,
            }
        )
    new_total = sum(row["new_state_changed"] for row in episode_rows)
    parent_total = sum(row["parent_state_changed"] for row in episode_rows)
    return {
        "paired_episodes": len(episode_rows),
        "new_state_changed_transitions": new_total,
        "parent_state_changed_transitions": parent_total,
        "gain": new_total - parent_total,
        "positive_episode_fraction": sum(row["gain"] > 0 for row in episode_rows)
        / max(len(episode_rows), 1),
        "positive_task_fraction": sum(
            values["new"] > values["parent"] for values in by_task.values()
        )
        / max(len(by_task), 1),
        "by_domain": {
            name: {
                "new": int(values["new"]),
                "parent": int(values["parent"]),
                "gain": int(values["new"] - values["parent"]),
            }
            for name, values in sorted(by_domain.items())
        },
        "by_task": {
            name: {
                "new": int(values["new"]),
                "parent": int(values["parent"]),
                "gain": int(values["new"] - values["parent"]),
            }
            for name, values in sorted(by_task.items())
        },
        "parent_by_tool": {
            name: dict(values) for name, values in sorted(parent_by_tool.items())
        },
        "episodes": episode_rows,
    }


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    failed = [name for name, passed in summary["gate_checks"].items() if not passed]
    lines = [
        "# tau3 interaction-faithful repair pilot",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Frozen result",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for name in (
        "episodes_complete",
        "natural_user_messages",
        "adjacent_assistant_tool_transitions",
        "episodes_with_two_or_more_assistant_transitions",
        "tasks_with_at_least_one_assistant_transition",
        "unique_executed_assistant_tools",
        "agent_tool_decision_rate",
        "dominant_agent_action_fraction",
        "state_changed_assistant_transitions",
        "state_unchanged_assistant_transitions",
        "tasks_with_state_changed_assistant_transition",
        "domains_with_state_changed_assistant_transition",
        "paired_state_changed_transition_gain_over_parent",
        "supported_transition_targets",
        "forced_budget_stop_episodes",
        "runtime_failures",
    ):
        lines.append(f"| {name} | {_fmt(metrics[name])} |")
    lines.extend(
        [
            "",
            "## Gate clauses",
            "",
            *[
                f"- {name}: `{'PASS' if passed else 'FAIL'}`"
                for name, passed in summary["gate_checks"].items()
            ],
            "",
            "## Required counterevidence",
            "",
            f"- Failed clauses: `{json.dumps(failed, sort_keys=True)}`.",
            f"- Parent/new paired mutation comparison: `{json.dumps(summary['paired_parent']['by_domain'], sort_keys=True)}`.",
            f"- Termination distribution: `{json.dumps(summary['termination_counts'], sort_keys=True)}`.",
            f"- Agent tool errors: {metrics['assistant_tool_errors']}; user tool errors: {metrics['user_tool_errors']}.",
            f"- User tool events: {metrics['user_tool_events']}; serialization retries: {metrics['serialization_retry_calls']}.",
            "- The agent prompt contained policy, agent tools, and causally observed dialogue only; the private UserScenario remained in the user simulator.",
            "- A GO authorizes only the already-frozen task-disjoint predictive-method comparison. Large collection still requires that method gate to pass.",
            "- A NO-GO preserves this evidence and forbids threshold relaxation, method training, and scale-up.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_interactive_outcomes":
        raise ValueError("interaction protocol was not frozen before outcomes")
    if file_sha256(args.manifest) != protocol["frozen_manifest"]["sha256"]:
        raise ValueError("interactive manifest hash differs")
    if file_sha256(args.parent_dataset) != protocol["parent_no_go"][
        "dataset_sha256"
    ]:
        raise ValueError("parent paired dataset hash differs")
    for relative, expected in protocol["implementation_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"interactive implementation differs: {relative}")

    output_paths = sorted((args.archive / "outputs").glob("chunk*.json"))
    audit_paths = sorted((args.archive / "audits").glob("chunk*.json"))
    if not output_paths or not audit_paths:
        raise FileNotFoundError("interactive chunks or audits are missing")
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    chunk_indexes: set[int] = set()
    declared_chunk_counts: set[int] = set()
    endpoint_calls = 0
    for path in output_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol_id"] != protocol["protocol_id"]:
            raise ValueError(f"chunk protocol differs: {path}")
        chunk_indexes.add(int(payload["chunk_index"]))
        declared_chunk_counts.add(int(payload["num_chunks"]))
        episodes.extend(payload["episodes"])
        failures.extend(payload["failures"])
        endpoint_calls += int(payload.get("real_external_endpoint_calls", 0))
    if len(declared_chunk_counts) != 1:
        raise ValueError("chunk files disagree on num_chunks")
    num_chunks = next(iter(declared_chunk_counts))
    chunk_audits = [
        json.loads(path.read_text(encoding="utf-8")) for path in audit_paths
    ]
    complete_chunk_surface = (
        chunk_indexes == set(range(num_chunks))
        and len(audit_paths) == num_chunks
        and {int(row["chunk_index"]) for row in chunk_audits}
        == set(range(num_chunks))
    )
    for row in chunk_audits:
        failures.extend(row.get("runtime_failures", ()))
        endpoint_calls += int(row.get("real_external_endpoint_calls", 0))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset, dataset_audit = build_interactive_dataset(manifest, episodes)
    parent_dataset = json.loads(args.parent_dataset.read_text(encoding="utf-8"))
    paired = _paired_parent_metrics(episodes, parent_dataset)
    manifest_rows = {row["episode_id"]: row for row in manifest["rows"]}
    dataset_rows = {row["episode_id"]: row for row in dataset["episodes"]}

    completions_nonempty = True
    legal_tools = True
    termination_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    assistant_tools: Counter[str] = Counter()
    user_tools: Counter[str] = Counter()
    assistant_changed_by_tool: Counter[str] = Counter()
    assistant_errors = 0
    user_errors = 0
    nondeterministic = 0
    private_exposures = 0
    natural_user_messages = 0
    agent_logical_calls = 0
    user_logical_calls = 0
    physical_calls = 0
    retry_calls = 0
    forced_stops = 0
    communication_errors = 0
    state_changed = 0
    state_unchanged = 0
    user_tool_events = 0
    tasks_with_transition: set[str] = set()
    tasks_with_changed: set[str] = set()
    domains_with_changed: set[str] = set()
    episodes_with_two = 0
    total_combined_events = 0
    user_stop_generations = 0
    for episode in episodes:
        row = manifest_rows[episode["episode_id"]]
        dataset_episode = dataset_rows[episode["episode_id"]]
        termination = str(episode["termination"]).lower()
        termination_counts[termination] += 1
        communication_errors += int(
            termination in {"agent_error", "user_error"}
            or termination.endswith(".agent_error")
            or termination.endswith(".user_error")
        )
        records = [*episode["agent_decisions"], *episode["user_generations"]]
        completions_nonempty = completions_nonempty and all(
            bool(str(record.get("completion", "")).strip())
            and (
                record.get("retry_completion") is None
                or bool(str(record["retry_completion"]).strip())
            )
            for record in records
        )
        private_exposures += int(episode["agent_private_scenario_exposures"])
        natural_user_messages += int(episode["natural_user_message_count"])
        agent_logical_calls += int(episode["agent_logical_calls"])
        user_logical_calls += int(episode["user_logical_calls"])
        physical_calls += int(episode["agent_physical_calls"]) + int(
            episode["user_physical_calls"]
        )
        retry_calls += int(episode["serialization_retry_calls"])
        forced_stops += int(
            bool(episode["agent_forced_budget_stop"])
            or bool(episode["user_forced_budget_stop"])
        )
        user_stop_generations += sum(
            any(marker in str(record["completion"]) for marker in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"))
            for record in episode["user_generations"]
        )
        action_counts.update(
            str(prefix["targets"]["next_action"])
            for prefix in dataset_episode["prefixes"]
        )
        assistant_episode_events = [
            event
            for event in episode["combined_tool_events"]
            if event["requestor"] == "assistant"
        ]
        total_combined_events += len(episode["combined_tool_events"])
        if len(assistant_episode_events) >= 2:
            episodes_with_two += 1
        if assistant_episode_events:
            tasks_with_transition.add(str(episode["task_key"]))
        agent_names = {
            schema["function"]["name"]
            for schema in row["agent_interface"]["tool_schemas"]
        }
        user_names = {
            schema["function"]["name"]
            for schema in row["user_private_input"]["tool_schemas"]
        }
        for event in episode["combined_tool_events"]:
            name = str(event["action"]["name"])
            requestor = str(event["requestor"])
            legal_tools = legal_tools and name in (
                agent_names if requestor == "assistant" else user_names
            )
            nondeterministic += int(not bool(event["replica_identical"]))
            if requestor == "assistant":
                assistant_tools[name] += 1
                assistant_errors += int(event["status"] == "error")
                changed = bool(event["state_changed"])
                state_changed += int(changed)
                state_unchanged += int(not changed)
                assistant_changed_by_tool[name] += int(changed)
                if changed:
                    tasks_with_changed.add(str(episode["task_key"]))
                    domains_with_changed.add(str(episode["domain"]))
            else:
                user_tool_events += 1
                user_tools[name] += 1
                user_errors += int(event["status"] == "error")

    total_agent_decisions = sum(action_counts.values())
    tool_decisions = sum(
        count for action, count in action_counts.items() if not action.endswith("::TEXT")
    )
    tool_rate = tool_decisions / max(total_agent_decisions, 1)
    dominant = max(action_counts.values(), default=0) / max(total_agent_decisions, 1)
    target_names = list(protocol["transition_targets"]["names"])
    support = _transition_support(dataset, target_names)
    supported = _supported_targets(
        support,
        minimum_each_class=int(
            protocol["transition_targets"][
                "minimum_each_class_in_training_and_confirmation"
            ]
        ),
    )
    unique_failures = {
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        for row in failures
    }
    metrics = {
        "episodes_complete": len(episodes),
        "runtime_failures": len(unique_failures),
        "agent_private_scenario_exposures": private_exposures,
        "natural_user_messages": natural_user_messages,
        "adjacent_assistant_tool_transitions": int(
            dataset_audit["adjacent_transitions"]
        ),
        "episodes_with_two_or_more_assistant_transitions": episodes_with_two,
        "tasks_with_at_least_one_assistant_transition": len(tasks_with_transition),
        "unique_executed_assistant_tools": len(assistant_tools),
        "agent_tool_decision_rate": tool_rate,
        "dominant_agent_action_fraction": dominant,
        "state_changed_assistant_transitions": state_changed,
        "state_unchanged_assistant_transitions": state_unchanged,
        "tasks_with_state_changed_assistant_transition": len(tasks_with_changed),
        "domains_with_state_changed_assistant_transition": len(domains_with_changed),
        "paired_state_changed_transition_gain_over_parent": int(paired["gain"]),
        "supported_transition_targets": len(supported),
        "communication_error_terminations": communication_errors,
        "forced_budget_stop_episodes": forced_stops,
        "assistant_tool_errors": assistant_errors,
        "user_tool_errors": user_errors,
        "user_tool_events": user_tool_events,
        "agent_logical_calls": agent_logical_calls,
        "user_logical_calls": user_logical_calls,
        "logical_llm_calls": agent_logical_calls + user_logical_calls,
        "serialization_retry_calls": retry_calls,
        "physical_llm_calls": physical_calls,
        "exact_tool_replay_calls": 2 * total_combined_events,
        "nondeterministic_tool_events": nondeterministic,
        "real_external_endpoint_calls": endpoint_calls,
        "user_stop_generations": user_stop_generations,
    }
    budget = protocol["fixed_budget"]
    integrity_checks = {
        "complete_chunk_surface": complete_chunk_surface,
        "all_chunk_audits_passed": all(
            bool(row.get("passed")) for row in chunk_audits
        ),
        "unique_complete_episode_surface": len(episodes)
        == len({row["episode_id"] for row in episodes})
        == int(protocol["source"]["episodes"]),
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "all_agent_and_user_completions_nonempty": completions_nonempty,
        "all_tool_names_in_role_presented_schema": legal_tools,
        "all_complete_tool_sequences_live_and_replicas_identical": nondeterministic
        == 0,
        "zero_real_external_endpoint_calls": endpoint_calls == 0,
        "zero_agent_private_scenario_exposures": private_exposures == 0,
        "task_disjoint_splits": bool(dataset_audit["task_disjoint"]),
        "causal_label_blind_states": bool(dataset_audit["causal_label_blind_states"]),
        "agent_llm_call_budget_respected": agent_logical_calls
        <= int(budget["maximum_agent_llm_calls"]),
        "user_llm_call_budget_respected": user_logical_calls
        <= int(budget["maximum_user_llm_calls"]),
        "logical_llm_call_budget_respected": agent_logical_calls
        + user_logical_calls
        <= int(budget["maximum_logical_llm_calls"]),
        "serialization_retry_budget_respected": retry_calls
        <= int(budget["maximum_serialization_retry_calls"]),
        "physical_llm_call_budget_respected": physical_calls
        <= int(budget["maximum_total_physical_llm_calls"]),
        "exact_tool_replay_budget_respected": 2 * total_combined_events
        <= int(budget["maximum_exact_tool_replay_calls"]),
        "single_shared_model_contract": len(
            {row["shared_model_identity_sha256"] for row in manifest["rows"]}
        )
        == 1,
        "paired_parent_surface_exact": int(paired["paired_episodes"])
        == int(protocol["source"]["episodes"]),
    }
    gate_checks = evaluate_interactive_data_gate(
        metrics=metrics,
        integrity_checks=integrity_checks,
        gate=protocol["data_sufficiency_gate"],
        minimum_supported_targets=int(
            protocol["transition_targets"]["minimum_scored_targets"]
        ),
    )
    passed = all(gate_checks.values())
    decision = (
        "INTERACTION_DATA_GO__AUTHORIZE_FROZEN_METHOD_TEST"
        if passed
        else "INTERACTION_DATA_NO_GO__DO_NOT_SCALE_OR_RUN_METHOD_TEST"
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "passed": passed,
        "metrics": metrics,
        "gate_checks": gate_checks,
        "integrity_checks": integrity_checks,
        "transition_target_support": support,
        "supported_transition_target_names": supported,
        "paired_parent": paired,
        "termination_counts": dict(sorted(termination_counts.items())),
        "agent_action_counts": dict(sorted(action_counts.items())),
        "assistant_tool_counts": dict(sorted(assistant_tools.items())),
        "assistant_state_changed_by_tool": dict(
            sorted(assistant_changed_by_tool.items())
        ),
        "user_tool_counts": dict(sorted(user_tools.items())),
        "dataset_audit": dataset_audit,
        "chunk_output_sha256": {
            str(path.relative_to(args.archive)): file_sha256(path)
            for path in output_paths
        },
        "chunk_audit_sha256": {
            str(path.relative_to(args.archive)): file_sha256(path)
            for path in audit_paths
        },
        "manifest_sha256": file_sha256(args.manifest),
        "protocol_sha256": file_sha256(args.protocol),
        "parent_dataset_sha256": file_sha256(args.parent_dataset),
        "claim_boundary": (
            "A complete data GO authorizes only the frozen predictive-method test; "
            "large-scale collection requires that second gate too."
        ),
    }
    _write_json(args.dataset, dataset)
    _write_json(args.audit, dataset_audit)
    summary["dataset_file_sha256"] = file_sha256(args.dataset)
    summary["dataset_audit_file_sha256"] = file_sha256(args.audit)
    _write_json(args.output, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
