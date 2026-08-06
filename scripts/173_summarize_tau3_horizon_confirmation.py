"""Merge the 96-episode horizon confirmation and apply its frozen gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_horizon import compare_parent_prefixes
from wmagentattack.tau3_horizon_confirmation import (
    episode_reproducibility,
    evaluate_confirmation_gate,
)
from wmagentattack.tau3_interactive import build_interactive_dataset
from wmagentattack.tau3_multistep import file_sha256


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _load_chunks(
    archive: Path, protocol_id: str, expected_chunks: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path], list[Path]]:
    outputs = sorted((archive / "outputs").glob("chunk*.json"))
    audits = sorted((archive / "audits").glob("chunk*.json"))
    if len(outputs) != expected_chunks or len(audits) != expected_chunks:
        raise ValueError("confirmation chunk surface is incomplete")
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    indexes: set[int] = set()
    counts: set[int] = set()
    for path in outputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol_id"] != protocol_id:
            raise ValueError("chunk protocol differs")
        indexes.add(int(payload["chunk_index"]))
        counts.add(int(payload["num_chunks"]))
        episodes.extend(payload["episodes"])
        failures.extend(payload["failures"])
    if counts != {expected_chunks} or indexes != set(range(expected_chunks)):
        raise ValueError("confirmation chunk indexes differ")
    for path in audits:
        failures.extend(
            json.loads(path.read_text(encoding="utf-8")).get(
                "runtime_failures", []
            )
        )
    return episodes, failures, outputs, audits


def _load_episode_directory(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("chunk*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload["episodes"]:
            episode_id = str(episode["episode_id"])
            if episode_id in rows:
                raise ValueError("duplicate archived episode")
            rows[episode_id] = episode
    return rows


def _support(
    dataset: Mapping[str, Any], names: list[str], minimum: int
) -> tuple[dict[str, Any], list[str]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for episode in dataset["episodes"]:
        for transition in episode["transitions"]:
            for name in names:
                label = str(int(float(transition["target"][name]) >= 0.5))
                counts[name][str(episode["split"])][label] += 1
    result = {
        name: {
            split: {
                "negative": int(counts[name][split]["0"]),
                "positive": int(counts[name][split]["1"]),
            }
            for split in ("training", "calibration", "confirmation")
        }
        for name in names
    }
    supported = sorted(
        name
        for name, splits in result.items()
        if all(
            int(splits[split][label]) >= minimum
            for split in ("training", "confirmation")
            for label in ("negative", "positive")
        )
    )
    return result, supported


def _assistant_stats(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    events = [
        event
        for episode in episodes
        for event in episode["combined_tool_events"]
        if event["requestor"] == "assistant"
    ]
    errors = sum(event["status"] == "error" for event in events)
    changed = sum(bool(event["state_changed"]) for event in events)
    return {
        "events": len(events),
        "errors": errors,
        "error_rate": errors / max(len(events), 1),
        "changed": changed,
        "unchanged": len(events) - changed,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    failed = [name for name, value in summary["gate_checks"].items() if not value]
    metrics = summary["metrics"]
    metric_names = (
        "episodes_complete",
        "forced_budget_stop_episodes",
        "parent_forced_budget_stop_episodes",
        "relative_forced_stop_reduction_vs_parent",
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
        "paired_state_changed_transition_gain",
        "supported_transition_targets",
        "assistant_tool_error_rate",
        "parent_assistant_tool_error_rate",
        "assistant_tool_error_rate_increase_over_parent",
        "out_of_pilot_episodes",
        "out_of_pilot_tasks",
        "out_of_pilot_state_changed_assistant_transitions",
        "out_of_pilot_tasks_with_state_changed_assistant_transition",
        "out_of_pilot_domains_with_state_changed_assistant_transition",
        "pilot_overlap_episodes_reproduced",
        "runtime_failures",
    )
    lines = [
        "# tau3 full-horizon confirmation",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for name in metric_names:
        value = metrics[name]
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {name} | {rendered} |")
    lines.extend(
        [
            "",
            "## Gate clauses",
            "",
            *[
                f"- {name}: `{'PASS' if value else 'FAIL'}`"
                for name, value in summary["gate_checks"].items()
            ],
            "",
            "## Counterevidence",
            "",
            f"- Failed clauses: `{json.dumps(failed, sort_keys=True)}`.",
            f"- Paired state changes by domain: `{json.dumps(summary['paired_by_domain'], sort_keys=True)}`.",
            f"- Out-of-pilot state changes by domain: `{json.dumps(summary['out_of_pilot_by_domain'], sort_keys=True)}`.",
            f"- Parent prefix mismatches: `{json.dumps(summary['prefix_mismatches'], sort_keys=True)}`.",
            f"- Pilot-overlap mismatches: `{json.dumps(summary['pilot_overlap_mismatches'], sort_keys=True)}`.",
            "- User-side tools remain exogenous and were not relabeled as assistant outcomes.",
            "- A GO authorizes only the frozen predictive-method comparison, not scale-up.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--parent-output-dir", type=Path, required=True)
    parser.add_argument("--pilot-output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_interactive_outcomes":
        raise ValueError("confirmation protocol was not frozen before outcomes")
    if file_sha256(args.manifest) != protocol["frozen_manifest"]["sha256"]:
        raise ValueError("confirmation manifest hash differs")
    for relative, expected in protocol["implementation_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"confirmation implementation differs: {relative}")

    expected_chunks = int(protocol["execution"]["summary_chunks_expected"])
    episodes, failures, output_paths, audit_paths = _load_chunks(
        args.archive, protocol["protocol_id"], expected_chunks
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {str(row["episode_id"]): row for row in manifest["rows"]}
    episode_rows = {str(row["episode_id"]): row for row in episodes}
    if set(manifest_rows) != set(episode_rows):
        raise ValueError("confirmation outputs differ from the manifest")
    dataset, dataset_audit = build_interactive_dataset(manifest, episodes)
    _write_json(args.dataset, dataset)
    _write_json(args.audit, dataset_audit)

    parent_all = _load_episode_directory(args.parent_output_dir)
    parents = {key: parent_all[key] for key in manifest_rows}
    prefix_details = {
        key: compare_parent_prefixes(episode_rows[key], parents[key])
        for key in sorted(parents)
    }
    prefix_mismatches = [
        key for key, value in prefix_details.items() if not value["all_equal"]
    ]

    pilot_all = _load_episode_directory(args.pilot_output_dir)
    pilot_ids = set(map(str, manifest["pilot_overlap_episode_ids"]))
    if set(pilot_all) != pilot_ids:
        raise ValueError("passed pilot output surface differs from overlap manifest")
    pilot_reproducibility = {
        key: episode_reproducibility(episode_rows[key], pilot_all[key])
        for key in sorted(pilot_ids)
    }
    pilot_overlap_mismatches = [
        key for key, value in pilot_reproducibility.items() if not value
    ]
    holdout_ids = set(map(str, manifest["out_of_pilot_episode_ids"]))

    candidate_stats = _assistant_stats(episodes)
    parent_stats = _assistant_stats(list(parents.values()))
    candidate_forced = sum(
        bool(row["agent_forced_budget_stop"])
        or bool(row["user_forced_budget_stop"])
        for row in episodes
    )
    parent_forced = sum(
        bool(row["agent_forced_budget_stop"])
        or bool(row["user_forced_budget_stop"])
        for row in parents.values()
    )
    relative_reduction = (
        (parent_forced - candidate_forced) / parent_forced
        if parent_forced
        else float(candidate_forced == 0)
    )

    tasks_changed: set[str] = set()
    domains_changed: set[str] = set()
    holdout_tasks: set[str] = set()
    holdout_changed_tasks: set[str] = set()
    holdout_changed_domains: set[str] = set()
    holdout_changed = 0
    paired_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    holdout_by_domain: Counter[str] = Counter()
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        parent = parents[episode_id]
        new = sum(
            event["requestor"] == "assistant" and bool(event["state_changed"])
            for event in episode["combined_tool_events"]
        )
        old = sum(
            event["requestor"] == "assistant" and bool(event["state_changed"])
            for event in parent["combined_tool_events"]
        )
        domain = str(episode["domain"])
        task = str(episode["task_key"])
        paired_by_domain[domain].update(candidate=new, parent=old)
        if new:
            tasks_changed.add(task)
            domains_changed.add(domain)
        if episode_id in holdout_ids:
            holdout_tasks.add(task)
            holdout_changed += new
            holdout_by_domain[domain] += new
            if new:
                holdout_changed_tasks.add(task)
                holdout_changed_domains.add(domain)

    names = list(protocol["held_fixed"]["transition_targets"])
    gate = protocol["confirmation_gate"]
    support, supported = _support(
        dataset,
        names,
        int(gate["minimum_each_transition_class_in_training_and_confirmation"]),
    )
    dataset_rows = {
        str(row["episode_id"]): row for row in dataset["episodes"]
    }
    action_counts: Counter[str] = Counter()
    assistant_tools: Counter[str] = Counter()
    tasks_with_transition: set[str] = set()
    episodes_with_two = 0
    records = []
    legal = True
    nondeterministic = 0
    private_exposures = 0
    endpoint_calls = 0
    combined_events = 0
    communication_errors = 0
    natural_user_messages = 0
    for episode in episodes:
        row = manifest_rows[str(episode["episode_id"])]
        data_episode = dataset_rows[str(episode["episode_id"])]
        records.extend([*episode["agent_decisions"], *episode["user_generations"]])
        action_counts.update(
            str(prefix["targets"]["next_action"])
            for prefix in data_episode["prefixes"]
        )
        assistant_episode_events = [
            event
            for event in episode["combined_tool_events"]
            if event["requestor"] == "assistant"
        ]
        if len(assistant_episode_events) >= 2:
            episodes_with_two += 1
        if assistant_episode_events:
            tasks_with_transition.add(str(episode["task_key"]))
        agent_tools = {
            item["function"]["name"]
            for item in row["agent_interface"]["tool_schemas"]
        }
        user_tools = {
            item["function"]["name"]
            for item in row["user_private_input"]["tool_schemas"]
        }
        for event in episode["combined_tool_events"]:
            requestor = str(event["requestor"])
            name = str(event["action"]["name"])
            legal = legal and name in (
                agent_tools if requestor == "assistant" else user_tools
            )
            nondeterministic += int(not bool(event["replica_identical"]))
            if requestor == "assistant":
                assistant_tools[name] += 1
        termination = str(episode["termination"]).lower()
        communication_errors += int(
            termination in {"agent_error", "user_error"}
            or termination.endswith(".agent_error")
            or termination.endswith(".user_error")
        )
        combined_events += len(episode["combined_tool_events"])
        private_exposures += int(episode["agent_private_scenario_exposures"])
        endpoint_calls += int(episode.get("real_external_endpoint_calls", 0))
        natural_user_messages += int(episode["natural_user_message_count"])

    total_agent_decisions = sum(action_counts.values())
    tool_decisions = sum(
        count
        for action, count in action_counts.items()
        if not action.endswith("::TEXT")
    )
    tool_rate = tool_decisions / max(total_agent_decisions, 1)
    dominant = max(action_counts.values(), default=0) / max(
        total_agent_decisions, 1
    )
    unique_failures = {
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        for row in failures
    }
    metrics = {
        "episodes_complete": len(episodes),
        "runtime_failures": len(unique_failures),
        "agent_private_scenario_exposures": private_exposures,
        "real_external_endpoint_calls": endpoint_calls,
        "communication_error_terminations": communication_errors,
        "forced_budget_stop_episodes": candidate_forced,
        "parent_forced_budget_stop_episodes": parent_forced,
        "relative_forced_stop_reduction_vs_parent": relative_reduction,
        "natural_user_messages": natural_user_messages,
        "adjacent_assistant_tool_transitions": int(
            dataset_audit["adjacent_transitions"]
        ),
        "episodes_with_two_or_more_assistant_transitions": episodes_with_two,
        "tasks_with_at_least_one_assistant_transition": len(
            tasks_with_transition
        ),
        "unique_executed_assistant_tools": len(assistant_tools),
        "agent_tool_decision_rate": tool_rate,
        "dominant_agent_action_fraction": dominant,
        "state_changed_assistant_transitions": candidate_stats["changed"],
        "state_unchanged_assistant_transitions": candidate_stats["unchanged"],
        "tasks_with_state_changed_assistant_transition": len(tasks_changed),
        "domains_with_state_changed_assistant_transition": len(domains_changed),
        "paired_state_changed_transition_gain": candidate_stats["changed"]
        - parent_stats["changed"],
        "supported_transition_targets": len(supported),
        "assistant_tool_error_rate": candidate_stats["error_rate"],
        "parent_assistant_tool_error_rate": parent_stats["error_rate"],
        "assistant_tool_error_rate_increase_over_parent": candidate_stats[
            "error_rate"
        ]
        - parent_stats["error_rate"],
        "assistant_tool_errors": candidate_stats["errors"],
        "parent_assistant_tool_errors": parent_stats["errors"],
        "out_of_pilot_episodes": len(holdout_ids),
        "out_of_pilot_tasks": len(holdout_tasks),
        "out_of_pilot_state_changed_assistant_transitions": holdout_changed,
        "out_of_pilot_tasks_with_state_changed_assistant_transition": len(
            holdout_changed_tasks
        ),
        "out_of_pilot_domains_with_state_changed_assistant_transition": len(
            holdout_changed_domains
        ),
        "pilot_overlap_episodes_reproduced": len(pilot_ids)
        - len(pilot_overlap_mismatches),
        "agent_logical_calls": sum(row["agent_logical_calls"] for row in episodes),
        "user_logical_calls": sum(row["user_logical_calls"] for row in episodes),
        "logical_llm_calls": sum(
            row["agent_logical_calls"] + row["user_logical_calls"]
            for row in episodes
        ),
        "physical_llm_calls": sum(
            row["agent_physical_calls"] + row["user_physical_calls"]
            for row in episodes
        ),
        "serialization_retry_calls": sum(
            row["serialization_retry_calls"] for row in episodes
        ),
        "exact_replay_calls": 2 * combined_events,
    }
    budget = protocol["fixed_budget"]
    manifest_audit = json.loads(
        (args.archive / "manifest_audit.json").read_text(encoding="utf-8")
    )
    chunk_audits = [
        json.loads(path.read_text(encoding="utf-8")) for path in audit_paths
    ]
    completions_nonempty = all(
        str(record.get("completion", "")).strip()
        and (
            record.get("retry_completion") is None
            or str(record["retry_completion"]).strip()
        )
        for record in records
    )
    label_blind = bool(manifest_audit["passed"]) and not manifest_audit[
        "forbidden_outcome_inputs_read"
    ]
    integrity = {
        "complete_chunk_surface": len(output_paths)
        == len(audit_paths)
        == expected_chunks,
        "all_chunk_audits_passed": all(row["passed"] for row in chunk_audits),
        "all_completions_nonempty": completions_nonempty,
        "all_tool_names_legal": legal,
        "all_exact_replays_identical": nondeterministic == 0,
        "all_parent_prefixes_equivalent": not prefix_mismatches,
        "all_pilot_overlap_episodes_reproduced": not pilot_overlap_mismatches,
        "label_blind_panel_selection": label_blind,
        "label_blind_full_surface": label_blind,
        "task_disjoint_splits": bool(dataset_audit["task_disjoint"]),
        "causal_label_blind_states": bool(dataset_audit["causal_label_blind_states"]),
        "zero_runtime_failures": not unique_failures,
        "zero_private_scenario_exposures": private_exposures == 0,
        "zero_real_external_endpoints": endpoint_calls == 0,
        "logical_call_budget_respected": metrics["logical_llm_calls"]
        <= int(budget["maximum_logical_llm_calls"]),
        "physical_call_budget_respected": metrics["physical_llm_calls"]
        <= int(budget["maximum_physical_llm_calls"]),
        "retry_budget_respected": metrics["serialization_retry_calls"]
        <= int(budget["maximum_serialization_retry_calls"]),
        "exact_replay_budget_respected": metrics["exact_replay_calls"]
        <= int(budget["maximum_exact_replay_calls"]),
    }
    checks = evaluate_confirmation_gate(metrics, integrity, gate)
    passed = all(checks.values())
    decision = protocol["decisions"]["go" if passed else "no_go"]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "passed": passed,
        "metrics": metrics,
        "gate_checks": checks,
        "integrity_checks": integrity,
        "transition_target_support": support,
        "supported_transition_target_names": supported,
        "paired_by_domain": {
            domain: {
                "candidate": int(values["candidate"]),
                "parent": int(values["parent"]),
                "gain": int(values["candidate"] - values["parent"]),
            }
            for domain, values in sorted(paired_by_domain.items())
        },
        "out_of_pilot_by_domain": dict(sorted(holdout_by_domain.items())),
        "prefix_equivalence": prefix_details,
        "prefix_mismatches": prefix_mismatches,
        "pilot_overlap_reproducibility": pilot_reproducibility,
        "pilot_overlap_mismatches": pilot_overlap_mismatches,
        "dataset_sha256": file_sha256(args.dataset),
        "dataset_audit_sha256": file_sha256(args.audit),
        "manifest_sha256": file_sha256(args.manifest),
        "protocol_sha256": file_sha256(args.protocol),
        "chunk_output_sha256": {
            str(path.relative_to(args.archive)): file_sha256(path)
            for path in output_paths
        },
        "chunk_audit_sha256": {
            str(path.relative_to(args.archive)): file_sha256(path)
            for path in audit_paths
        },
        "claim_boundary": "A confirmation GO authorizes only the frozen predictive-method comparison.",
    }
    _write_json(args.output, summary)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
