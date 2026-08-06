"""Merge the horizon-only pilot and apply its frozen paired gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_horizon import (
    compare_parent_prefixes,
    evaluate_horizon_gate,
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


def _load_chunk_episodes(
    archive: Path, protocol_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path], list[Path]]:
    outputs = sorted((archive / "outputs").glob("chunk*.json"))
    audits = sorted((archive / "audits").glob("chunk*.json"))
    if not outputs or not audits:
        raise FileNotFoundError("horizon chunks are missing")
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
    if len(counts) != 1 or indexes != set(range(next(iter(counts)))):
        raise ValueError("chunk surface is incomplete")
    if len(audits) != next(iter(counts)):
        raise ValueError("chunk audit surface is incomplete")
    for path in audits:
        failures.extend(
            json.loads(path.read_text(encoding="utf-8")).get(
                "runtime_failures", []
            )
        )
    return episodes, failures, outputs, audits


def _load_parent_episodes(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("chunk*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload["episodes"]:
            episode_id = str(episode["episode_id"])
            if episode_id in rows:
                raise ValueError("duplicate parent episode")
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
    lines = [
        "# tau3 bounded-horizon pilot",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for name in (
        "episodes_complete",
        "forced_budget_stop_episodes",
        "parent_forced_budget_stop_episodes",
        "relative_forced_stop_reduction_vs_parent",
        "adjacent_assistant_tool_transitions",
        "state_changed_assistant_transitions",
        "state_unchanged_assistant_transitions",
        "tasks_with_state_changed_assistant_transition",
        "domains_with_state_changed_assistant_transition",
        "paired_state_changed_transition_gain",
        "supported_transition_targets",
        "assistant_tool_error_rate",
        "parent_assistant_tool_error_rate",
        "assistant_tool_error_rate_increase_over_parent",
        "runtime_failures",
    ):
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
            f"- Paired changes by domain: `{json.dumps(summary['paired_by_domain'], sort_keys=True)}`.",
            f"- Parent prefix mismatches: `{json.dumps(summary['prefix_mismatches'], sort_keys=True)}`.",
            "- User-side tools remain exogenous context and were not relabeled as assistant outcomes.",
            "- A GO authorizes only a separately frozen 96-episode confirmation.",
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_interactive_outcomes":
        raise ValueError("horizon protocol was not frozen before outcomes")
    if file_sha256(args.manifest) != protocol["frozen_manifest"]["sha256"]:
        raise ValueError("horizon manifest hash differs")
    for relative, expected in protocol["implementation_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"horizon implementation differs: {relative}")

    episodes, failures, output_paths, audit_paths = _load_chunk_episodes(
        args.archive, protocol["protocol_id"]
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {str(row["episode_id"]): row for row in manifest["rows"]}
    if set(manifest_rows) != {str(row["episode_id"]) for row in episodes}:
        raise ValueError("candidate episode surface differs from the manifest")
    dataset, dataset_audit = build_interactive_dataset(manifest, episodes)
    _write_json(args.dataset, dataset)
    _write_json(args.audit, dataset_audit)

    parent_all = _load_parent_episodes(args.parent_output_dir)
    parents = {key: parent_all[key] for key in manifest_rows}
    prefix_details = {
        key: compare_parent_prefixes(
            next(row for row in episodes if str(row["episode_id"]) == key),
            parents[key],
        )
        for key in sorted(parents)
    }
    prefix_mismatches = [
        key for key, value in prefix_details.items() if not value["all_equal"]
    ]

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
    paired_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        parent = parents[str(episode["episode_id"])]
        new = sum(
            event["requestor"] == "assistant" and bool(event["state_changed"])
            for event in episode["combined_tool_events"]
        )
        old = sum(
            event["requestor"] == "assistant" and bool(event["state_changed"])
            for event in parent["combined_tool_events"]
        )
        domain = str(episode["domain"])
        paired_by_domain[domain].update(candidate=new, parent=old)
        if new:
            tasks_changed.add(str(episode["task_key"]))
            domains_changed.add(domain)

    names = list(protocol["held_fixed"]["transition_targets"])
    support, supported = _support(
        dataset,
        names,
        int(protocol["pilot_gate"]["minimum_each_transition_class_in_training_and_confirmation"]),
    )
    records = [
        record
        for episode in episodes
        for record in [*episode["agent_decisions"], *episode["user_generations"]]
    ]
    legal = True
    nondeterministic = 0
    private_exposures = 0
    endpoint_calls = 0
    combined_events = 0
    for episode in episodes:
        row = manifest_rows[str(episode["episode_id"])]
        agent_tools = {
            item["function"]["name"]
            for item in row["agent_interface"]["tool_schemas"]
        }
        user_tools = {
            item["function"]["name"]
            for item in row["user_private_input"]["tool_schemas"]
        }
        for event in episode["combined_tool_events"]:
            legal = legal and event["action"]["name"] in (
                agent_tools if event["requestor"] == "assistant" else user_tools
            )
            nondeterministic += int(not bool(event["replica_identical"]))
        combined_events += len(episode["combined_tool_events"])
        private_exposures += int(episode["agent_private_scenario_exposures"])
        endpoint_calls += int(episode.get("real_external_endpoint_calls", 0))
    unique_failures = {
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        for row in failures
    }
    metrics = {
        "episodes_complete": len(episodes),
        "runtime_failures": len(unique_failures),
        "agent_private_scenario_exposures": private_exposures,
        "real_external_endpoint_calls": endpoint_calls,
        "forced_budget_stop_episodes": candidate_forced,
        "parent_forced_budget_stop_episodes": parent_forced,
        "relative_forced_stop_reduction_vs_parent": relative_reduction,
        "adjacent_assistant_tool_transitions": int(
            dataset_audit["adjacent_transitions"]
        ),
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
    integrity = {
        "complete_chunk_surface": len(output_paths) == len(audit_paths) == 2,
        "all_chunk_audits_passed": all(row["passed"] for row in chunk_audits),
        "all_completions_nonempty": all(
            str(record.get("completion", "")).strip() for record in records
        ),
        "all_tool_names_legal": legal,
        "all_exact_replays_identical": nondeterministic == 0,
        "all_parent_prefixes_equivalent": not prefix_mismatches,
        "label_blind_panel_selection": bool(manifest_audit["passed"])
        and not manifest_audit["forbidden_outcome_inputs_read"],
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
    checks = evaluate_horizon_gate(metrics, integrity, protocol["pilot_gate"])
    passed = all(checks.values())
    decision = (
        "HORIZON_PILOT_GO__AUTHORIZE_FULL_96_CONFIRMATION"
        if passed
        else "HORIZON_PILOT_NO_GO__DO_NOT_RUN_FULL_CONFIRMATION"
    )
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
        "prefix_equivalence": prefix_details,
        "prefix_mismatches": prefix_mismatches,
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
        "claim_boundary": "A pilot GO authorizes only a frozen 96-episode horizon confirmation.",
    }
    _write_json(args.output, summary)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
