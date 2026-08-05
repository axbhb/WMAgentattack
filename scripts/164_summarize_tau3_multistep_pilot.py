"""Merge frozen tau3 multi-step chunks and apply the preregistered data gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_multistep import (
    action_candidate_id,
    build_dataset,
    file_sha256,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _fmt(value: float | int) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _transition_support(
    dataset: Mapping[str, Any], target_names: Sequence[str]
) -> dict[str, Any]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for episode in dataset["episodes"]:
        split = str(episode["split"])
        for transition in episode["transitions"]:
            for name in target_names:
                value = int(float(transition["target"][name]) >= 0.5)
                counts[name][split][str(value)] += 1
    output: dict[str, Any] = {}
    for name in target_names:
        output[name] = {
            split: {
                "negative": int(counts[name][split]["0"]),
                "positive": int(counts[name][split]["1"]),
            }
            for split in ("training", "calibration", "confirmation")
        }
    return output


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


def _markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# tau3 multi-step scale-readiness pilot",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Frozen data result",
        "",
        "| Metric | Result | Frozen threshold | Pass |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "episodes_complete": "Complete episodes",
        "adjacent_transitions": "Executed adjacent transitions",
        "episodes_with_two_or_more_transitions": "Episodes with >=2 transitions",
        "tasks_with_at_least_one_transition": "Tasks with >=1 transition",
        "unique_executed_tools": "Unique executed tools",
        "tool_decision_rate": "Tool-decision rate",
        "dominant_action_fraction": "Dominant-action fraction",
        "state_changed_transitions": "State-changing transitions",
        "state_unchanged_transitions": "State-unchanged transitions",
        "supported_transition_targets": "Supported transition targets",
    }
    for name, threshold in summary["display_thresholds"].items():
        lines.append(
            f"| {labels[name]} | {_fmt(metrics[name])} | {threshold} | "
            f"{'PASS' if summary['gate_checks'][name] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
        ]
    )
    for name, passed in summary["integrity_checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Transition-label support",
            "",
            "| Target | Train neg/pos | Calibration neg/pos | Confirmation neg/pos | Authorized |",
            "|---|---:|---:|---:|---|",
        ]
    )
    supported = set(summary["supported_transition_target_names"])
    for name, splits in summary["transition_target_support"].items():
        lines.append(
            f"| {name} | {splits['training']['negative']}/{splits['training']['positive']} | "
            f"{splits['calibration']['negative']}/{splits['calibration']['positive']} | "
            f"{splits['confirmation']['negative']}/{splits['confirmation']['positive']} | "
            f"{'YES' if name in supported else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Counterevidence and authorization boundary",
            "",
            f"- Termination distribution: `{json.dumps(summary['termination_counts'], sort_keys=True)}`.",
            f"- Runtime failures: {summary['metrics']['runtime_failures']}; non-deterministic executions: {summary['metrics']['nondeterministic_transitions']}.",
            "- The reference trajectories were used only to select structurally multi-step strata; reference actions and outcomes were not victim-model inputs.",
            "- Passing this gate authorizes only the frozen task-disjoint predictive-method comparison. It does not authorize attack generation, Dreamer training, or unrestricted collection.",
            "- Large-scale collection is authorized only if the subsequent observation-aware Semantic Markov v4 method gate also passes without changing its thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_victim_outcomes":
        raise ValueError("protocol was not frozen before victim outcomes")
    if file_sha256(args.manifest) != protocol["frozen_manifest"]["sha256"]:
        raise ValueError("manifest hash differs from frozen protocol")
    for relative_path, expected in protocol["implementation_sha256"].items():
        if file_sha256(ROOT / relative_path) != expected:
            raise ValueError(f"implementation hash differs: {relative_path}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_paths = sorted((args.archive / "outputs").glob("chunk*.json"))
    audit_paths = sorted((args.archive / "audits").glob("chunk*.json"))
    if not output_paths or not audit_paths:
        raise FileNotFoundError("pilot chunks or chunk audits are missing")
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    chunk_indexes: set[int] = set()
    declared_chunk_counts: set[int] = set()
    endpoint_calls = 0
    for path in output_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol_id"] != protocol["protocol_id"]:
            raise ValueError(f"chunk protocol mismatch: {path}")
        chunk_indexes.add(int(payload["chunk_index"]))
        declared_chunk_counts.add(int(payload["num_chunks"]))
        episodes.extend(payload["episodes"])
        failures.extend(payload["failures"])
        endpoint_calls += int(payload.get("real_external_endpoint_calls", 0))
    if len(declared_chunk_counts) != 1:
        raise ValueError("chunk files disagree on num_chunks")
    num_chunks = next(iter(declared_chunk_counts))
    complete_chunk_surface = chunk_indexes == set(range(num_chunks))
    if len(audit_paths) != num_chunks:
        complete_chunk_surface = False
    chunk_audits = [
        json.loads(path.read_text(encoding="utf-8")) for path in audit_paths
    ]
    audit_indexes = {int(row["chunk_index"]) for row in chunk_audits}
    complete_chunk_surface = complete_chunk_surface and audit_indexes == set(
        range(num_chunks)
    )
    chunk_audit_passed = all(bool(row.get("passed")) for row in chunk_audits)
    for row in chunk_audits:
        failures.extend(row.get("runtime_failures", ()))
        endpoint_calls += int(row.get("real_external_endpoint_calls", 0))

    dataset, dataset_audit = build_dataset(manifest, episodes)
    manifest_rows = {row["episode_id"]: row for row in manifest["rows"]}
    prefix_targets: list[str] = []
    legal_tools = True
    completions_nonempty = True
    nondeterministic = 0
    executed_tools = Counter()
    state_changed = 0
    state_unchanged = 0
    tasks_with_transition: set[str] = set()
    episodes_with_two = 0
    termination = Counter()
    for episode in episodes:
        manifest_row = manifest_rows[episode["episode_id"]]
        termination[str(episode["termination"])] += 1
        completions_nonempty = completions_nonempty and all(
            bool(str(query.get("completion", "")).strip())
            for query in episode["queries"]
        )
        prefix_targets.extend(
            str(prefix["targets"]["next_action"])
            for prefix in episode["prefixes"]
        )
        if len(episode["transitions"]) >= 2:
            episodes_with_two += 1
        if episode["transitions"]:
            tasks_with_transition.add(str(episode["task_key"]))
        legal_names = {
            schema["function"]["name"]
            for schema in manifest_row["model_input"]["tool_schemas"]
        }
        for event in episode["transitions"]:
            executed_tools[str(event["action"]["name"])] += 1
            legal_tools = legal_tools and str(event["action"]["name"]) in legal_names
            nondeterministic += int(not bool(event.get("replica_identical")))
            state_changed += int(bool(event["state_changed"]))
            state_unchanged += int(not bool(event["state_changed"]))
        for prefix, query in zip(episode["prefixes"], episode["queries"]):
            expected_target = action_candidate_id(
                episode["domain"], manifest_row["model_input"], query["decision"]
            )
            legal_tools = legal_tools and expected_target == prefix["targets"]["next_action"]

    target_counts = Counter(prefix_targets)
    text_decisions = sum(target.endswith("::TEXT") for target in prefix_targets)
    tool_decisions = len(prefix_targets) - text_decisions
    tool_rate = tool_decisions / len(prefix_targets) if prefix_targets else 0.0
    dominant = max(target_counts.values(), default=0) / max(len(prefix_targets), 1)
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
    metrics = {
        "episodes_complete": len(episodes),
        "runtime_failures": len({json.dumps(row, sort_keys=True) for row in failures}),
        "adjacent_transitions": int(dataset_audit["adjacent_transitions"]),
        "episodes_with_two_or_more_transitions": episodes_with_two,
        "tasks_with_at_least_one_transition": len(tasks_with_transition),
        "unique_executed_tools": len(executed_tools),
        "tool_decision_rate": tool_rate,
        "dominant_action_fraction": dominant,
        "state_changed_transitions": state_changed,
        "state_unchanged_transitions": state_unchanged,
        "supported_transition_targets": len(supported),
        "nondeterministic_transitions": nondeterministic,
    }
    gate = protocol["data_sufficiency_gate"]
    integrity_checks = {
        "complete_chunk_surface": complete_chunk_surface,
        "all_chunk_audits_passed": chunk_audit_passed,
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "all_completions_nonempty": completions_nonempty,
        "all_tool_names_in_presented_schema": legal_tools,
        "all_executed_actions_exact_replica_identical": nondeterministic == 0,
        "zero_real_external_endpoint_calls": endpoint_calls == 0,
        "task_disjoint_splits": bool(dataset_audit["task_disjoint"]),
        "causal_label_blind_states": bool(
            dataset_audit["causal_label_blind_states"]
        ),
    }
    gate_checks = {
        "episodes_complete": metrics["episodes_complete"]
        == int(gate["expected_episodes_complete"]),
        "adjacent_transitions": metrics["adjacent_transitions"]
        >= int(gate["minimum_adjacent_transitions"]),
        "episodes_with_two_or_more_transitions": metrics[
            "episodes_with_two_or_more_transitions"
        ]
        >= int(gate["minimum_episodes_with_two_or_more_transitions"]),
        "tasks_with_at_least_one_transition": metrics[
            "tasks_with_at_least_one_transition"
        ]
        >= int(gate["minimum_tasks_with_at_least_one_transition"]),
        "unique_executed_tools": metrics["unique_executed_tools"]
        >= int(gate["minimum_unique_executed_tools"]),
        "tool_decision_rate": float(gate["minimum_tool_decision_rate"])
        <= tool_rate
        <= float(gate["maximum_tool_decision_rate"]),
        "dominant_action_fraction": dominant
        <= float(gate["maximum_dominant_action_fraction"]),
        "state_changed_transitions": state_changed
        >= int(gate["minimum_state_changed_transitions"]),
        "state_unchanged_transitions": state_unchanged
        >= int(gate["minimum_state_unchanged_transitions"]),
        "supported_transition_targets": len(supported)
        >= int(protocol["transition_targets"]["minimum_scored_targets"]),
    }
    passed = all(integrity_checks.values()) and all(gate_checks.values())
    decision = (
        "DATA_FORM_GO__AUTHORIZE_FROZEN_METHOD_TEST"
        if passed
        else "DATA_FORM_NO_GO__DO_NOT_SCALE_OR_RUN_METHOD_TEST"
    )
    display_thresholds = {
        "episodes_complete": f"== {gate['expected_episodes_complete']}",
        "adjacent_transitions": f">= {gate['minimum_adjacent_transitions']}",
        "episodes_with_two_or_more_transitions": f">= {gate['minimum_episodes_with_two_or_more_transitions']}",
        "tasks_with_at_least_one_transition": f">= {gate['minimum_tasks_with_at_least_one_transition']}",
        "unique_executed_tools": f">= {gate['minimum_unique_executed_tools']}",
        "tool_decision_rate": f"[{gate['minimum_tool_decision_rate']}, {gate['maximum_tool_decision_rate']}]",
        "dominant_action_fraction": f"<= {gate['maximum_dominant_action_fraction']}",
        "state_changed_transitions": f">= {gate['minimum_state_changed_transitions']}",
        "state_unchanged_transitions": f">= {gate['minimum_state_unchanged_transitions']}",
        "supported_transition_targets": f">= {protocol['transition_targets']['minimum_scored_targets']}",
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "passed": passed,
        "metrics": metrics,
        "gate_checks": gate_checks,
        "integrity_checks": integrity_checks,
        "display_thresholds": display_thresholds,
        "executed_tool_counts": dict(sorted(executed_tools.items())),
        "target_action_counts": dict(sorted(target_counts.items())),
        "termination_counts": dict(sorted(termination.items())),
        "transition_target_support": support,
        "supported_transition_target_names": supported,
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
        "real_external_endpoint_calls": endpoint_calls,
        "claim_boundary": (
            "A GO authorizes only the frozen task-disjoint predictive-method test. "
            "Large-scale collection requires the method gate too."
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
