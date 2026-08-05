"""Causal contracts for the interaction-faithful tau3 data repair."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .semantic_state_v3 import (
    find_semantic_state_v3_leakage,
    semantic_state_v3_payload,
)
from .tau3_multistep import (
    DATASET_SCHEMA_VERSION,
    append_ledger_event,
    candidate_catalog_for_model_input,
    source_prefix,
    stable_hash,
    transition_target,
)


MANIFEST_SCHEMA_VERSION = "wmagentattack.tau3_interactive_manifest.v1"
DATASET_SCHEMA_VERSION_INTERACTIVE = DATASET_SCHEMA_VERSION + ".interactive"


def role_seed(episode_seed: int, role: str, generation_index: int) -> int:
    if role == "agent":
        return int(episode_seed) * 1009 + int(generation_index)
    if role == "user":
        return int(episode_seed) * 2003 + int(generation_index)
    raise ValueError(f"unknown interaction role: {role}")


def runtime_agent_model_input(
    agent_interface: Mapping[str, Any], natural_user_messages: Sequence[str]
) -> dict[str, Any]:
    messages = [
        str(message).strip()
        for message in natural_user_messages
        if str(message).strip()
    ]
    if not messages:
        raise ValueError("agent prefix has no causal natural user message")
    if set(agent_interface) != {"tool_schemas", "policy"}:
        raise ValueError("agent interface contains non-whitelisted fields")
    return {
        "trusted_goal": "\n".join(
            f"User turn {index + 1}: {message}"
            for index, message in enumerate(messages)
        ),
        "tool_schemas": json.loads(
            json.dumps(agent_interface["tool_schemas"], ensure_ascii=False)
        ),
        "policy": str(agent_interface["policy"]),
    }


def normalized_tool_event(
    raw: Mapping[str, Any], *, assistant_index: int | None = None
) -> dict[str, Any]:
    output = {
        "index": int(
            raw["combined_index"] if assistant_index is None else assistant_index
        ),
        "combined_index": int(raw["combined_index"]),
        "requestor": str(raw["requestor"]),
        "action": {
            "name": str(raw["action"]["name"]),
            "arguments": json.loads(
                json.dumps(raw["action"]["arguments"], ensure_ascii=False, default=str)
            ),
        },
        "status": str(raw["status"]),
        "error": raw.get("error"),
        "output": raw.get("output"),
        "state_before_sha256": str(raw["state_before_sha256"]),
        "state_after_sha256": str(raw["state_after_sha256"]),
        "state_changed": bool(raw["state_changed"]),
        "replica_identical": bool(raw.get("replica_identical", False)),
    }
    return output


def reconstruct_agent_surface(
    *,
    episode_id: str,
    domain: str,
    agent_interface: Mapping[str, Any],
    agent_decisions: Sequence[Mapping[str, Any]],
    combined_tool_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assistant_events = [
        normalized_tool_event(raw, assistant_index=index)
        for index, raw in enumerate(
            event
            for event in combined_tool_events
            if event["requestor"] == "assistant"
        )
    ]
    prefixes = []
    transitions = []
    prior_events: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {
        "records": [],
        "conflicts": [],
        "execution_receipts": [],
    }
    assistant_event_index = 0
    private_exposures = 0
    for expected_index, decision_record in enumerate(agent_decisions):
        if int(decision_record["generation_index"]) != expected_index:
            raise ValueError("agent generation indices are not consecutive")
        provenance = set(map(str, decision_record["agent_input_provenance"]))
        if "user_private_scenario" in provenance:
            private_exposures += 1
        if not provenance <= {
            "official_agent_instruction",
            "domain_policy",
            "agent_tool_schemas",
            "natural_user_messages",
            "own_text_messages",
            "own_tool_calls_and_receipts",
        }:
            raise ValueError("agent prompt provenance contains an unknown source")
        model_input = runtime_agent_model_input(
            agent_interface, decision_record["natural_user_messages"]
        )
        decision = decision_record["decision"]
        prefix = source_prefix(
            episode_id=episode_id,
            domain=domain,
            model_input=model_input,
            prefix_index=expected_index,
            prior_events=prior_events,
            ledger=ledger,
            decision=decision,
        )
        prefix["agent_visible_dialogue_sha256"] = str(
            decision_record["agent_visible_dialogue_sha256"]
        )
        prefixes.append(prefix)
        if decision["kind"] != "tool_call":
            continue
        if assistant_event_index >= len(assistant_events):
            raise ValueError("agent tool decision has no executed event")
        event = assistant_events[assistant_event_index]
        if event["action"] != {
            "name": decision["name"],
            "arguments": decision["arguments"],
        }:
            raise ValueError("agent tool decision differs from executed event")
        if not event["replica_identical"]:
            raise ValueError("interactive tool sequence replicas differ")
        event["decision_prefix_index"] = expected_index
        prior_events.append(event)
        ledger = append_ledger_event(
            ledger,
            episode_id=episode_id,
            domain=domain,
            event=event,
        )
        transitions.append(event)
        assistant_event_index += 1
    if assistant_event_index != len(assistant_events):
        raise ValueError("executed assistant events are not represented by prefixes")
    audit = {
        "agent_private_scenario_exposures": private_exposures,
        "assistant_tool_events": len(assistant_events),
        "user_tool_events": sum(
            event["requestor"] == "user" for event in combined_tool_events
        ),
    }
    return prefixes, transitions, audit


def build_interactive_dataset(
    manifest: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_rows = {row["episode_id"]: row for row in manifest["rows"]}
    if set(manifest_rows) != {row["episode_id"] for row in episodes}:
        raise ValueError("interactive outputs do not exactly match the manifest")
    catalog: dict[str, dict[str, Any]] = {}
    output_episodes = []
    leakage = []
    split_tasks: dict[str, set[str]] = defaultdict(set)
    transition_positive = Counter()
    private_exposures = 0
    user_tool_events = 0
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        manifest_row = manifest_rows[episode["episode_id"]]
        catalog_input = runtime_agent_model_input(
            manifest_row["agent_interface"], ["runtime user message placeholder"]
        )
        for key, descriptor in candidate_catalog_for_model_input(
            episode["domain"], catalog_input
        ).items():
            previous = catalog.setdefault(key, descriptor)
            if stable_hash(previous) != stable_hash(descriptor):
                raise ValueError("interactive candidate schema collision")
        prefixes, assistant_events, reconstruction_audit = reconstruct_agent_surface(
            episode_id=episode["episode_id"],
            domain=episode["domain"],
            agent_interface=manifest_row["agent_interface"],
            agent_decisions=episode["agent_decisions"],
            combined_tool_events=episode["combined_tool_events"],
        )
        private_exposures += int(
            reconstruction_audit["agent_private_scenario_exposures"]
        )
        user_tool_events += int(reconstruction_audit["user_tool_events"])
        semantic_prefixes = []
        for prefix in prefixes:
            state = semantic_state_v3_payload(prefix["features"])
            findings = find_semantic_state_v3_leakage(state)
            if findings:
                leakage.append(
                    {
                        "episode_id": episode["episode_id"],
                        "prefix_index": prefix["prefix_index"],
                        "findings": list(findings),
                    }
                )
            semantic_prefixes.append(
                {
                    "prefix_index": prefix["prefix_index"],
                    "features": {"semantic_state_v3": state},
                    "targets": prefix["targets"],
                }
            )
        transition_rows = []
        for index, event in enumerate(assistant_events):
            prefix_index = int(event["decision_prefix_index"])
            target = transition_target(
                trusted_goal=prefixes[prefix_index]["features"]["trusted_goal"],
                prior_events=assistant_events[:index],
                event=event,
            )
            transition_positive.update(
                name for name, value in target.items() if value == 1.0
            )
            transition_rows.append(
                {
                    "transition_index": index,
                    "prefix_index": prefix_index,
                    "action": prefixes[prefix_index]["targets"]["next_action"],
                    "target": target,
                }
            )
        split_tasks[episode["split"]].add(episode["task_key"])
        output_episodes.append(
            {
                "episode_id": episode["episode_id"],
                "parent_episode_id": episode["parent_episode_id"],
                "task_id": episode["task_key"],
                "suite": f"tau3:{episode['domain']}",
                "domain": episode["domain"],
                "split": episode["split"],
                "track": f"tau3:{episode['domain']}",
                "run_seed": episode["llm_seed"],
                "prefixes": prefixes,
                "semantic_prefixes": semantic_prefixes,
                "transitions": transition_rows,
                "termination": episode["termination"],
                "natural_user_message_count": episode[
                    "natural_user_message_count"
                ],
                "user_tool_event_count": reconstruction_audit["user_tool_events"],
            }
        )
    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION_INTERACTIVE,
        "protocol_id": manifest["protocol_id"],
        "candidate_catalog": dict(sorted(catalog.items())),
        "episodes": output_episodes,
    }
    audit = {
        "episodes": len(output_episodes),
        "prefixes": sum(len(row["prefixes"]) for row in output_episodes),
        "adjacent_transitions": sum(
            len(row["transitions"]) for row in output_episodes
        ),
        "candidate_count": len(catalog),
        "transition_positive_counts": dict(sorted(transition_positive.items())),
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "task_disjoint": not any(overlaps.values()),
        "semantic_state_leakage": leakage,
        "causal_label_blind_states": not leakage,
        "agent_private_scenario_exposures": private_exposures,
        "user_tool_events": user_tool_events,
        "dataset_content_sha256": stable_hash(dataset),
    }
    return dataset, audit
