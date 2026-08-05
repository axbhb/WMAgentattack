"""Causal contracts for the frozen tau3 multi-step scale-readiness pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .clean_evidence_probe import hashed_text
from .markov_sufficiency import (
    representation_feature_size,
    semantic_markov_feature_vector,
)
from .semantic_state_v3 import (
    find_semantic_state_v3_leakage,
    semantic_state_v3_payload,
)


MANIFEST_SCHEMA_VERSION = "wmagentattack.tau3_multistep_manifest.v1"
DATASET_SCHEMA_VERSION = "wmagentattack.tau3_multistep_dataset.v1"
TEXT_ACTION = "TEXT"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")
_FORBIDDEN_MODEL_KEYS = {
    "evaluation_criteria",
    "expert_calls",
    "final_state_sha256",
    "future_actions",
    "gold_actions",
    "reference_actions",
    "reward",
    "state_changed",
    "success_label",
    "task_success",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def normalized_tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else canonical_json(value)
    return {token.lower() for token in _WORD_RE.findall(text.replace("_", " "))}


def task_key(domain: str, source_split: str, task_id: str) -> str:
    return stable_hash(
        {"domain": domain, "source_split": source_split, "task_id": task_id}
    )


def candidate_id(domain: str, schema: Mapping[str, Any] | None) -> str:
    if schema is None:
        return f"tau3::{domain}::{TEXT_ACTION}"
    function = schema["function"]
    signature = stable_hash(function)[:12]
    return f"tau3::{domain}::{signature}::{function['name']}"


def candidate_catalog_for_model_input(
    domain: str, model_input: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    output = {
        candidate_id(domain, schema): {
            "domain": domain,
            "kind": "tool",
            "function": schema["function"],
        }
        for schema in model_input["tool_schemas"]
    }
    output[candidate_id(domain, None)] = {
        "domain": domain,
        "kind": "text_or_stop",
        "function": {
            "name": TEXT_ACTION,
            "description": "Return a textual response instead of calling a tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    return output


def action_candidate_id(
    domain: str,
    model_input: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    if decision["kind"] != "tool_call":
        return candidate_id(domain, None)
    matches = [
        schema
        for schema in model_input["tool_schemas"]
        if schema["function"]["name"] == decision["name"]
    ]
    if len(matches) != 1:
        raise ValueError("tool decision does not map to one presented schema")
    return candidate_id(domain, matches[0])


def allocate_stratum(
    task_keys: Sequence[str],
    *,
    seed: str,
    domain: str,
    stratum: str,
    counts: Mapping[str, int],
) -> dict[str, str]:
    expected = sum(int(value) for value in counts.values())
    if len(task_keys) != expected or len(set(task_keys)) != expected:
        raise ValueError("stratum size differs from frozen split counts")
    ordered = sorted(
        task_keys,
        key=lambda key: stable_hash([seed, domain, stratum, key]),
    )
    output: dict[str, str] = {}
    start = 0
    for split in ("training", "calibration", "confirmation"):
        stop = start + int(counts[split])
        for key in ordered[start:stop]:
            output[key] = split
        start = stop
    if start != len(ordered):
        raise ValueError("not all stratum tasks received a split")
    return output


def visible_observation(event: Mapping[str, Any]) -> str:
    if event["status"] == "error":
        error = event.get("error") or {}
        return "EXECUTION_ERROR " + canonical_json(error)
    return "TOOL_OUTPUT " + canonical_json(event.get("output"))


def append_ledger_event(
    ledger: Mapping[str, Any],
    *,
    episode_id: str,
    domain: str,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    output = json.loads(json.dumps(ledger, ensure_ascii=False, default=str))
    index = int(event["index"])
    action = event["action"]
    observation = visible_observation(event)
    entity_key = {
        "tool_name": str(action["name"]),
        "arguments": action["arguments"],
    }
    output["records"].append(
        {
            "record_id": f"record::{stable_hash(episode_id)[:16]}::{index:03d}",
            "entity_type": "tau3_tool_observation",
            "entity_key": entity_key,
            "entity_candidates": [
                {
                    "entity_id": "ENTITY::tau3::" + stable_hash(entity_key)[:24],
                    "entity_key": entity_key,
                }
            ],
            "link_status": "UNIQUE",
            "attributes": [
                {
                    "name": "visible_output",
                    "value": observation,
                    "kind": "SINGLE_VALUED",
                }
            ],
            "context": {"domain": domain},
            "source_tool": str(action["name"]),
            "source_arguments": action["arguments"],
            "call_index": index,
            "execution_status": str(event["status"]),
        }
    )
    output["execution_receipts"].append(
        {
            "episode_id": episode_id,
            "call_index": index,
            "tool_name": str(action["name"]),
            "arguments_fingerprint": stable_hash(action["arguments"]),
            "observation_fingerprint": stable_hash(observation),
            "execution_status": str(event["status"]),
        }
    )
    return output


def source_prefix(
    *,
    episode_id: str,
    domain: str,
    model_input: Mapping[str, Any],
    prefix_index: int,
    prior_events: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = candidate_catalog_for_model_input(domain, model_input)
    legal = sorted(catalog)
    target = action_candidate_id(domain, model_input, decision)
    if target not in legal:
        raise ValueError("next action target is not legal")
    if prior_events:
        previous = prior_events[-1]
        last_action = {
            "function": action_candidate_id(
                domain,
                model_input,
                {"kind": "tool_call", **previous["action"]},
            ),
            "arguments": previous["action"]["arguments"],
        }
        last_observation = visible_observation(previous)
        receipt = {
            "status": previous["status"],
            "error_type": (
                (previous.get("error") or {}).get("type")
                if previous["status"] == "error"
                else None
            ),
            "output_type": (
                type(previous.get("output")).__name__
                if previous["status"] != "error"
                else None
            ),
        }
    else:
        last_action = {"function": "<START>", "arguments": {}}
        last_observation = ""
        receipt = {"status": "start", "error_type": None, "output_type": None}
    features = {
        "trusted_goal": str(model_input["trusted_goal"]),
        "track": f"tau3:{domain}",
        "prefix_index": int(prefix_index),
        "legal_tools": legal,
        "last_action": last_action,
        "last_observation": last_observation,
        "execution_receipt": receipt,
        "ledger_v2": json.loads(json.dumps(ledger, ensure_ascii=False, default=str)),
    }
    leaked = _walk_keys(features) & _FORBIDDEN_MODEL_KEYS
    if leaked:
        raise ValueError(f"causal prefix leakage: {sorted(leaked)}")
    return {
        "prefix_index": int(prefix_index),
        "features": features,
        "targets": {
            "next_action": target,
            "argument_keys": sorted(
                decision.get("arguments", {}) if decision["kind"] == "tool_call" else {}
            ),
        },
        "decision_kind": str(decision["kind"]),
        "episode_id": episode_id,
    }


def transition_target(
    *,
    trusted_goal: str,
    prior_events: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
) -> dict[str, float]:
    observation = visible_observation(event)
    prior_observations = [visible_observation(row) for row in prior_events]
    goal_tokens = normalized_tokens(trusted_goal)
    prior_overlap = set().union(
        *(normalized_tokens(row) & goal_tokens for row in prior_observations)
    ) if prior_observations else set()
    current_overlap = normalized_tokens(observation) & goal_tokens
    output = event.get("output")
    return {
        "state_changed": float(bool(event["state_changed"])),
        "execution_error": float(event["status"] == "error"),
        "output_nonempty": float(output not in (None, "", [], {})),
        "goal_overlap_gained": float(bool(current_overlap - prior_overlap)),
        "novel_observation": float(
            stable_hash(observation) not in {stable_hash(row) for row in prior_observations}
        ),
    }


def observed_semantic_markov_v4_feature_vector(
    source_prefix_value: Mapping[str, Any], *, hash_dimension: int
) -> np.ndarray:
    """Add current causal observation/receipt/ledger channels at fixed width."""

    base = semantic_markov_feature_vector(
        source_prefix_value, hash_dimension=hash_dimension
    ).copy()
    features = source_prefix_value["features"]
    ledger = features["ledger_v2"]
    causal_ledger = {
        "records": [
            {
                "entity_type": row.get("entity_type"),
                "entity_key": row.get("entity_key", {}),
                "attributes": row.get("attributes", ()),
                "context": row.get("context", {}),
                "source_tool": row.get("source_tool"),
                "source_arguments": row.get("source_arguments", {}),
                "execution_status": row.get("execution_status"),
            }
            for row in ledger.get("records", ())
        ],
        "conflicts": ledger.get("conflicts", ()),
        "receipts": [
            {
                "tool_name": row.get("tool_name"),
                "execution_status": row.get("execution_status"),
            }
            for row in ledger.get("execution_receipts", ())
        ],
    }
    starts = (4 * hash_dimension, 5 * hash_dimension, 6 * hash_dimension)
    base[starts[0] : starts[0] + hash_dimension] = hashed_text(
        features["last_observation"], hash_dimension, "tau3-v4-current-observation"
    )
    base[starts[1] : starts[1] + hash_dimension] = hashed_text(
        features["execution_receipt"], hash_dimension, "tau3-v4-current-receipt"
    )
    base[starts[2] : starts[2] + hash_dimension] = hashed_text(
        causal_ledger, hash_dimension, "tau3-v4-causal-ledger"
    )
    numeric = base[7 * hash_dimension :]
    receipts = list(ledger.get("execution_receipts", ()))
    records = list(ledger.get("records", ()))
    errors = sum(row.get("execution_status") == "error" for row in receipts)
    tools = Counter(str(row.get("tool_name")) for row in receipts)
    last_tool = str(features["last_action"].get("function", "<START>")).rsplit(
        "::", 1
    )[-1]
    numeric[:9] = (
        math.log1p(int(features["prefix_index"])),
        math.log1p(len(receipts)),
        math.log1p(len(records)),
        math.log1p(errors),
        math.log1p(len(tools)),
        math.log1p(tools.get(last_tool, 0)),
        math.log1p(len(str(features["last_observation"]))),
        float(bool(features["last_observation"])),
        float(features["execution_receipt"].get("status") == "error"),
    )
    expected = representation_feature_size(hash_dimension)
    if base.shape != (expected,) or not np.isfinite(base).all():
        raise ValueError("observed Semantic Markov v4 feature integrity failure")
    return base.astype(np.float32, copy=False)


def build_dataset(
    manifest: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_rows = {row["episode_id"]: row for row in manifest["rows"]}
    if set(manifest_rows) != {row["episode_id"] for row in episodes}:
        raise ValueError("episode outputs do not exactly match the manifest")
    catalog: dict[str, dict[str, Any]] = {}
    output_episodes = []
    leakage = []
    split_tasks: dict[str, set[str]] = defaultdict(set)
    transition_counts = Counter()
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        manifest_row = manifest_rows[episode["episode_id"]]
        for key, descriptor in candidate_catalog_for_model_input(
            episode["domain"], manifest_row["model_input"]
        ).items():
            previous = catalog.setdefault(key, descriptor)
            if stable_hash(previous) != stable_hash(descriptor):
                raise ValueError("candidate schema collision")
        semantic_prefixes = []
        for prefix in episode["prefixes"]:
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
        transitions = []
        for index, event in enumerate(episode["transitions"]):
            target = transition_target(
                trusted_goal=manifest_row["model_input"]["trusted_goal"],
                prior_events=episode["transitions"][:index],
                event=event,
            )
            transition_counts.update(
                name for name, value in target.items() if value == 1.0
            )
            transitions.append(
                {
                    "transition_index": index,
                    "prefix_index": index,
                    "action": episode["prefixes"][index]["targets"]["next_action"],
                    "target": target,
                }
            )
        split_tasks[episode["split"]].add(episode["task_key"])
        output_episodes.append(
            {
                "episode_id": episode["episode_id"],
                "task_id": episode["task_key"],
                "suite": f"tau3:{episode['domain']}",
                "domain": episode["domain"],
                "split": episode["split"],
                "track": f"tau3:{episode['domain']}",
                "run_seed": episode["llm_seed"],
                "prefixes": episode["prefixes"],
                "semantic_prefixes": semantic_prefixes,
                "transitions": transitions,
                "termination": episode["termination"],
            }
        )
    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
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
        "transition_positive_counts": dict(sorted(transition_counts.items())),
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "task_disjoint": not any(overlaps.values()),
        "semantic_state_leakage": leakage,
        "causal_label_blind_states": not leakage,
        "dataset_content_sha256": stable_hash(dataset),
    }
    return dataset, audit
