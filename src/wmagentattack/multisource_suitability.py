"""Task-disjoint source adapters for the multi-source suitability experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hybrid_semantic_world_model import semantic_state_v3_feature_vector
from .markov_sufficiency import (
    FROZEN_SUFFICIENCY_VARIANTS,
    full_history_diagnostic_feature_vector,
    representation_feature_size,
    semantic_markov_feature_vector,
)
from .semantic_state_v3 import build_structured_semantic_state_v3


SUITABILITY_SCHEMA_VERSION = "wmagentattack.multisource_suitability.v1"
TEXT_ACTION = "TEXT"

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")
_FORBIDDEN_MODEL_KEYS = {
    "completion",
    "decision",
    "execution",
    "group_id",
    "prompt_sha256",
    "retry_completion",
    "row_id",
    "run_seed",
    "runtime_error",
    "simulator_audit_only",
    "source_index",
    "variant",
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_goal(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip().lower()


def task_fingerprint(source: str, trusted_goal: str) -> str:
    return stable_hash(
        {"source": source, "normalized_trusted_goal": normalized_goal(trusted_goal)}
    )


def split_task_fingerprints(
    task_fingerprints: Sequence[str],
    *,
    source: str,
    seed: str,
    train_ratio: float,
    calibration_ratio: float,
) -> dict[str, str]:
    unique = sorted(set(task_fingerprints))
    if len(unique) < 3:
        raise ValueError("at least three task units are required")
    ordered = sorted(unique, key=lambda key: stable_hash([seed, source, key]))
    train_count = max(1, int(math.floor(len(ordered) * train_ratio)))
    calibration_count = max(
        1, int(math.floor(len(ordered) * calibration_ratio))
    )
    if train_count + calibration_count >= len(ordered):
        calibration_count = 1
        train_count = len(ordered) - 2
    output = {}
    for index, key in enumerate(ordered):
        if index < train_count:
            split = "training"
        elif index < train_count + calibration_count:
            split = "calibration"
        else:
            split = "confirmation"
        output[key] = split
    return output


def candidate_id(
    source: str, tool_name: str, schema: Mapping[str, Any] | None = None
) -> str:
    if schema is None:
        return f"{source}::{tool_name}"
    signature = stable_hash(schema["function"])[:12]
    return f"{source}::{signature}::{tool_name}"


def _tool_name(schema: Mapping[str, Any]) -> str:
    return str(schema["function"]["name"])


def _safe_track(record: Mapping[str, Any]) -> str:
    source = str(record["source"])
    metadata = record.get("metadata", {})
    if source == "tool_sandbox":
        return f"tool_sandbox:{metadata.get('primary_category', 'OTHER')}"
    if source == "tau3":
        return f"tau3:{metadata.get('domain', 'unknown')}"
    return "injecagent:published_observation"


def _visible_observation(record: Mapping[str, Any]) -> str:
    if record["source"] == "injecagent":
        return str(record["model_input"].get("observation", ""))
    return ""


def _prior_tool(record: Mapping[str, Any]) -> str:
    if record["source"] == "injecagent":
        return str(record.get("metadata", {}).get("user_tool", "<OBSERVED_TOOL>"))
    return "<START>"


def causal_model_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only causal fields allowed to enter an action representation."""

    model_input = record["model_input"]
    schemas = model_input["tool_schemas"]
    return {
        "source": str(record["source"]),
        "trusted_goal": str(model_input["trusted_goal"]),
        "track": _safe_track(record),
        "tool_schemas": schemas,
        "legal_tool_names": [_tool_name(schema) for schema in schemas],
        "visible_observation": _visible_observation(record),
        "visible_prior_tool": _prior_tool(record),
    }


def _source_prefix(model_input: Mapping[str, Any]) -> dict[str, Any]:
    observed = bool(model_input["visible_observation"])
    prior_tool = str(model_input["visible_prior_tool"])
    return {
        "features": {
            "trusted_goal": model_input["trusted_goal"],
            "track": model_input["track"],
            "prefix_index": int(observed),
            "legal_tools": model_input["legal_tool_names"],
            "last_action": (
                {"function": prior_tool, "arguments": {}}
                if observed
                else {"function": "<START>", "arguments": {}}
            ),
            "last_observation": model_input["visible_observation"],
            "execution_receipt": {
                "status": "success" if observed else "start",
                "error_type": None,
                "output_type": "visible_observation" if observed else None,
            },
        }
    }


def _structured_state(model_input: Mapping[str, Any]) -> Any:
    observed = bool(model_input["visible_observation"])
    prior_tool = str(model_input["visible_prior_tool"])
    receipts = []
    records = []
    if observed:
        receipts.append(
            {"call_index": 0, "tool_name": prior_tool, "execution_status": "success"}
        )
        records.append(
            {
                "call_index": 0,
                "entity_type": "visible_tool_observation",
                "entity_key": {"source_tool": prior_tool},
                "entity_candidates": [],
                "link_status": "UNLINKED",
                "attributes": [
                    {
                        "name": "visible_content",
                        "value": model_input["visible_observation"],
                        "kind": "TEXT",
                    }
                ],
                "context": {"source": model_input["source"]},
                "source_tool": prior_tool,
                "source_arguments": {},
                "execution_status": "success",
            }
        )
    features = {
        "trusted_goal": model_input["trusted_goal"],
        "track": model_input["track"],
        "prefix_index": int(observed),
        "legal_tools": model_input["legal_tool_names"],
        "last_action": (
            {"function": prior_tool, "arguments": {}}
            if observed
            else {"function": "<START>", "arguments": {}}
        ),
        "last_observation": model_input["visible_observation"],
        "execution_receipt": {
            "status": "success" if observed else "start",
            "error_type": None,
            "output_type": "visible_observation" if observed else None,
        },
        "ledger_v2": {
            "records": records,
            "conflicts": [],
            "execution_receipts": receipts,
        },
    }
    return build_structured_semantic_state_v3(features)


def representation_vector(
    row: Mapping[str, Any], *, variant: str, hash_dimension: int
) -> np.ndarray:
    if variant not in FROZEN_SUFFICIENCY_VARIANTS:
        raise ValueError(f"unsupported frozen representation: {variant}")
    model_input = row["causal_model_input"]
    source_prefix = _source_prefix(model_input)
    if variant == "semantic_markov":
        output = semantic_markov_feature_vector(
            source_prefix, hash_dimension=hash_dimension
        )
    elif variant == "structured_markov_v3":
        output = semantic_state_v3_feature_vector(
            _structured_state(model_input), hash_dimension=hash_dimension
        )
    else:
        output = full_history_diagnostic_feature_vector(
            [source_prefix], prefix_index=0, hash_dimension=hash_dimension
        )
    expected = representation_feature_size(hash_dimension)
    if output.shape != (expected,):
        raise ValueError(f"representation shape mismatch: {output.shape}")
    return output


def tfidf_state_text(row: Mapping[str, Any]) -> str:
    model_input = row["causal_model_input"]
    return "\n".join(
        [
            f"SOURCE {model_input['source']}",
            f"TRACK {model_input['track']}",
            f"GOAL {model_input['trusted_goal']}",
            f"PRIOR_TOOL {model_input['visible_prior_tool']}",
            f"OBSERVATION {model_input['visible_observation']}",
            "TOOLS " + json.dumps(model_input["tool_schemas"], sort_keys=True),
        ]
    )


def candidate_text(descriptor: Mapping[str, Any]) -> str:
    return json.dumps(descriptor, ensure_ascii=False, sort_keys=True)


def build_suitability_dataset(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_config = protocol["split"]
    expected_counts = protocol["source"]["expected_source_counts"]
    observed_counts = Counter(str(row["source"]) for row in records)
    if dict(sorted(observed_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("source counts differ from the frozen protocol")

    task_maps = {}
    for source in sorted(observed_counts):
        source_tasks = [
            task_fingerprint(source, str(row["model_input"]["trusted_goal"]))
            for row in records
            if row["source"] == source
        ]
        task_maps[source] = split_task_fingerprints(
            source_tasks,
            source=source,
            seed=str(split_config["seed"]),
            train_ratio=float(split_config["ratios"]["training"]),
            calibration_ratio=float(split_config["ratios"]["calibration"]),
        )

    candidate_catalog: dict[str, dict[str, Any]] = {}
    schema_conflicts = []
    output_rows = []
    for record in records:
        source = str(record["source"])
        causal = causal_model_input(record)
        task_key = task_fingerprint(source, causal["trusted_goal"])
        legal = []
        for schema in causal["tool_schemas"]:
            name = _tool_name(schema)
            key = candidate_id(source, name, schema)
            descriptor = {
                "source": source,
                "kind": "tool",
                "function": schema["function"],
            }
            previous = candidate_catalog.setdefault(key, descriptor)
            if stable_hash(previous) != stable_hash(descriptor):
                schema_conflicts.append(key)
            legal.append(key)
        text_key = candidate_id(source, TEXT_ACTION)
        candidate_catalog.setdefault(
            text_key,
            {
                "source": source,
                "kind": "text_or_stop",
                "function": {
                    "name": TEXT_ACTION,
                    "description": "Return a textual response instead of calling a tool.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
        legal.append(text_key)
        target_name = (
            str(record["decision"]["name"])
            if record["decision"]["kind"] == "tool_call"
            else TEXT_ACTION
        )
        if target_name == TEXT_ACTION:
            target = text_key
        else:
            matching = [
                key
                for key, schema in zip(legal, causal["tool_schemas"])
                if _tool_name(schema) == target_name
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"target tool schema is not unique in the legal interface: {target_name}"
                )
            target = matching[0]
        execution = record["execution"]
        exact = execution.get("tier") == "exact"
        replica = execution.get("replica_0", {}) if exact else {}
        safe_row = {
            "row_id": str(record["row_id"]),
            "source": source,
            "task_key": task_key,
            "split": task_maps[source][task_key],
            "group_id": str(record["group_id"]),
            "variant": str(record["variant"]),
            "causal_model_input": causal,
            "legal_candidate_ids": legal,
            "target_candidate_id": target,
            "target_is_tool": target_name != TEXT_ACTION,
            "exact_outcome": {
                "available": exact,
                "execution_error": bool(exact and replica.get("status") == "error"),
                "state_changed": bool(exact and replica.get("state_changed") is True),
                "output_nonempty": bool(exact and replica.get("output") is not None),
            },
        }
        safe_row["causal_input_fingerprint"] = stable_hash(causal)
        output_rows.append(safe_row)

    output_rows.sort(key=lambda row: row["row_id"])
    dataset = {
        "schema_version": SUITABILITY_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "rows": output_rows,
        "candidate_catalog": dict(sorted(candidate_catalog.items())),
    }

    split_tasks: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    split_fingerprints: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in output_rows:
        split_tasks[row["source"]][row["split"]].add(row["task_key"])
        split_fingerprints[row["source"]][row["split"]].add(
            row["causal_input_fingerprint"]
        )
    task_overlaps = {}
    fingerprint_overlaps = {}
    for source in sorted(observed_counts):
        for left, right in (
            ("training", "calibration"),
            ("training", "confirmation"),
            ("calibration", "confirmation"),
        ):
            key = f"{source}::{left}::{right}"
            task_overlaps[key] = sorted(
                split_tasks[source][left] & split_tasks[source][right]
            )
            fingerprint_overlaps[key] = sorted(
                split_fingerprints[source][left]
                & split_fingerprints[source][right]
            )

    pair_splits: dict[str, set[str]] = defaultdict(set)
    pair_variants: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        if row["source"] == "injecagent":
            pair_splits[row["group_id"]].add(row["split"])
            pair_variants[row["group_id"]].add(row["variant"])
    bad_pairs = sorted(
        group
        for group in pair_splits
        if len(pair_splits[group]) != 1
        or pair_variants[group] != {"clean", "poisoned"}
    )

    source_audits = {}
    exact_probe_authorized = {}
    preflight = protocol["preflight_gate"]
    for source in sorted(observed_counts):
        rows = [row for row in output_rows if row["source"] == source]
        exact = [row for row in rows if row["exact_outcome"]["available"]]
        exact_by_split = {}
        for split in ("training", "calibration", "confirmation"):
            selected = [row for row in exact if row["split"] == split]
            exact_by_split[split] = {
                "rows": len(selected),
                "errors": sum(row["exact_outcome"]["execution_error"] for row in selected),
                "successes": sum(
                    not row["exact_outcome"]["execution_error"] for row in selected
                ),
                "state_changed": sum(
                    row["exact_outcome"]["state_changed"] for row in selected
                ),
            }
        total_errors = sum(row["exact_outcome"]["execution_error"] for row in exact)
        total_successes = len(exact) - total_errors
        minimum_each = int(
            preflight["minimum_each_error_class_per_training_and_confirmation"]
        )
        authorized = (
            len(exact) >= int(preflight["minimum_exact_rows_for_error_probe"])
            and total_errors >= int(preflight["minimum_exact_errors_for_error_probe"])
            and total_successes
            >= int(preflight["minimum_exact_successes_for_error_probe"])
            and all(
                exact_by_split[split]["errors"] >= minimum_each
                and exact_by_split[split]["successes"] >= minimum_each
                for split in ("training", "confirmation")
            )
        )
        exact_probe_authorized[source] = authorized
        source_audits[source] = {
            "rows": len(rows),
            "task_units": len({row["task_key"] for row in rows}),
            "task_counts": {
                split: len(split_tasks[source][split])
                for split in ("training", "calibration", "confirmation")
            },
            "row_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "target_candidates": len({row["target_candidate_id"] for row in rows}),
            "text_rows": sum(not row["target_is_tool"] for row in rows),
            "tool_rows": sum(row["target_is_tool"] for row in rows),
            "exact_rows": len(exact),
            "exact_errors": total_errors,
            "exact_successes": total_successes,
            "state_changed_rows": sum(
                row["exact_outcome"]["state_changed"] for row in exact
            ),
            "exact_by_split": exact_by_split,
            "adjacent_semantic_transitions": 0,
            "error_probe_authorized": authorized,
        }

    checks = {
        "expected_rows": len(output_rows) == int(protocol["source"]["expected_rows"]),
        "expected_source_counts": dict(sorted(observed_counts.items()))
        == dict(sorted(expected_counts.items())),
        "minimum_task_units": all(
            row["task_units"]
            >= int(preflight["minimum_task_units_per_source"])
            for row in source_audits.values()
        ),
        "minimum_confirmation_tasks": all(
            row["task_counts"]["confirmation"]
            >= int(preflight["minimum_confirmation_tasks_per_source"])
            for row in source_audits.values()
        ),
        "zero_task_overlap": not any(task_overlaps.values()),
        "zero_causal_input_fingerprint_overlap": not any(
            fingerprint_overlaps.values()
        ),
        "target_in_legal_candidates": all(
            row["target_candidate_id"] in row["legal_candidate_ids"]
            for row in output_rows
        ),
        "injecagent_pairs_within_split": not bad_pairs,
        "candidate_schema_consistency": not schema_conflicts,
        "forbidden_model_keys_absent": all(
            not (_FORBIDDEN_MODEL_KEYS & set(row["causal_model_input"]))
            for row in output_rows
        ),
    }
    audit = {
        "schema_version": SUITABILITY_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "source_audits": source_audits,
        "error_probe_authorized": exact_probe_authorized,
        "task_overlaps": task_overlaps,
        "causal_input_fingerprint_overlaps": fingerprint_overlaps,
        "bad_injecagent_pairs": bad_pairs,
        "candidate_schema_conflicts": sorted(set(schema_conflicts)),
        "rows": len(output_rows),
        "candidate_count": len(candidate_catalog),
        "dataset_content_sha256": stable_hash(dataset),
        "structural_counterevidence": {
            "adjacent_semantic_transitions": 0,
            "full_five_label_evidence_head_trainable": False,
            "reason": "Every record is a one-step decision; clean/poison pairs are interventions, not temporal transitions.",
        },
    }
    return dataset, audit
