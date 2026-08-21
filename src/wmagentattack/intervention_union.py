"""Deterministic, group-aware union of v17/v18/v19 intervention transitions."""

from __future__ import annotations

import copy
import datetime as dt
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_state import canonical_json_value, stable_fingerprint
from .hybrid_semantic_world_model import evidence_delta_target
from .semantic_state_v3 import find_semantic_state_v3_leakage


INTERVENTION_UNION_SCHEMA_VERSION = "wmagentattack.intervention_union.v20"
SOURCE_VERSIONS = ("v17_legal_fork", "v18_parameter_boundary", "v19_persistence_conflict")
FOLD_BY_DIFFICULTY = {"L1": 0, "L2": 1, "L3": 2}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _date_kind(value: str) -> str | None:
    try:
        dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return "VALID_DATETIME" if "T" in value or " " in value else "VALID_DATE"


def _argument_value_class(value: Any) -> Any:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {
            "type": "integer",
            "range": "missing_entity_sentinel" if value >= 2_000_000_000 else "ordinary",
        }
    if isinstance(value, float):
        return {"type": "number", "sign": "positive" if value > 0 else "nonpositive"}
    if isinstance(value, str):
        lowered = value.lower()
        if "missing" in lowered or "not-a-date" in lowered:
            category = "invalid_or_missing_sentinel"
        elif lowered.startswith("wm-v19-") or lowered.startswith("wm v19 "):
            category = "synthetic_intervention_value"
        elif EMAIL_RE.match(value):
            category = "email"
        elif (date_kind := _date_kind(value)) is not None:
            category = date_kind.lower()
        elif value in {"r", "rw"}:
            category = "permission"
        else:
            category = "text"
        return {"type": "string", "category": category, "length_bucket": min(len(value) // 8, 8)}
    if isinstance(value, list):
        return {"type": "list", "length_bucket": min(len(value), 8)}
    if isinstance(value, Mapping):
        return {"type": "object", "keys": sorted(map(str, value))}
    return {"type": type(value).__name__}


def normalized_action_descriptor(
    action: Mapping[str, Any], current_state: Mapping[str, Any]
) -> dict[str, Any]:
    arguments = canonical_json_value(action.get("arguments", {}))
    state_text = str(canonical_json_value(current_state))
    rows = []
    for key, value in sorted(arguments.items()):
        exact_link = False
        if value not in (None, "", [], {}):
            exact_link = str(value) in state_text
        rows.append(
            {
                "field": str(key),
                "value_class": _argument_value_class(value),
                "exact_value_observed_in_state": exact_link,
            }
        )
    return {
        "tool_id": str(action["tool_id"]),
        "arguments": rows,
    }


def effect_descriptor(
    current_state: Mapping[str, Any], next_state: Mapping[str, Any]
) -> dict[str, Any]:
    current_records = list(current_state.get("evidence_records", []))
    next_records = list(next_state.get("evidence_records", []))
    if next_records[: len(current_records)] != current_records:
        raise ValueError("semantic evidence history was rewritten")
    current_conflicts = list(current_state.get("conflicts", []))
    next_conflicts = list(next_state.get("conflicts", []))
    if next_conflicts[: len(current_conflicts)] != current_conflicts:
        raise ValueError("semantic conflict history was rewritten")
    added_records = []
    for record in next_records[len(current_records) :]:
        added_records.append(
            {
                "entity_type": record.get("entity_type"),
                "link_status": record.get("link_status"),
                "source_tool": record.get("source_tool"),
                "attributes": sorted(
                    {
                        (
                            str(attribute.get("name")),
                            str(attribute.get("kind")),
                        )
                        for attribute in record.get("attributes", [])
                    }
                ),
            }
        )
    current_matched = set(
        current_state.get("goal_evidence", {}).get("matched_fact_terms", [])
    )
    next_matched = set(next_state.get("goal_evidence", {}).get("matched_fact_terms", []))
    delta = evidence_delta_target(current_state, next_state).astype(int).tolist()
    return canonical_json_value(
        {
            "delta_bits": delta,
            "execution_status": next_state.get("execution", {}).get("last_status"),
            "added_records": added_records,
            "added_conflicts": [
                {
                    "attribute_name": row.get("attribute_name"),
                    "reason": row.get("reason"),
                }
                for row in next_conflicts[len(current_conflicts) :]
            ],
            "newly_matched_goal_term_count": len(next_matched - current_matched),
        }
    )


def effect_tokens(effect: Mapping[str, Any]) -> list[str]:
    tokens = [
        f"delta_bit_{index}={int(value)}"
        for index, value in enumerate(effect["delta_bits"])
    ]
    tokens.append(f"execution={effect['execution_status']}")
    tokens.append(f"matched_count={min(int(effect['newly_matched_goal_term_count']), 3)}")
    for record in effect["added_records"]:
        tokens.extend(
            (
                f"entity={record['entity_type']}",
                f"link={record['link_status']}",
                f"source={record['source_tool']}",
            )
        )
        for name, kind in record["attributes"]:
            tokens.append(f"attribute={record['entity_type']}::{name}::{kind}")
    for conflict in effect["added_conflicts"]:
        tokens.append(f"conflict={conflict['attribute_name']}")
    return sorted(set(tokens))


def _catalog_insert(
    catalog: dict[str, Any], state_ref: str, state: Mapping[str, Any]
) -> None:
    payload = canonical_json_value(state)
    if stable_fingerprint(payload) != state_ref:
        raise ValueError("semantic state reference mismatch")
    if state_ref in catalog and catalog[state_ref] != payload:
        raise ValueError("semantic state hash collision")
    catalog[state_ref] = payload


def _base_catalog(*datasets: Mapping[str, Any]) -> dict[str, Any]:
    catalog: dict[str, Any] = {}
    for dataset in datasets:
        for field in ("current_state_catalog", "next_state_catalog"):
            for state_ref, state in dataset.get(field, {}).items():
                _catalog_insert(catalog, str(state_ref), state)
    return catalog


def _raw_transition(
    *,
    source_version: str,
    task_id: str,
    suite: str,
    difficulty: str,
    root_ref: str,
    pair_ref: str,
    sequence_ref: str | None,
    step_index: int,
    role: str,
    current_state: Mapping[str, Any],
    action: Mapping[str, Any],
    next_state: Mapping[str, Any],
) -> dict[str, Any]:
    current_ref = stable_fingerprint(current_state)
    next_ref = stable_fingerprint(next_state)
    normalized_action = normalized_action_descriptor(action, current_state)
    effect = effect_descriptor(current_state, next_state)
    model_payload = {
        "current_semantic_state": canonical_json_value(current_state),
        "normalized_action": normalized_action,
    }
    target_payload = {
        "execution_error": int(effect["execution_status"] == "error"),
        "delta_bits": effect["delta_bits"],
        "effect_tokens": effect_tokens(effect),
    }
    return {
        "raw_transition_ref": stable_fingerprint(
            {
                "source_version": source_version,
                "task_id": task_id,
                "root_ref": root_ref,
                "pair_ref": pair_ref,
                "sequence_ref": sequence_ref,
                "step_index": step_index,
                "current_ref": current_ref,
                "action": action,
                "next_ref": next_ref,
            }
        ),
        "source_version": source_version,
        "task_id": task_id,
        "suite": suite,
        "difficulty": difficulty,
        "confirmation_fold": FOLD_BY_DIFFICULTY[difficulty],
        "root_ref": root_ref,
        "pair_ref": pair_ref,
        "sequence_ref": sequence_ref,
        "step_index": step_index,
        "role": role,
        "current_state_ref": current_ref,
        "next_state_ref": next_ref,
        "full_action_audit_only": canonical_json_value(action),
        "model_input": model_payload,
        "model_target": target_payload,
    }


def build_intervention_union(
    v17: Mapping[str, Any],
    v18: Mapping[str, Any],
    v19: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = _base_catalog(v17, v18)
    raw_rows = []
    source_counts = Counter()
    for source_version, dataset in (
        ("v17_legal_fork", v17),
        ("v18_parameter_boundary", v18),
    ):
        for outcome in dataset["counterfactual_outcomes"]:
            metadata = outcome["metadata"]
            current_ref = str(outcome["model_visible"]["current_state_ref"])
            next_state = outcome["model_visible"]["next_semantic_state"]
            next_ref = str(outcome["model_visible"]["next_state_ref"])
            _catalog_insert(catalog, next_ref, next_state)
            action = outcome["model_visible"]["candidate_action"]
            root_ref = current_ref
            pair_ref = stable_fingerprint(
                {
                    "source_version": source_version,
                    "root_ref": root_ref,
                    "tool_id": action["tool_id"] if source_version.startswith("v18") else None,
                }
            )
            raw_rows.append(
                _raw_transition(
                    source_version=source_version,
                    task_id=str(metadata["task_id"]),
                    suite=str(metadata["suite"]),
                    difficulty=str(metadata["difficulty"]),
                    root_ref=root_ref,
                    pair_ref=pair_ref,
                    sequence_ref=None,
                    step_index=0,
                    role="legal_action_fork" if source_version.startswith("v17") else "parameter_boundary",
                    current_state=catalog[current_ref],
                    action=action,
                    next_state=next_state,
                )
            )
            source_counts[source_version] += 1
    for sequence in v19["sequences"]:
        root_ref = str(sequence["model_visible"]["root_state_ref"])
        current_state = catalog[root_ref]
        for step in sorted(sequence["model_visible"]["steps"], key=lambda row: row["step_index"]):
            next_state = step["next_semantic_state"]
            next_ref = str(step["next_state_ref"])
            _catalog_insert(catalog, next_ref, next_state)
            raw_rows.append(
                _raw_transition(
                    source_version="v19_persistence_conflict",
                    task_id=str(sequence["task_id"]),
                    suite=str(sequence["suite"]),
                    difficulty=str(sequence["difficulty"]),
                    root_ref=root_ref,
                    pair_ref=str(sequence["pair_ref"]),
                    sequence_ref=str(sequence["sequence_ref"]),
                    step_index=int(step["step_index"]),
                    role=str(step["role"]),
                    current_state=current_state,
                    action=step["candidate_action"],
                    next_state=next_state,
                )
            )
            current_state = next_state
            source_counts["v19_persistence_conflict"] += 1

    canonical_by_key: dict[str, dict[str, Any]] = {}
    target_conflicts = []
    for row in raw_rows:
        input_key = stable_fingerprint(
            {
                "task_id": row["task_id"],
                "current_state_ref": row["current_state_ref"],
                "model_input": row["model_input"],
                "full_action": row["full_action_audit_only"],
            }
        )
        target_key = stable_fingerprint(row["model_target"])
        if input_key in canonical_by_key:
            existing = canonical_by_key[input_key]
            if stable_fingerprint(existing["model_target"]) != target_key:
                target_conflicts.append(input_key)
                continue
            existing["source_versions"] = sorted(
                set(existing["source_versions"] + [row["source_version"]])
            )
            existing["raw_transition_refs"].append(row["raw_transition_ref"])
            existing["group_memberships"].append(
                {
                    key: row[key]
                    for key in ("source_version", "root_ref", "pair_ref", "sequence_ref", "step_index", "role")
                }
            )
            continue
        canonical_by_key[input_key] = {
            "transition_ref": stable_fingerprint(
                {"input_key": input_key, "target_key": target_key}
            ),
            "task_id": row["task_id"],
            "suite": row["suite"],
            "difficulty": row["difficulty"],
            "confirmation_fold": row["confirmation_fold"],
            "current_state_ref": row["current_state_ref"],
            "next_state_ref": row["next_state_ref"],
            "source_versions": [row["source_version"]],
            "raw_transition_refs": [row["raw_transition_ref"]],
            "group_memberships": [
                {
                    key: row[key]
                    for key in ("source_version", "root_ref", "pair_ref", "sequence_ref", "step_index", "role")
                }
            ],
            "model_input": row["model_input"],
            "model_target": row["model_target"],
        }
    rows = sorted(canonical_by_key.values(), key=lambda row: row["transition_ref"])
    leakage = {}
    for row in rows:
        findings = find_semantic_state_v3_leakage(row["model_input"]["current_semantic_state"])
        if findings:
            leakage[row["transition_ref"]] = list(findings)
    task_folds = defaultdict(set)
    root_folds = defaultdict(set)
    pair_folds = defaultdict(set)
    sequence_folds = defaultdict(set)
    for row in rows:
        fold = row["confirmation_fold"]
        task_folds[row["task_id"]].add(fold)
        for membership in row["group_memberships"]:
            root_folds[membership["root_ref"]].add(fold)
            pair_folds[membership["pair_ref"]].add(fold)
            if membership["sequence_ref"] is not None:
                sequence_folds[membership["sequence_ref"]].add(fold)
    forbidden_model_input_keys = {
        "task_id", "suite", "difficulty", "confirmation_fold", "root_ref",
        "pair_ref", "sequence_ref", "source_version", "audit_only",
        "simulator_audit_only", "utility", "security", "final_report",
    }
    input_key_leakage = {
        key
        for row in rows
        for key in row["model_input"]
        if key in forbidden_model_input_keys
    }
    audit = {
        "source_raw_transition_counts": dict(sorted(source_counts.items())),
        "raw_transition_count": len(raw_rows),
        "canonical_transition_count": len(rows),
        "deduplicated_occurrences": len(raw_rows) - len(rows),
        "tasks": len(task_folds),
        "suites": sorted({row["suite"] for row in rows}),
        "difficulties": sorted({row["difficulty"] for row in rows}),
        "fold_counts": dict(sorted(Counter(row["confirmation_fold"] for row in rows).items())),
        "target_conflicts": sorted(target_conflicts),
        "semantic_state_leakage": leakage,
        "model_input_group_key_leakage": sorted(input_key_leakage),
        "task_cross_fold_groups": sorted(key for key, folds in task_folds.items() if len(folds) != 1),
        "root_cross_fold_groups": sorted(key for key, folds in root_folds.items() if len(folds) != 1),
        "pair_cross_fold_groups": sorted(key for key, folds in pair_folds.items() if len(folds) != 1),
        "sequence_cross_fold_groups": sorted(key for key, folds in sequence_folds.items() if len(folds) != 1),
        "unique_transition_refs": len({row["transition_ref"] for row in rows}) == len(rows),
        "all_sources_present": sorted(source_counts) == sorted(SOURCE_VERSIONS),
        "all_semantic_steps_adjacent": all(
            row["model_input"]["current_semantic_state"]["step_index"] + 1
            == catalog[row["next_state_ref"]]["step_index"]
            for row in rows
        ),
    }
    dataset = {
        "schema_version": INTERVENTION_UNION_SCHEMA_VERSION,
        "scope": "clean-only grouped v17+v18+v19 intervention transition union",
        "loader_contract": {
            "task_suite_difficulty_and_group_ids_are_split_or_audit_only": True,
            "model_input_contains_only_semantic_state_and_normalized_action": True,
            "full_action_values_are_audit_only": True,
            "no_utility_security_attack_or_final_outcome_labels": True,
            "pairs_and_sequences_never_cross_confirmation_folds": True,
        },
        "effect_token_vocabulary": sorted(
            {token for row in rows for token in row["model_target"]["effect_tokens"]}
        ),
        "transitions": rows,
    }
    return canonical_json_value(dataset), canonical_json_value(audit)
