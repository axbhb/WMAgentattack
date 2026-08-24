"""Typed, privacy-safe record--goal relation representation for v31."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .bound_successor_world_model import record_signature
from .decision_state import canonical_json_value


SCHEMA_VERSION = "wmagentattack.typed_relation_contract.v31"
_TOKEN = re.compile(r"[a-z0-9]+")
_FORBIDDEN_KEYS = {
    "normalized_goal", "fact_terms", "value", "task_id", "task_id_split_only",
    "suite", "suite_split_only", "utility", "security", "attack", "outcome",
    "final_outcome", "matched_goal_terms",
}


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


def lexical_tokens(value: str) -> set[str]:
    tokens = set(_TOKEN.findall(str(value).lower().replace("_", " ")))
    expanded = set(tokens)
    for token in tokens:
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
        for suffix in ("name", "number", "count", "time", "date", "price", "size", "status", "owner", "email", "address", "permission"):
            if token.endswith(suffix) and token != suffix:
                expanded.add(suffix)
                expanded.add(token[: -len(suffix)])
    return {token for token in expanded if token}


def decode_record(signature: str) -> dict[str, Any]:
    value = json.loads(signature)
    return {
        "entity_type": str(value["entity_type"]),
        "link_status": str(value["link_status"]),
        "attributes": sorted(
            [
                {"name": str(row["name"]), "kind": str(row.get("kind", "UNKNOWN"))}
                for row in value.get("attributes", ())
            ],
            key=lambda row: (row["name"], row["kind"]),
        ),
    }


def schema_vocabulary(signatures: Sequence[str]) -> dict[str, set[str]]:
    entities: set[str] = set()
    attributes: set[str] = set()
    entity_tokens: set[str] = set()
    attribute_tokens: set[str] = set()
    for signature in signatures:
        record = decode_record(signature)
        entities.add(record["entity_type"])
        entity_tokens |= lexical_tokens(record["entity_type"])
        for attribute in record["attributes"]:
            attributes.add(attribute["name"])
            attribute_tokens |= lexical_tokens(attribute["name"])
    return {
        "entities": entities,
        "attributes": attributes,
        "entity_tokens": entity_tokens,
        "attribute_tokens": attribute_tokens,
    }


def typed_goal_units(
    goal: Mapping[str, Any], action: Mapping[str, Any], vocabulary: Mapping[str, set[str]]
) -> list[dict[str, Any]]:
    fields = {
        str(row.get("field", ""))
        for row in action.get("arguments", ())
        if str(row.get("field", ""))
    }
    field_tokens = set().union(*(lexical_tokens(field) for field in fields)) if fields else set()
    typed_values: dict[str, set[str]] = {}
    for mention in goal.get("typed_mentions", ()):
        kind = str(mention.get("kind", "unknown")).lower()
        for token in lexical_tokens(str(mention.get("value", ""))):
            typed_values.setdefault(token, set()).add(kind)
    operations = {str(value).lower() for value in goal.get("operation_terms", ())}
    logic = {str(value).lower() for value in goal.get("logic_terms", ())}
    units = []
    for index, term_value in enumerate(goal.get("fact_terms", ())):
        term = str(term_value).lower()
        tokens = lexical_tokens(term)
        roles = set()
        if tokens & vocabulary["attribute_tokens"]:
            roles.add("ATTRIBUTE_TOKEN")
        if tokens & vocabulary["entity_tokens"]:
            roles.add("ENTITY_TOKEN")
        if tokens & field_tokens:
            roles.add("ACTION_FIELD_TOKEN")
        for token in tokens:
            for kind in typed_values.get(token, ()):
                roles.add(f"VALUE_KIND:{kind}")
        if term in operations:
            roles.add("OPERATION_TOKEN")
        if term in logic:
            roles.add("LOGIC_TOKEN")
        if not roles:
            roles.add("LEXICAL_TOKEN")
        context = "|".join(sorted(operations | logic))
        units.append({
            "index": index,
            "unit_hash": stable_hash("v31-goal-unit", term),
            "context_hash": stable_hash("v31-goal-context", context),
            "roles": sorted(roles),
            "_text": term,
            "_query": (
                f"query: goal fact {term}; operations {' '.join(sorted(operations)) or 'none'}; "
                f"logic {' '.join(sorted(logic)) or 'none'}"
            ),
        })
    return units


def record_description(signature: str) -> str:
    record = decode_record(signature)
    attributes = " ".join(
        f"{row['name'].replace('_', ' ')} {row['kind'].lower()}"
        for row in record["attributes"]
    )
    return (
        f"passage: evidence record entity {record['entity_type'].replace('_', ' ')}; "
        f"link {record['link_status'].lower()}; attributes {attributes or 'none'}"
    )


def structural_relation(
    unit: Mapping[str, Any], signature: str, action: Mapping[str, Any]
) -> tuple[list[str], float]:
    record = decode_record(signature)
    term_tokens = lexical_tokens(str(unit["_text"]))
    entity_tokens = lexical_tokens(record["entity_type"])
    attribute_tokens = set().union(
        *(lexical_tokens(row["name"]) for row in record["attributes"])
    ) if record["attributes"] else set()
    action_fields = {
        str(row.get("field", "")) for row in action.get("arguments", ()) if row.get("field")
    }
    field_tokens = set().union(*(lexical_tokens(value) for value in action_fields)) if action_fields else set()
    types = []
    score = 0.0
    if term_tokens & attribute_tokens:
        types.append("DIRECT_ATTRIBUTE")
        score = max(score, 1.0)
    if term_tokens & entity_tokens:
        types.append("DIRECT_ENTITY")
        score = max(score, 0.9)
    if term_tokens & field_tokens:
        types.append("ACTION_FIELD_BRIDGE")
        score = max(score, 0.75)
    if any(str(role).startswith("VALUE_KIND:") for role in unit["roles"]):
        types.append("TYPED_VALUE_CARRIER")
        score = max(score, 0.55)
    if not types:
        types.append("SEMANTIC_ONLY")
    return sorted(types), score


def relation_score(structural_score: float, semantic_similarity: float) -> float:
    semantic_01 = max(0.0, min(1.0, (float(semantic_similarity) + 1.0) / 2.0))
    return 0.65 * float(structural_score) + 0.35 * semantic_01


def has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if _FORBIDDEN_KEYS & set(value):
            return True
        return any(has_forbidden_key(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(has_forbidden_key(child) for child in value)
    return False


def sanitize_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_json_value({
        "index": int(unit["index"]),
        "unit_hash": str(unit["unit_hash"]),
        "context_hash": str(unit["context_hash"]),
        "roles": list(unit["roles"]),
    })


def gold_pairs(row: Mapping[str, Any]) -> set[tuple[str, int]]:
    output = set()
    target = row["model_target"]["relational_successor_delta"]
    for record in target["added_evidence_records"]:
        signature = record_signature(record)
        for index in record["newly_matched_goal_term_indices"]:
            output.add((signature, int(index)))
    return output

