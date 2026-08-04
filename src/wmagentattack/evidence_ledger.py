"""Deterministic, goal-conditioned evidence memory for clean AgentDojo traces.

The ledger consumes only causally observed tool results and execution metadata.
It never stores utility, security, expert coverage, or future events. Expert
coverage may be used later as a target, but is deliberately absent here.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .decision_state import canonical_json_value, stable_fingerprint


EVIDENCE_LEDGER_SCHEMA_VERSION = "wmagentattack.evidence_ledger.v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*")
_FIELD_RE = re.compile(
    r"(?:^|[\n,;])\s*([A-Za-z][A-Za-z0-9 _/()-]{0,48})\s*:\s*",
    flags=re.MULTILINE,
)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVIDENCE_LEDGER_SCHEMA_VERSION
    item_id: str
    entity_candidates: tuple[str, ...]
    attribute: str
    value: str
    source_tool: str
    source_arguments: dict[str, Any]
    step_index: int = Field(ge=0)
    execution_status: Literal["success", "error"]
    goal_overlap: float = Field(ge=0.0, le=1.0)
    argument_link_status: Literal["UNIQUE", "AMBIGUOUS", "UNLINKED"]
    novelty: Literal["new", "duplicate"]
    conflict_status: Literal["none", "conflict"]
    state_provenance: Literal["read_only", "mutating"]
    content_fingerprint: str
    outcome_labels_present: bool = False


class EvidenceLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVIDENCE_LEDGER_SCHEMA_VERSION
    items: tuple[EvidenceItem, ...] = ()
    outcome_labels_present: bool = False


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(str(text)))


def _normal_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _flatten_literal(value: Any, path: tuple[str, ...] = ()):  # noqa: ANN202
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            yield from _flatten_literal(item, (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _flatten_literal(item, (*path, str(index)))
        return
    attribute = ".".join(path) if path else "observation"
    yield attribute, _normal_text(value)


def parse_observation_facts(text: str) -> tuple[tuple[str, str], ...]:
    """Parse synthetic tool text without a learned or task-specific ontology."""

    normalized = str(text or "").strip()
    if not normalized:
        return (("empty_observation", ""),)

    if normalized[0] in "[{(" and normalized[-1] in "]})":
        try:
            literal = ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            literal = None
        if literal is not None:
            rows = tuple(_flatten_literal(literal))
            if rows:
                return rows

    matches = list(_FIELD_RE.finditer(normalized))
    if matches:
        rows = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            attribute = "_".join(tokenize(match.group(1))) or "observation"
            value = _normal_text(normalized[start:end].strip(" \n,;"))
            rows.append((attribute, value))
        if rows:
            return tuple(rows)

    lines = tuple(_normal_text(line) for line in normalized.splitlines() if line.strip())
    if len(lines) > 1:
        return tuple(("observation", line) for line in lines)
    return (("observation", _normal_text(normalized)),)


def _argument_scalars(value: Any) -> tuple[str, ...]:
    rows = []
    if isinstance(value, Mapping):
        for item in value.values():
            rows.extend(_argument_scalars(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            rows.extend(_argument_scalars(item))
    elif value is not None:
        rows.append(_normal_text(value))
    return tuple(rows)


def _entity_candidates(arguments: Mapping[str, Any], value: str) -> tuple[str, ...]:
    lowered = value.lower()
    candidates = sorted(
        {
            item
            for item in _argument_scalars(arguments)
            if item and item.lower() in lowered
        }
    )
    return tuple(candidates or ["UNLINKED"])


def _goal_overlap(goal: str, attribute: str, value: str) -> float:
    goal_tokens = set(tokenize(goal))
    fact_tokens = set(tokenize(f"{attribute} {value}"))
    if not goal_tokens or not fact_tokens:
        return 0.0
    return len(goal_tokens & fact_tokens) / len(fact_tokens)


def _link_status(counts: Mapping[str, int]) -> Literal["UNIQUE", "AMBIGUOUS", "UNLINKED"]:
    if int(counts.get("ambiguous", 0)) > 0:
        return "AMBIGUOUS"
    if int(counts.get("unique", 0)) > 0:
        return "UNIQUE"
    return "UNLINKED"


def update_evidence_ledger(
    ledger: EvidenceLedger,
    *,
    goal: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    observation_text: str,
    step_index: int,
    execution_status: Literal["success", "error"],
    error_type: str | None,
    argument_link_resolution: Mapping[str, int],
    state_changed: bool,
) -> EvidenceLedger:
    """Append deterministic evidence items from one executed tool result."""

    arguments = canonical_json_value(dict(arguments))
    if execution_status == "error":
        facts = (("execution_error", str(error_type or "unknown_error")),)
    else:
        facts = parse_observation_facts(observation_text)

    existing_fingerprints = {item.content_fingerprint for item in ledger.items}
    values_by_key: dict[tuple[tuple[str, ...], str], set[str]] = {}
    for item in ledger.items:
        key = (item.entity_candidates, item.attribute)
        values_by_key.setdefault(key, set()).add(item.value)

    new_items = list(ledger.items)
    for local_index, (attribute, value) in enumerate(facts):
        entities = _entity_candidates(arguments, value)
        payload = {
            "entities": entities,
            "attribute": attribute,
            "value": value,
            "source_tool": tool_name,
            "source_arguments": arguments,
            "execution_status": execution_status,
        }
        fingerprint = stable_fingerprint(payload)
        key = (entities, attribute)
        prior_values = values_by_key.get(key, set())
        conflict = bool(prior_values and value not in prior_values)
        new_items.append(
            EvidenceItem(
                item_id=f"evidence-{step_index:03d}-{local_index:03d}",
                entity_candidates=entities,
                attribute=str(attribute),
                value=str(value),
                source_tool=str(tool_name),
                source_arguments=arguments,
                step_index=step_index,
                execution_status=execution_status,
                goal_overlap=_goal_overlap(goal, str(attribute), str(value)),
                argument_link_status=_link_status(argument_link_resolution),
                novelty=(
                    "duplicate" if fingerprint in existing_fingerprints else "new"
                ),
                conflict_status="conflict" if conflict else "none",
                state_provenance="mutating" if state_changed else "read_only",
                content_fingerprint=fingerprint,
            )
        )
        existing_fingerprints.add(fingerprint)
        values_by_key.setdefault(key, set()).add(value)
    return EvidenceLedger(items=tuple(new_items))


def evidence_item_text(item: EvidenceItem) -> str:
    """Serialize only causally observed semantic fields for feature hashing."""

    arguments = json.dumps(
        item.source_arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"entity {' '.join(item.entity_candidates)} attribute {item.attribute} "
        f"value {item.value} source {item.source_tool} arguments {arguments} "
        f"status {item.execution_status} link {item.argument_link_status} "
        f"goal_overlap {item.goal_overlap:.6f} "
        f"novelty {item.novelty} conflict {item.conflict_status} "
        f"provenance {item.state_provenance}"
    )


def ledger_text(ledger: EvidenceLedger) -> str:
    return "\n".join(evidence_item_text(item) for item in ledger.items)


def ledger_length_features(ledger: EvidenceLedger) -> dict[str, float]:
    text = ledger_text(ledger)
    return {
        "item_count": float(len(ledger.items)),
        "character_count": float(len(text)),
        "token_count": float(len(tokenize(text))),
        "error_item_count": float(
            sum(item.execution_status == "error" for item in ledger.items)
        ),
        "duplicate_item_count": float(
            sum(item.novelty == "duplicate" for item in ledger.items)
        ),
        "conflict_item_count": float(
            sum(item.conflict_status == "conflict" for item in ledger.items)
        ),
        "mean_goal_overlap": (
            sum(item.goal_overlap for item in ledger.items) / len(ledger.items)
            if ledger.items
            else 0.0
        ),
    }
