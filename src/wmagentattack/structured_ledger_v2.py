"""Structured, victim-observed evidence ledger for AgentDojo runtime outputs."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .clean_state_instrumentation import canonical_call_signature
from .decision_state import canonical_json_value, stable_fingerprint
from .state_storage_v2 import VisibilityScope


STRUCTURED_LEDGER_SCHEMA_VERSION = "wmagentattack.structured_ledger.v2"
_FIELD_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_PRICE_RANGE_RE = re.compile(
    r"Price\s+range\s*:\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_FLIGHT_RE = re.compile(
    r"^Airline:\s*(.*?),\s*Flight Number:\s*(.*?),\s*"
    r"Departure Time:\s*(.*?),\s*Arrival Time:\s*(.*?),\s*"
    r"Price:\s*(-?\d+(?:\.\d+)?),\s*Contact Information:\s*(.*)$"
)


class AttributeKind(str, Enum):
    SINGLE_VALUED = "SINGLE_VALUED"
    SET_VALUED = "SET_VALUED"
    TIME_SCOPED = "TIME_SCOPED"
    RANGE = "RANGE"


class AdapterMode(str, Enum):
    VALUE = "VALUE"
    SCALAR_LIST = "SCALAR_LIST"
    USER_FIELDS = "USER_FIELDS"
    NAME_LIST_TEXT = "NAME_LIST_TEXT"
    ENTITY_MAP = "ENTITY_MAP"
    PRICE_RANGE_MAP = "PRICE_RANGE_MAP"
    RATING_REVIEWS_MAP = "RATING_REVIEWS_MAP"
    FLIGHT_LINES = "FLIGHT_LINES"
    OBJECT = "OBJECT"
    OBJECT_LIST = "OBJECT_LIST"
    MUTATION_ACK = "MUTATION_ACK"


class ExecutionChannelStatus(str, Enum):
    PROPOSED = "PROPOSED"
    EXECUTED_SUCCESS = "EXECUTED_SUCCESS"
    EXECUTED_ERROR = "EXECUTED_ERROR"
    TERMINAL_UNEXECUTED = "TERMINAL_UNEXECUTED"
    CENSORED = "CENSORED"


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    mode: AdapterMode
    entity_type: str
    fixed_entity_key: dict[str, Any] | None = None
    text_prefix: str | None = None
    attribute_name: str | None = None
    attribute_kind: AttributeKind | None = None
    context_argument_fields: tuple[str, ...] = ()
    entity_argument_fields: tuple[str, ...] = ()
    entity_key_fields: tuple[str, ...] = ()
    set_fields: tuple[str, ...] = ()
    time_scoped_fields: tuple[str, ...] = ()
    split_set_values: bool = False


class AdapterRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    benchmark_version: str
    suite: str
    adapters: dict[str, AdapterSpec]


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_key: dict[str, Any]
    score: float = Field(ge=0.0, le=1.0)


class StructuredAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    name: str
    value: Any
    kind: AttributeKind


class ItemAttributeInput(BaseModel):
    """Label-blind attribute supplied by an item-level entity linker."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: Any
    kind: AttributeKind


class StructuredEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STRUCTURED_LEDGER_SCHEMA_VERSION
    record_id: str
    entity_type: str
    entity_key: dict[str, Any]
    resolved_entity_id: str | None
    provisional_entity_id: str | None
    entity_candidates: tuple[EntityCandidate, ...]
    link_status: Literal["UNIQUE", "AMBIGUOUS", "UNLINKED"]
    attributes: tuple[StructuredAttribute, ...]
    context: dict[str, Any]
    source_tool: str
    source_arguments: dict[str, Any]
    call_index: int = Field(ge=0)
    record_index: int = Field(ge=0)
    execution_status: Literal["success", "error"]
    observation_scope: Literal[VisibilityScope.VICTIM_OBSERVED] = (
        VisibilityScope.VICTIM_OBSERVED
    )
    state_provenance: Literal["read_only", "mutating"]
    outcome_labels_present: bool = False

    @model_validator(mode="after")
    def validate_link_state(self):
        if self.link_status == "UNIQUE":
            if self.resolved_entity_id is None or len(self.entity_candidates) != 1:
                raise ValueError("unique record requires one resolved entity candidate")
            if self.provisional_entity_id is not None:
                raise ValueError("unique record cannot have a provisional entity")
        elif self.link_status == "AMBIGUOUS":
            if self.resolved_entity_id is not None or len(self.entity_candidates) < 2:
                raise ValueError("ambiguous record requires multiple unresolved candidates")
            if self.provisional_entity_id is None:
                raise ValueError("ambiguous record requires a local provisional ID")
        else:
            if self.resolved_entity_id is not None or self.entity_candidates:
                raise ValueError("unlinked record cannot have resolved candidates")
            if self.provisional_entity_id is None:
                raise ValueError("unlinked record requires a local provisional ID")
        return self


class ConflictRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    entity_id: str
    attribute_name: str
    context_fingerprint: str
    left_fact_id: str
    right_fact_id: str
    reason: Literal["incompatible_single_value", "non_overlapping_range"]


class ExecutionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    call_index: int = Field(ge=0)
    tool_name: str
    arguments_fingerprint: str
    observation_fingerprint: str
    execution_status: Literal["success", "error"]


class StructuredEvidenceLedgerV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STRUCTURED_LEDGER_SCHEMA_VERSION
    records: tuple[StructuredEvidenceRecord, ...] = ()
    conflicts: tuple[ConflictRelation, ...] = ()
    execution_receipts: tuple[ExecutionReceipt, ...] = ()
    outcome_labels_present: bool = False


class LedgerUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ledger: StructuredEvidenceLedgerV2
    channel_status: ExecutionChannelStatus
    added_records: int = Field(ge=0)
    added_conflicts: int = Field(ge=0)
    ignored_without_observation: bool


class _AttributeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: Any
    kind: AttributeKind


class _RecordDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_key: dict[str, Any] = Field(default_factory=dict)
    candidate_entity_keys: tuple[dict[str, Any], ...] = ()
    attributes: tuple[_AttributeDraft, ...]
    context: dict[str, Any] = Field(default_factory=dict)


def load_adapter_registry(path: Path) -> AdapterRegistry:
    return AdapterRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def _field_name(value: str) -> str:
    return _FIELD_TOKEN_RE.sub("_", str(value).lower()).strip("_") or "value"


def _context(spec: AdapterSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: canonical_json_value(arguments.get(field))
        for field in spec.context_argument_fields
        if field in arguments
    }


def _set_value(value: Any, *, split_strings: bool) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
    elif split_strings and isinstance(value, str):
        rows = [part.strip() for part in value.split(",") if part.strip()]
    else:
        rows = [value]
    canonical = [canonical_json_value(row) for row in rows]
    keyed = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in canonical}
    return [keyed[key] for key in sorted(keyed)]


def _value_entity_key(
    spec: AdapterSpec, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    if spec.fixed_entity_key is not None:
        return canonical_json_value(spec.fixed_entity_key)
    return {
        field: canonical_json_value(arguments[field])
        for field in spec.entity_argument_fields
        if field in arguments
    }


def _value_draft(
    spec: AdapterSpec, value: Any, arguments: Mapping[str, Any]
) -> _RecordDraft:
    if spec.attribute_name is None:
        raise ValueError("VALUE requires attribute_name")
    return _RecordDraft(
        entity_key=_value_entity_key(spec, arguments),
        attributes=(
            _AttributeDraft(
                name=_field_name(spec.attribute_name),
                value=canonical_json_value(value),
                kind=spec.attribute_kind or AttributeKind.SINGLE_VALUED,
            ),
        ),
        context=_context(spec, arguments),
    )


def _scalar_list_drafts(
    spec: AdapterSpec, value: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("SCALAR_LIST requires a non-string sequence")
    key_name = _field_name(spec.attribute_name or "name")
    rows = []
    for item in value:
        canonical_item = canonical_json_value(item)
        if isinstance(canonical_item, (Mapping, list)):
            raise TypeError("SCALAR_LIST items must be scalar")
        rows.append(
            _RecordDraft(
                entity_key={key_name: canonical_item},
                attributes=(
                    _AttributeDraft(
                        name="observed",
                        value=True,
                        kind=AttributeKind.SINGLE_VALUED,
                    ),
                ),
                context=_context(spec, arguments),
            )
        )
    return rows


def _entity_map_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if not isinstance(output, Mapping):
        raise TypeError(f"{spec.mode.value} requires a mapping output")
    if spec.attribute_name is None or spec.attribute_kind is None:
        raise ValueError("ENTITY_MAP adapter is missing attribute metadata")
    rows = []
    for entity_name, raw_value in output.items():
        value = (
            _set_value(raw_value, split_strings=spec.split_set_values)
            if spec.attribute_kind == AttributeKind.SET_VALUED
            else canonical_json_value(raw_value)
        )
        rows.append(
            _RecordDraft(
                entity_key={"name": str(entity_name)},
                attributes=(
                    _AttributeDraft(
                        name=spec.attribute_name,
                        value=value,
                        kind=spec.attribute_kind,
                    ),
                ),
                context=_context(spec, arguments),
            )
        )
    return rows


def _name_list_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if not isinstance(output, str):
        raise TypeError("NAME_LIST_TEXT requires a string output")
    if spec.attribute_name is None or spec.attribute_kind is None:
        raise ValueError("NAME_LIST_TEXT adapter is missing attribute metadata")
    first, *remaining = output.splitlines()
    if ":" not in first:
        raise ValueError(f"name-list output lacks a prefix separator: {output!r}")
    prefix, first_name = first.split(":", 1)
    if spec.text_prefix and not prefix.strip().startswith(spec.text_prefix.rstrip(":")):
        raise ValueError(f"unexpected name-list prefix for {spec.text_prefix!r}")
    names = [first_name.strip(), *(line.strip() for line in remaining)]
    return [
        _RecordDraft(
            entity_key={"name": name},
            attributes=(
                _AttributeDraft(
                    name=spec.attribute_name,
                    value=True,
                    kind=spec.attribute_kind,
                ),
            ),
            context=_context(spec, arguments),
        )
        for name in names
        if name
    ]


def _price_range_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if not isinstance(output, Mapping):
        raise TypeError("PRICE_RANGE_MAP requires a mapping output")
    rows = []
    for name, value in output.items():
        match = _PRICE_RANGE_RE.fullmatch(str(value).strip())
        if match is None:
            raise ValueError(f"invalid price range for {name}: {value!r}")
        rows.append(
            _RecordDraft(
                entity_key={"name": str(name)},
                attributes=(
                    _AttributeDraft(
                        name="price_range",
                        value={"min": float(match.group(1)), "max": float(match.group(2))},
                        kind=AttributeKind.RANGE,
                    ),
                ),
                context=_context(spec, arguments),
            )
        )
    return rows


def _rating_review_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if not isinstance(output, Mapping):
        raise TypeError("RATING_REVIEWS_MAP requires a mapping output")
    rows = []
    for name, value in output.items():
        lines = str(value).splitlines()
        if not lines or not lines[0].lower().startswith("rating:"):
            raise ValueError(f"invalid rating/review output for {name}")
        rating = float(lines[0].split(":", 1)[1].strip())
        review_lines = lines[1:]
        if review_lines and review_lines[0].lower().startswith("reviews:"):
            first_review = review_lines[0].split(":", 1)[1].strip()
            review_lines = ([first_review] if first_review else []) + review_lines[1:]
        reviews = _set_value(review_lines, split_strings=False)
        rows.append(
            _RecordDraft(
                entity_key={"name": str(name)},
                attributes=(
                    _AttributeDraft(
                        name="rating", value=rating, kind=AttributeKind.SINGLE_VALUED
                    ),
                    _AttributeDraft(
                        name="reviews", value=reviews, kind=AttributeKind.SET_VALUED
                    ),
                ),
                context=_context(spec, arguments),
            )
        )
    return rows


def _flight_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    if not isinstance(output, str):
        raise TypeError("FLIGHT_LINES requires a string output")
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        match = _FLIGHT_RE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"invalid flight output line: {line!r}")
        airline, number, departure, arrival, price, contact = match.groups()
        rows.append(
            _RecordDraft(
                entity_key={"airline": airline, "flight_number": number},
                attributes=(
                    _AttributeDraft(
                        name="departure_time",
                        value=departure,
                        kind=AttributeKind.TIME_SCOPED,
                    ),
                    _AttributeDraft(
                        name="arrival_time", value=arrival, kind=AttributeKind.TIME_SCOPED
                    ),
                    _AttributeDraft(
                        name="price", value=float(price), kind=AttributeKind.SINGLE_VALUED
                    ),
                    _AttributeDraft(
                        name="contact_information",
                        value=contact,
                        kind=AttributeKind.SINGLE_VALUED,
                    ),
                ),
                context=_context(spec, arguments),
            )
        )
    return rows


def _object_draft(
    spec: AdapterSpec, value: Any, arguments: Mapping[str, Any]
) -> _RecordDraft:
    payload = canonical_json_value(value)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{spec.mode.value} requires object records")
    entity_key = {
        field: payload[field] for field in spec.entity_key_fields if field in payload
    }
    attributes = []
    for raw_name, raw_value in payload.items():
        if raw_name in spec.entity_key_fields:
            continue
        name = _field_name(raw_name)
        if raw_name in spec.set_fields:
            kind = AttributeKind.SET_VALUED
            normalized = _set_value(raw_value, split_strings=False)
        elif raw_name in spec.time_scoped_fields:
            kind = AttributeKind.TIME_SCOPED
            normalized = canonical_json_value(raw_value)
        else:
            kind = AttributeKind.SINGLE_VALUED
            normalized = canonical_json_value(raw_value)
        attributes.append(_AttributeDraft(name=name, value=normalized, kind=kind))
    return _RecordDraft(
        entity_key=entity_key,
        attributes=tuple(attributes),
        context=_context(spec, arguments),
    )


def _mutation_draft(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> _RecordDraft:
    key = {
        field: canonical_json_value(arguments[field])
        for field in spec.entity_argument_fields
        if field in arguments
    }
    if len(key) != len(spec.entity_argument_fields):
        raise ValueError("mutation acknowledgement lacks entity arguments")
    return _RecordDraft(
        entity_key=key,
        attributes=(
            _AttributeDraft(
                name="execution_acknowledged",
                value=True,
                kind=AttributeKind.SINGLE_VALUED,
            ),
            _AttributeDraft(
                name="observation",
                value=canonical_json_value(output),
                kind=AttributeKind.SINGLE_VALUED,
            ),
        ),
        context=_context(spec, arguments),
    )


def _extract_drafts(
    spec: AdapterSpec, output: Any, arguments: Mapping[str, Any]
) -> list[_RecordDraft]:
    canonical_output = canonical_json_value(output)
    if spec.mode == AdapterMode.VALUE:
        return [_value_draft(spec, canonical_output, arguments)]
    if spec.mode == AdapterMode.SCALAR_LIST:
        return _scalar_list_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.USER_FIELDS:
        if not isinstance(canonical_output, Mapping) or spec.fixed_entity_key is None:
            raise TypeError("USER_FIELDS requires a mapping and fixed entity key")
        return [
            _RecordDraft(
                entity_key=spec.fixed_entity_key,
                attributes=tuple(
                    _AttributeDraft(
                        name=_field_name(name),
                        value=canonical_json_value(value),
                        kind=AttributeKind.SINGLE_VALUED,
                    )
                    for name, value in canonical_output.items()
                ),
                context=_context(spec, arguments),
            )
        ]
    if spec.mode == AdapterMode.NAME_LIST_TEXT:
        return _name_list_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.ENTITY_MAP:
        return _entity_map_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.PRICE_RANGE_MAP:
        return _price_range_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.RATING_REVIEWS_MAP:
        return _rating_review_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.FLIGHT_LINES:
        return _flight_drafts(spec, canonical_output, arguments)
    if spec.mode == AdapterMode.OBJECT:
        return [_object_draft(spec, canonical_output, arguments)]
    if spec.mode == AdapterMode.OBJECT_LIST:
        if not isinstance(canonical_output, list):
            raise TypeError("OBJECT_LIST requires a list output")
        return [_object_draft(spec, row, arguments) for row in canonical_output]
    if spec.mode == AdapterMode.MUTATION_ACK:
        return [_mutation_draft(spec, canonical_output, arguments)]
    raise AssertionError(f"unhandled adapter mode: {spec.mode}")


def _provisional_id(episode_id: str, call_index: int, record_index: int) -> str:
    episode_hash = stable_fingerprint(episode_id)[:16]
    return f"PROVISIONAL::{episode_hash}::{call_index:03d}::{record_index:03d}"


def _entity_id(entity_type: str, entity_key: Mapping[str, Any]) -> str:
    suffix = stable_fingerprint(
        {"entity_type": entity_type, "entity_key": canonical_json_value(entity_key)}
    )[:24]
    return f"ENTITY::{entity_type}::{suffix}"


def _build_record(
    *,
    spec: AdapterSpec,
    draft: _RecordDraft,
    episode_id: str,
    call_index: int,
    record_index: int,
    tool_name: str,
    arguments: Mapping[str, Any],
    execution_status: Literal["success", "error"],
    state_changed: bool,
) -> StructuredEvidenceRecord:
    record_id = (
        f"record::{stable_fingerprint(episode_id)[:16]}::{call_index:03d}::{record_index:03d}"
    )
    candidate_keys_by_fingerprint = {
        stable_fingerprint(canonical_json_value(key)): canonical_json_value(key)
        for key in draft.candidate_entity_keys
    }
    candidate_keys = tuple(
        candidate_keys_by_fingerprint[key] for key in sorted(candidate_keys_by_fingerprint)
    )
    candidates = tuple(
        EntityCandidate(
            entity_id=_entity_id(spec.entity_type, key),
            entity_key=canonical_json_value(key),
            score=1.0 / len(candidate_keys),
        )
        for key in candidate_keys
    )
    if draft.entity_key:
        resolved = _entity_id(spec.entity_type, draft.entity_key)
        candidates = (
            EntityCandidate(
                entity_id=resolved,
                entity_key=canonical_json_value(draft.entity_key),
                score=1.0,
            ),
        )
        link_status: Literal["UNIQUE", "AMBIGUOUS", "UNLINKED"] = "UNIQUE"
        provisional = None
    elif len(candidates) == 1:
        resolved = candidates[0].entity_id
        link_status = "UNIQUE"
        provisional = None
    elif len(candidates) >= 2:
        resolved = None
        link_status = "AMBIGUOUS"
        provisional = _provisional_id(episode_id, call_index, record_index)
    else:
        resolved = None
        candidates = ()
        link_status = "UNLINKED"
        provisional = _provisional_id(episode_id, call_index, record_index)
    attributes = tuple(
        StructuredAttribute(
            fact_id=f"{record_id}::fact::{index:03d}",
            name=attribute.name,
            value=canonical_json_value(attribute.value),
            kind=attribute.kind,
        )
        for index, attribute in enumerate(draft.attributes)
    )
    return StructuredEvidenceRecord(
        record_id=record_id,
        entity_type=spec.entity_type,
        entity_key=canonical_json_value(
            draft.entity_key
            if draft.entity_key
            else candidates[0].entity_key
            if len(candidates) == 1
            else {}
        ),
        resolved_entity_id=resolved,
        provisional_entity_id=provisional,
        entity_candidates=candidates,
        link_status=link_status,
        attributes=attributes,
        context=canonical_json_value(draft.context),
        source_tool=tool_name,
        source_arguments=canonical_json_value(dict(arguments)),
        call_index=call_index,
        record_index=record_index,
        execution_status=execution_status,
        state_provenance="mutating" if state_changed else "read_only",
    )


def build_item_linkage_record(
    *,
    family: str,
    entity_type: str,
    episode_id: str,
    call_index: int,
    record_index: int,
    source_tool: str,
    source_arguments: Mapping[str, Any],
    attributes: Sequence[ItemAttributeInput],
    entity_key: Mapping[str, Any] | None = None,
    candidate_entity_keys: Sequence[Mapping[str, Any]] = (),
    context: Mapping[str, Any] | None = None,
    execution_status: Literal["success", "error"] = "success",
    state_changed: bool = False,
) -> StructuredEvidenceRecord:
    """Build one record from item-local linkage evidence without global fallbacks."""

    if entity_key and candidate_entity_keys:
        raise ValueError("explicit entity key and candidate set are mutually exclusive")
    spec = AdapterSpec(
        family=family,
        mode=AdapterMode.OBJECT,
        entity_type=entity_type,
    )
    draft = _RecordDraft(
        entity_key=canonical_json_value(dict(entity_key or {})),
        candidate_entity_keys=tuple(
            canonical_json_value(dict(candidate))
            for candidate in candidate_entity_keys
        ),
        attributes=tuple(
            _AttributeDraft(name=row.name, value=row.value, kind=row.kind)
            for row in attributes
        ),
        context=canonical_json_value(dict(context or {})),
    )
    return _build_record(
        spec=spec,
        draft=draft,
        episode_id=episode_id,
        call_index=call_index,
        record_index=record_index,
        tool_name=source_tool,
        arguments=source_arguments,
        execution_status=execution_status,
        state_changed=state_changed,
    )


def _error_record(
    *,
    episode_id: str,
    call_index: int,
    tool_name: str,
    arguments: Mapping[str, Any],
    error_type: str | None,
) -> StructuredEvidenceRecord:
    spec = AdapterSpec(
        family="execution", mode=AdapterMode.MUTATION_ACK, entity_type="execution_error"
    )
    draft = _RecordDraft(
        attributes=(
            _AttributeDraft(
                name="error_type",
                value=str(error_type or "unknown_error"),
                kind=AttributeKind.SINGLE_VALUED,
            ),
        )
    )
    return _build_record(
        spec=spec,
        draft=draft,
        episode_id=episode_id,
        call_index=call_index,
        record_index=0,
        tool_name=tool_name,
        arguments=arguments,
        execution_status="error",
        state_changed=False,
    )


def _range_disjoint(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return left != right
    try:
        return float(left["max"]) < float(right["min"]) or float(right["max"]) < float(left["min"])
    except (KeyError, TypeError, ValueError):
        return left != right


def _conflict_reason(
    left: StructuredAttribute, right: StructuredAttribute
) -> Literal["incompatible_single_value", "non_overlapping_range"] | None:
    if left.kind != right.kind or left.name != right.name:
        return None
    if left.value == right.value or left.kind == AttributeKind.SET_VALUED:
        return None
    if left.kind == AttributeKind.RANGE:
        return "non_overlapping_range" if _range_disjoint(left.value, right.value) else None
    return "incompatible_single_value"


def _new_conflicts(
    existing: Sequence[StructuredEvidenceRecord],
    additions: Sequence[StructuredEvidenceRecord],
    prior_conflicts: Sequence[ConflictRelation],
) -> tuple[ConflictRelation, ...]:
    known_ids = {conflict.conflict_id for conflict in prior_conflicts}
    output = []
    prior = list(existing)
    for record in additions:
        if record.resolved_entity_id is None:
            prior.append(record)
            continue
        context_fingerprint = stable_fingerprint(record.context)
        for previous in prior:
            if previous.resolved_entity_id != record.resolved_entity_id:
                continue
            if stable_fingerprint(previous.context) != context_fingerprint:
                continue
            for left in previous.attributes:
                for right in record.attributes:
                    reason = _conflict_reason(left, right)
                    if reason is None:
                        continue
                    ordered = sorted((left.fact_id, right.fact_id))
                    conflict_id = stable_fingerprint(
                        {
                            "entity": record.resolved_entity_id,
                            "attribute": right.name,
                            "context": context_fingerprint,
                            "facts": ordered,
                            "reason": reason,
                        }
                    )
                    if conflict_id in known_ids:
                        continue
                    known_ids.add(conflict_id)
                    output.append(
                        ConflictRelation(
                            conflict_id=conflict_id,
                            entity_id=record.resolved_entity_id,
                            attribute_name=right.name,
                            context_fingerprint=context_fingerprint,
                            left_fact_id=ordered[0],
                            right_fact_id=ordered[1],
                            reason=reason,
                        )
                    )
        prior.append(record)
    return tuple(output)


def update_structured_ledger(
    ledger: StructuredEvidenceLedgerV2,
    registry: AdapterRegistry,
    *,
    episode_id: str,
    call_index: int,
    channel_status: ExecutionChannelStatus,
    tool_name: str,
    arguments: Mapping[str, Any],
    runtime_output: Any = None,
    error_type: str | None = None,
    state_changed: bool = False,
    proposal_signature: str | None = None,
) -> LedgerUpdateResult:
    """Update from one causal execution-channel event and fail closed."""

    if channel_status in {
        ExecutionChannelStatus.PROPOSED,
        ExecutionChannelStatus.TERMINAL_UNEXECUTED,
        ExecutionChannelStatus.CENSORED,
    }:
        return LedgerUpdateResult(
            ledger=ledger,
            channel_status=channel_status,
            added_records=0,
            added_conflicts=0,
            ignored_without_observation=True,
        )
    actual_signature = canonical_call_signature(tool_name, arguments)
    if proposal_signature is not None and proposal_signature != actual_signature:
        raise ValueError("proposal/execution signature mismatch")
    if tool_name not in registry.adapters:
        raise KeyError(f"no structured output adapter for {tool_name}")

    execution_status: Literal["success", "error"] = (
        "error"
        if channel_status == ExecutionChannelStatus.EXECUTED_ERROR
        else "success"
    )
    observation_fingerprint = stable_fingerprint(
        {"output": canonical_json_value(runtime_output), "error_type": error_type}
    )
    receipt = ExecutionReceipt(
        episode_id=episode_id,
        call_index=call_index,
        tool_name=tool_name,
        arguments_fingerprint=stable_fingerprint(canonical_json_value(dict(arguments))),
        observation_fingerprint=observation_fingerprint,
        execution_status=execution_status,
    )
    for existing_receipt in ledger.execution_receipts:
        if (
            existing_receipt.episode_id == episode_id
            and existing_receipt.call_index == call_index
        ):
            if existing_receipt != receipt:
                raise ValueError("same executed call index replayed with different content")
            return LedgerUpdateResult(
                ledger=ledger,
                channel_status=channel_status,
                added_records=0,
                added_conflicts=0,
                ignored_without_observation=False,
            )

    if execution_status == "error":
        records = (
            _error_record(
                episode_id=episode_id,
                call_index=call_index,
                tool_name=tool_name,
                arguments=arguments,
                error_type=error_type,
            ),
        )
    else:
        spec = registry.adapters[tool_name]
        drafts = _extract_drafts(spec, runtime_output, arguments)
        records = tuple(
            _build_record(
                spec=spec,
                draft=draft,
                episode_id=episode_id,
                call_index=call_index,
                record_index=index,
                tool_name=tool_name,
                arguments=arguments,
                execution_status="success",
                state_changed=state_changed,
            )
            for index, draft in enumerate(drafts)
        )
    existing_by_id = {record.record_id: record for record in ledger.records}
    additions = []
    for record in records:
        previous = existing_by_id.get(record.record_id)
        if previous is None:
            additions.append(record)
        elif previous != record:
            raise ValueError("same record ID replayed with different structured content")
    conflicts = _new_conflicts(ledger.records, additions, ledger.conflicts)
    updated = StructuredEvidenceLedgerV2(
        records=(*ledger.records, *additions),
        conflicts=(*ledger.conflicts, *conflicts),
        execution_receipts=(*ledger.execution_receipts, receipt),
    )
    return LedgerUpdateResult(
        ledger=updated,
        channel_status=channel_status,
        added_records=len(additions),
        added_conflicts=len(conflicts),
        ignored_without_observation=False,
    )
