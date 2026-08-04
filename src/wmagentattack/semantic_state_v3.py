"""Causal, label-blind semantic state for the hybrid AgentDojo world model.

The state is built only from the trusted goal, legal tool interface, executed
victim actions, observed execution receipts, and entity-preserving ledger
records.  Hidden simulator deltas, expert paths, future calls, and final labels
are deliberately outside the schema.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_state import canonical_json_value, stable_fingerprint


SEMANTIC_STATE_V3_SCHEMA_VERSION = "wmagentattack.semantic_state.v3"

_ALLOWED_FEATURE_KEYS = {
    "trusted_goal",
    "track",
    "prefix_index",
    "legal_tools",
    "last_action",
    "last_observation",
    "execution_receipt",
    "causal_state_summary",  # accepted for compatibility but never consumed
    "ledger_v2",
}
_FORBIDDEN_KEYS = {
    "factorized",
    "utility",
    "security",
    "task_success",
    "attack_success",
    "policy_violation",
    "reward",
    "score",
    "loss",
    "label",
    "labels",
    "targets",
    "checker",
    "checker_result",
    "ground_truth",
    "ground_truth_output",
    "required_calls",
    "proof_contract",
    "expert_calls",
    "expert_path",
    "future_calls",
    "future_observations",
    "final_report",
    "final_output",
    "outcome_labels_present",
}
_RUNTIME_ID_KEYS = {
    "record_id",
    "record_index",
    "fact_id",
    "conflict_id",
    "episode_id",
    "arguments_fingerprint",
    "observation_fingerprint",
    "resolved_entity_id",
    "provisional_entity_id",
    "entity_id",
    "left_fact_id",
    "right_fact_id",
}
_HIDDEN_ORACLE_KEYS = {
    "state_provenance",
    "state_changed",
    "canonical_state_delta",
    "causal_state_summary",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_MONEY_RE = re.compile(
    r"(?:[$€£]\s*\d+(?:[,.]\d+)*)|(?:\b\d+(?:[,.]\d+)*\s*(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)",
    flags=re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,80})['\"]")

_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "you",
}
_OPERATION_TERMS = {
    "add",
    "book",
    "cancel",
    "check",
    "compare",
    "create",
    "delete",
    "edit",
    "find",
    "get",
    "list",
    "look",
    "read",
    "remove",
    "report",
    "reserve",
    "schedule",
    "search",
    "send",
    "share",
    "show",
    "summarize",
    "update",
}
_LOGIC_TERMS = {
    "after",
    "all",
    "any",
    "before",
    "between",
    "both",
    "cheapest",
    "each",
    "every",
    "exactly",
    "highest",
    "if",
    "least",
    "less",
    "lowest",
    "maximum",
    "minimum",
    "more",
    "most",
    "only",
    "than",
    "then",
    "unless",
    "when",
}


class TypedMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["email", "date", "time", "money", "quoted", "number"]
    value: str


class GoalSemanticFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_goal: str
    fact_terms: tuple[str, ...]
    operation_terms: tuple[str, ...]
    logic_terms: tuple[str, ...]
    typed_mentions: tuple[TypedMention, ...]
    has_condition: bool
    has_comparison: bool
    requires_set_coverage: bool
    requires_uniqueness: bool


class SemanticAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: Any
    kind: str


class SemanticEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_position: int = Field(ge=0)
    call_index: int = Field(ge=0)
    entity_type: str
    entity_key: dict[str, Any]
    entity_candidates: tuple[dict[str, Any], ...]
    link_status: Literal["UNIQUE", "AMBIGUOUS", "UNLINKED"]
    attributes: tuple[SemanticAttribute, ...]
    context: dict[str, Any]
    source_tool: str
    source_arguments: dict[str, Any]
    execution_status: Literal["success", "error"]
    matched_goal_terms: tuple[str, ...]


class SemanticConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute_name: str
    reason: str


class ToolExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    tool_name: str
    execution_status: Literal["success", "error"]


class ExecutionSemanticSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    last_action: dict[str, Any]
    last_status: str
    last_error_type: str | None
    last_output_type: str | None
    observation_token_count: int = Field(ge=0)
    observation_line_count: int = Field(ge=0)
    observation_has_error_lexeme: bool
    history: tuple[ToolExecutionEvent, ...]
    tool_counts: dict[str, int]
    cumulative_errors: int = Field(ge=0)
    consecutive_errors: int = Field(ge=0)
    repeated_last_tool_count: int = Field(ge=0)


class GoalEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_fact_terms: tuple[str, ...]
    unmatched_fact_terms: tuple[str, ...]
    observed_entity_types: tuple[str, ...]
    observed_attribute_names: tuple[str, ...]
    unique_entity_records: int = Field(ge=0)
    ambiguous_entity_records: int = Field(ge=0)
    unlinked_entity_records: int = Field(ge=0)
    conflict_count: int = Field(ge=0)


class StructuredSemanticStateV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["wmagentattack.semantic_state.v3"] = (
        SEMANTIC_STATE_V3_SCHEMA_VERSION
    )
    policy_track: str
    step_index: int = Field(ge=0)
    goal: GoalSemanticFrame
    legal_actions: tuple[str, ...]
    execution: ExecutionSemanticSummary
    evidence_records: tuple[SemanticEvidenceRecord, ...]
    conflicts: tuple[SemanticConflict, ...]
    goal_evidence: GoalEvidenceSummary

    @model_validator(mode="after")
    def validate_prefix_causality(self):
        call_indices = [row.call_index for row in self.execution.history]
        record_indices = [row.call_index for row in self.evidence_records]
        if any(index >= self.step_index for index in (*call_indices, *record_indices)):
            raise ValueError("semantic state contains an event at or after its prefix")
        if call_indices != sorted(call_indices):
            raise ValueError("execution history must remain chronologically ordered")
        return self


def _normalized_text(value: str) -> str:
    return " ".join(str(value).split())


def _terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        value = json.dumps(
            canonical_json_value(value), ensure_ascii=False, sort_keys=True
        )
    expanded = value.replace("_", " ").replace("-", " ")
    return tuple(token.lower() for token in _WORD_RE.findall(expanded))


def _typed_mentions(goal: str) -> tuple[TypedMention, ...]:
    rows: set[tuple[str, str]] = set()
    for kind, pattern in (
        ("email", _EMAIL_RE),
        ("date", _DATE_RE),
        ("time", _TIME_RE),
        ("money", _MONEY_RE),
        ("quoted", _QUOTED_RE),
    ):
        for match in pattern.findall(goal):
            rows.add((kind, _normalized_text(match).lower()))
    occupied_values = {value for _, value in rows}
    for token in _terms(goal):
        if token.replace(".", "", 1).isdigit() and token not in occupied_values:
            rows.add(("number", token))
    return tuple(TypedMention(kind=kind, value=value) for kind, value in sorted(rows))


def build_goal_semantic_frame(trusted_goal: str) -> GoalSemanticFrame:
    normalized = _normalized_text(trusted_goal)
    tokens = tuple(dict.fromkeys(_terms(normalized)))
    operations = tuple(sorted(set(tokens) & _OPERATION_TERMS))
    logic = tuple(sorted(set(tokens) & _LOGIC_TERMS))
    facts = tuple(
        sorted(
            {
                token
                for token in tokens
                if token not in _STOP_TERMS
                and token not in _OPERATION_TERMS
                and token not in _LOGIC_TERMS
                and (len(token) > 1 or token.isdigit())
            }
        )
    )
    token_set = set(tokens)
    return GoalSemanticFrame(
        normalized_goal=normalized,
        fact_terms=facts,
        operation_terms=operations,
        logic_terms=logic,
        typed_mentions=_typed_mentions(normalized),
        has_condition=bool(token_set & {"if", "when", "unless", "then"}),
        has_comparison=bool(
            token_set
            & {
                "compare",
                "more",
                "less",
                "than",
                "highest",
                "lowest",
                "cheapest",
                "minimum",
                "maximum",
            }
        ),
        requires_set_coverage=bool(
            token_set & {"all", "every", "each", "both", "any"}
        ),
        requires_uniqueness=bool(token_set & {"only", "unique", "uniquely", "exactly"}),
    )


def _find_forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            child = (*path, key)
            if (
                lowered in _FORBIDDEN_KEYS
                or lowered.startswith("final_")
                or lowered.startswith("future_")
                or lowered.startswith("expert_")
                or lowered.endswith("_label")
                or lowered.endswith("_target")
            ):
                findings.append(".".join(child))
            findings.extend(_find_forbidden_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            findings.extend(_find_forbidden_paths(item, (*path, str(index))))
    return sorted(set(findings))


def _strip_nonsemantic_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_nonsemantic_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RUNTIME_ID_KEYS
            and str(key) not in _HIDDEN_ORACLE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_strip_nonsemantic_metadata(item) for item in value]
    return canonical_json_value(value)


def _build_evidence_records(
    ledger: Mapping[str, Any], *, step_index: int, goal_terms: set[str]
) -> tuple[SemanticEvidenceRecord, ...]:
    rows = []
    for position, raw_record in enumerate(ledger.get("records", ())):
        record = _strip_nonsemantic_metadata(raw_record)
        call_index = int(record.get("call_index", -1))
        if call_index < 0 or call_index >= step_index:
            raise ValueError("ledger record violates prefix causality")
        attributes = tuple(
            SemanticAttribute(
                name=str(attribute["name"]),
                value=canonical_json_value(attribute.get("value")),
                kind=str(attribute.get("kind", "UNKNOWN")),
            )
            for attribute in record.get("attributes", ())
        )
        semantic_content = {
            "entity_type": record.get("entity_type"),
            "entity_key": record.get("entity_key", {}),
            "entity_candidates": record.get("entity_candidates", ()),
            "attributes": [row.model_dump(mode="json") for row in attributes],
            "context": record.get("context", {}),
            "source_tool": record.get("source_tool"),
            "source_arguments": record.get("source_arguments", {}),
        }
        matched = tuple(sorted(goal_terms & set(_terms(semantic_content))))
        rows.append(
            SemanticEvidenceRecord(
                record_position=position,
                call_index=call_index,
                entity_type=str(record.get("entity_type", "unknown")),
                entity_key=canonical_json_value(record.get("entity_key", {})),
                entity_candidates=tuple(
                    canonical_json_value(candidate)
                    for candidate in record.get("entity_candidates", ())
                ),
                link_status=str(record.get("link_status", "UNLINKED")),
                attributes=attributes,
                context=canonical_json_value(record.get("context", {})),
                source_tool=str(record.get("source_tool", "<UNKNOWN>")),
                source_arguments=canonical_json_value(
                    record.get("source_arguments", {})
                ),
                execution_status=str(record.get("execution_status", "error")),
                matched_goal_terms=matched,
            )
        )
    return tuple(rows)


def _build_execution_summary(
    features: Mapping[str, Any], ledger: Mapping[str, Any], *, step_index: int
) -> ExecutionSemanticSummary:
    history = tuple(
        ToolExecutionEvent(
            call_index=int(row["call_index"]),
            tool_name=str(row["tool_name"]),
            execution_status=str(row["execution_status"]),
        )
        for row in ledger.get("execution_receipts", ())
    )
    if any(row.call_index >= step_index for row in history):
        raise ValueError("execution receipt violates prefix causality")
    counts = Counter(row.tool_name for row in history)
    cumulative_errors = sum(row.execution_status == "error" for row in history)
    consecutive_errors = 0
    for row in reversed(history):
        if row.execution_status != "error":
            break
        consecutive_errors += 1
    last_action = _strip_nonsemantic_metadata(features["last_action"])
    last_tool = str(last_action.get("function", "<START>"))
    receipt = _strip_nonsemantic_metadata(features["execution_receipt"])
    observation = str(features.get("last_observation", ""))
    observation_terms = _terms(observation)
    return ExecutionSemanticSummary(
        last_action=canonical_json_value(last_action),
        last_status=str(receipt.get("status", "unknown")),
        last_error_type=(
            None
            if receipt.get("error_type") in (None, "")
            else str(receipt.get("error_type"))
        ),
        last_output_type=(
            None
            if receipt.get("output_type") in (None, "")
            else str(receipt.get("output_type"))
        ),
        observation_token_count=len(observation_terms),
        observation_line_count=0 if not observation else len(observation.splitlines()),
        observation_has_error_lexeme=bool(
            set(observation_terms) & {"error", "failed", "failure", "invalid", "exception"}
        ),
        history=history,
        tool_counts=dict(sorted(counts.items())),
        cumulative_errors=cumulative_errors,
        consecutive_errors=consecutive_errors,
        repeated_last_tool_count=counts.get(last_tool, 0),
    )


def build_structured_semantic_state_v3(
    features: Mapping[str, Any],
) -> StructuredSemanticStateV3:
    """Build one semantic state from a positive whitelist of causal features."""

    unknown = sorted(set(features) - _ALLOWED_FEATURE_KEYS)
    if unknown:
        raise ValueError(f"unknown semantic-state feature fields: {unknown}")
    missing = sorted(
        {
            "trusted_goal",
            "track",
            "prefix_index",
            "legal_tools",
            "last_action",
            "last_observation",
            "execution_receipt",
            "ledger_v2",
        }
        - set(features)
    )
    if missing:
        raise ValueError(f"missing semantic-state feature fields: {missing}")
    forbidden = _find_forbidden_paths(features)
    if forbidden:
        raise ValueError(f"outcome/future/expert leakage: {forbidden}")

    step_index = int(features["prefix_index"])
    goal = build_goal_semantic_frame(str(features["trusted_goal"]))
    goal_terms = set(goal.fact_terms)
    ledger = _strip_nonsemantic_metadata(features["ledger_v2"])
    records = _build_evidence_records(
        ledger, step_index=step_index, goal_terms=goal_terms
    )
    conflicts = tuple(
        SemanticConflict(
            attribute_name=str(row.get("attribute_name", "unknown")),
            reason=str(row.get("reason", "unknown")),
        )
        for row in ledger.get("conflicts", ())
    )
    matched = tuple(
        sorted({term for record in records for term in record.matched_goal_terms})
    )
    link_counts = Counter(record.link_status for record in records)
    evidence = GoalEvidenceSummary(
        matched_fact_terms=matched,
        unmatched_fact_terms=tuple(sorted(goal_terms - set(matched))),
        observed_entity_types=tuple(sorted({row.entity_type for row in records})),
        observed_attribute_names=tuple(
            sorted({attribute.name for row in records for attribute in row.attributes})
        ),
        unique_entity_records=link_counts.get("UNIQUE", 0),
        ambiguous_entity_records=link_counts.get("AMBIGUOUS", 0),
        unlinked_entity_records=link_counts.get("UNLINKED", 0),
        conflict_count=len(conflicts),
    )
    return StructuredSemanticStateV3(
        policy_track=str(features["track"]),
        step_index=step_index,
        goal=goal,
        legal_actions=tuple(sorted(str(row) for row in features["legal_tools"])),
        execution=_build_execution_summary(
            features, ledger, step_index=step_index
        ),
        evidence_records=records,
        conflicts=conflicts,
        goal_evidence=evidence,
    )


def semantic_state_v3_payload(features: Mapping[str, Any]) -> dict[str, Any]:
    return build_structured_semantic_state_v3(features).model_dump(mode="json")


def semantic_state_v3_fingerprint(features: Mapping[str, Any]) -> str:
    return stable_fingerprint(semantic_state_v3_payload(features))


def find_semantic_state_v3_leakage(value: Any) -> tuple[str, ...]:
    """Return forbidden or oracle-only keys retained by an emitted state."""

    findings = _find_forbidden_paths(value)

    def walk(item: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = (*path, key)
                if key in _RUNTIME_ID_KEYS or key in _HIDDEN_ORACLE_KEYS:
                    findings.append(".".join(child_path))
                walk(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, child in enumerate(item):
                walk(child, (*path, str(index)))

    walk(value)
    return tuple(sorted(set(findings)))
