"""Frozen, label-blind event ontology for victim dynamics experiments.

The ontology is deliberately narrower than :class:`~wmagentattack.schema.StepRecord`.
Archival steps contain final outcome labels and raw observations; neither is a
valid victim-dynamics input.  This module keeps only the action, candidate
manifest, normalized argument shape, execution status, and explicitly audited
state/progress fields.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .schema import StepRecord


EVENT_ONTOLOGY_VERSION = "wmagentattack.event.v2.3"
CANDIDATE_POLICY_VERSION = "wmagentattack.candidates.current-state.v1"

EXCLUDED_OUTCOME_FIELDS = frozenset(
    {
        "task_success",
        "attack_success",
        "target_skill_success",
        "policy_violation",
        "risk_level",
        "utility_probability_target",
        "preservation_probability_target",
        "attack_probability_target",
        "joint_success_probability_target",
        "joint_outcome_counts",
        "joint_outcome_dirichlet_alpha",
        "joint_outcome_probability_target",
        "attack_utility_logit_residual_target",
        "skill_output",
        "current_observation",
        "agent_history",
        "untrusted_content",
    }
)


class CandidateSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class FrozenVictimEvent(BaseModel):
    """One label-blind victim action event under the frozen v2.3 ontology."""

    model_config = ConfigDict(extra="forbid")

    ontology_version: str = EVENT_ONTOLOGY_VERSION
    candidate_policy_version: str = CANDIDATE_POLICY_VERSION
    trajectory_id: str
    event_index: int = Field(ge=0)
    tool_schema_id: str
    skill_name: str
    normalized_argument_slots: tuple[str, ...]
    argument_entity_links: tuple[str, ...] | None = None
    tool_execution_status: Literal[
        "success", "error", "invalid_call", "no_tool_finish"
    ]
    canonical_state_delta: dict[str, Any] | None = None
    task_progress_delta: float | None = None
    irreversible_effect: bool | None = None
    termination_reason: Literal[
        "agent_finish", "trace_boundary_unknown"
    ] | None = None
    candidates: tuple[CandidateSkill, ...]
    candidate_source: Literal["initial_manifest", "exact_observed_state"]
    candidate_manifest_fingerprint: str
    unavailable_fields: tuple[str, ...]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_step(step: StepRecord | Mapping[str, Any]) -> StepRecord:
    return step if isinstance(step, StepRecord) else StepRecord.model_validate(step)


def normalize_victim_event(
    step: StepRecord | Mapping[str, Any],
    *,
    is_last_observed_event: bool,
) -> FrozenVictimEvent:
    """Normalize one archive step without copying outcomes or raw text state.

    ``canonical_state_delta``, ``task_progress_delta``, entity links, and
    irreversible-effect annotations are unavailable in the current v2 archive.
    They remain explicit nullable fields so a future exact-simulator adapter can
    populate them without changing event identity or silently inferring labels.
    """

    record = _as_step(step)
    if record.selected_skill == "finish":
        execution_status = "no_tool_finish"
    elif record.selected_skill == "invalid_tool_call":
        execution_status = "invalid_call"
    elif record.tool_error:
        execution_status = "error"
    else:
        execution_status = "success"

    termination_reason = None
    if record.selected_skill == "finish":
        termination_reason = "agent_finish"
    elif is_last_observed_event:
        termination_reason = "trace_boundary_unknown"

    candidates = tuple(
        CandidateSkill(
            name=name,
            description=str(record.candidate_skill_descriptions[name]),
        )
        for name in sorted(record.candidate_skills)
    )
    candidate_payload = [item.model_dump(mode="json") for item in candidates]
    unavailable = (
        "argument_entity_links",
        "canonical_state_delta",
        "task_progress_delta",
        "irreversible_effect",
    )
    return FrozenVictimEvent(
        trajectory_id=record.trajectory_id,
        event_index=record.step_id,
        tool_schema_id=record.selected_tool or record.selected_skill,
        skill_name=record.selected_skill,
        normalized_argument_slots=tuple(
            sorted(str(key) for key in record.skill_arguments)
        ),
        argument_entity_links=None,
        tool_execution_status=execution_status,
        canonical_state_delta=None,
        task_progress_delta=None,
        irreversible_effect=None,
        termination_reason=termination_reason,
        candidates=candidates,
        candidate_source=(
            "initial_manifest" if record.step_id == 0 else "exact_observed_state"
        ),
        candidate_manifest_fingerprint=_stable_hash(candidate_payload),
        unavailable_fields=unavailable,
    )


def ontology_specification() -> dict[str, Any]:
    """Return the immutable experiment-facing ontology specification."""

    return {
        "event_ontology_version": EVENT_ONTOLOGY_VERSION,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "candidate_sources_allowed": ["initial_manifest", "exact_observed_state"],
        "candidate_order": "lexicographic canonical order",
        "observed_or_derived_fields": [
            "tool_schema_id",
            "skill_name",
            "normalized_argument_slots",
            "tool_execution_status",
            "termination_reason",
            "candidates",
            "candidate_source",
            "candidate_manifest_fingerprint",
        ],
        "currently_unavailable_fields": [
            "argument_entity_links",
            "canonical_state_delta",
            "task_progress_delta",
            "irreversible_effect",
        ],
        "outcome_or_raw_text_fields_forbidden": sorted(EXCLUDED_OUTCOME_FIELDS),
        "current_model_consumes": [
            "skill_name",
            "normalized_argument_slots_as_auxiliary_target",
            "candidates",
            "tool_execution_status_as_audit_only",
            "termination_reason_as_audit_only",
        ],
        "current_model_does_not_yet_consume": [
            "candidate_description",
            "argument_entity_links",
            "canonical_state_delta",
            "task_progress_delta",
            "irreversible_effect",
        ],
        "pointer_scoring_status": (
            "candidate-masked tied compositional catalog; not yet a dynamic "
            "description/schema pointer encoder"
        ),
    }


def ontology_fingerprint() -> str:
    return _stable_hash(ontology_specification())
