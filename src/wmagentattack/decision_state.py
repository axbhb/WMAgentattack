"""Label-blind decision states and factorized attacker/victim events.

The original pipeline stored attacker configuration, victim tool choices,
tool outputs, and final checker labels in one trajectory object.  That format
is useful for archival replay but is too easy to misuse as a model input.  The
types in this module make the causal ordering explicit:

1. an attacker intervention is fixed before the victim rollout;
2. the victim emits a tool/argument/stop event;
3. the benchmark simulator applies the exact state transition and checkers.

Only fields available before (or in an explicitly observed prefix of) the
victim rollout are admitted to :class:`CanonicalDecisionState`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import StepRecord


SCHEMA_VERSION = "wmagentattack.decision_state.v1"

_OUTCOME_KEYS = {
    "utility",
    "security",
    "task_success",
    "attack_success",
    "target_skill_success",
    "policy_violation",
    "reward",
    "loss",
    "score",
    "result",
    "results",
    "outcome",
    "outcomes",
    "checker",
    "checker_result",
    "selected_tool",
    "selected_skill",
    "skill_output",
    "tool_output",
    "tool_error",
    "trajectory",
    "trajectories",
    "messages",
}


def canonical_json_value(value: Any) -> Any:
    """Convert a value to deterministic, finite JSON-compatible data."""

    if isinstance(value, BaseModel):
        return canonical_json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical decision state cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_outcome_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """List outcome-like fields present in an arbitrary source payload.

    The builder below uses a positive whitelist, so detected fields are
    dropped rather than copied.  Returning their paths provides an auditable
    warning when a caller accidentally feeds post-rollout metadata.
    """

    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            child_path = (*path, key)
            if (
                lowered in _OUTCOME_KEYS
                or lowered.startswith("final_")
                or lowered.endswith("_probability_target")
                or lowered.endswith("_success_count")
            ):
                findings.append(".".join(child_path))
            findings.extend(find_outcome_paths(item, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            findings.extend(find_outcome_paths(item, (*path, str(index))))
    return sorted(set(findings))


class CleanSolvabilityPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posterior_mean: float = Field(ge=0.0, le=1.0)
    trials: int | None = Field(default=None, ge=1)
    source: str
    source_split: str | None = None


class AttackerIntervention(BaseModel):
    """Static attacker action chosen before a victim rollout."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    family: str
    variant: str
    role: str
    attack_name: str
    injection_task_id: str | None = None
    injection_goal: str | None = None
    target_tool_sequence: list[dict[str, Any]] = Field(default_factory=list)
    candidate_injection_vectors: list[str] = Field(default_factory=list)
    endpoint_policy: str = "all"
    payload_position: str = "unknown"
    trigger_stage: str = "unknown"
    knowledge_level: str = "unknown"
    payload_sha256: str | None = None
    payload_text: str | None = None
    payload_by_vector: dict[str, str] = Field(default_factory=dict)
    payload_segments: list[str] = Field(default_factory=list)


class VictimActionEvent(BaseModel):
    """A victim-policy action, separated from the simulator transition."""

    model_config = ConfigDict(extra="forbid")

    event_index: int = Field(ge=0)
    tool_name: str | None = None
    skill_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    stop: bool = False
    parse_valid: bool = True
    source: Literal["observed", "imagined"] = "observed"

    @model_validator(mode="after")
    def stopped_event_has_no_required_tool(self):
        if not self.stop and not self.tool_name and self.skill_name != "invalid_tool_call":
            raise ValueError("non-stop victim event requires a tool_name")
        return self


class DecisionStateSourceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: str
    ignored_outcome_paths: list[str] = Field(default_factory=list)
    trusted_task_context_supplied: bool
    semantic_whitelist_version: str = "manifest-v1"


class CanonicalDecisionState(BaseModel):
    """A model input that is invariant to unobserved rollout outcomes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    decision_id: str
    suite: str
    task_id: str
    task_split: str | None = None
    task_group_id: str
    victim_model: str
    agent_scaffold: str
    defense: str
    trusted_goal: str | None = None
    candidate_tools: list[str] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    canonical_environment: dict[str, Any] = Field(default_factory=dict)
    completed_goal_slots: list[str] = Field(default_factory=list)
    remaining_goal_slots: list[str] = Field(default_factory=list)
    observed_victim_prefix: list[VictimActionEvent] = Field(default_factory=list)
    clean_solvability_prior: CleanSolvabilityPrior | None = None
    attacker_action: AttackerIntervention
    public_context: dict[str, Any] = Field(default_factory=dict)
    source_audit: DecisionStateSourceAudit
    state_fingerprint: str


def _semantic_state_payload(state: CanonicalDecisionState | dict[str, Any]) -> dict[str, Any]:
    payload = (
        state.model_dump(mode="json")
        if isinstance(state, CanonicalDecisionState)
        else dict(state)
    )
    payload.pop("state_fingerprint", None)
    # Audit diagnostics must not change the identity of the actual decision.
    payload.pop("source_audit", None)
    return canonical_json_value(payload)


def build_manifest_decision_state(
    row: Mapping[str, Any],
    *,
    victim_model: str,
    agent_scaffold: str,
    defense: str = "none",
    task_context: Mapping[str, Any] | None = None,
    include_clean_prior: bool = True,
) -> CanonicalDecisionState:
    """Build a label-blind state from one v2 manifest row.

    Source rows are intentionally read through a whitelist.  Consequently,
    appending utility/security labels or full trajectories to a manifest row
    cannot alter the semantic state or its fingerprint.
    """

    context = dict(task_context or {})
    row_id = str(row["row_id"])
    suite = str(row["suite"])
    task_id = str(row["user_task_id"])
    payload_hash = row.get("payload_sha256")
    if not payload_hash:
        attack_payload = {
            "payload": row.get("payload"),
            "payload_by_vector": row.get("payload_by_vector", {}),
            "payload_segments": row.get("payload_segments", []),
            "attack_name": row.get("attack_name"),
        }
        payload_hash = stable_fingerprint(attack_payload)

    attacker = AttackerIntervention(
        action_id=row_id,
        family=str(row.get("attack_family", "unknown")),
        variant=str(row.get("attack_variant", "unknown")),
        role=str(row.get("attack_role", "unknown")),
        attack_name=str(row.get("attack_name", "unknown")),
        injection_task_id=(
            str(row["injection_task_id"])
            if row.get("injection_task_id") is not None
            else None
        ),
        injection_goal=(
            str(row["injection_goal"]) if row.get("injection_goal") is not None else None
        ),
        target_tool_sequence=canonical_json_value(row.get("target_tool_sequence", [])),
        candidate_injection_vectors=[
            str(item) for item in row.get("candidate_injection_vectors", [])
        ],
        endpoint_policy=str(row.get("endpoint_policy", "all")),
        payload_position=str(row.get("payload_position", "unknown")),
        trigger_stage=str(row.get("trigger_stage", "unknown")),
        knowledge_level=str(row.get("knowledge_level", "unknown")),
        payload_sha256=str(payload_hash),
        payload_text=(str(row["payload"]) if row.get("payload") is not None else None),
        payload_by_vector={
            str(key): str(value)
            for key, value in row.get("payload_by_vector", {}).items()
        },
        payload_segments=[str(item) for item in row.get("payload_segments", [])],
    )

    clean_prior = None
    if include_clean_prior and row.get("base_success_rate") is not None:
        clean_prior = CleanSolvabilityPrior(
            posterior_mean=float(row["base_success_rate"]),
            trials=(
                int(row["base_success_attempts"])
                if row.get("base_success_attempts") is not None
                else None
            ),
            source="precomputed_clean_manifest",
            source_split=str(row.get("task_split", "unknown")),
        )

    draft: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": f"decision::{row_id}",
        "suite": suite,
        "task_id": task_id,
        "task_split": (
            str(row["task_split"]) if row.get("task_split") is not None else None
        ),
        "task_group_id": str(row.get("task_group_id", f"{suite}|{task_id}")),
        "victim_model": victim_model,
        "agent_scaffold": agent_scaffold,
        "defense": defense,
        "trusted_goal": (
            str(context["trusted_goal"])
            if context.get("trusted_goal") is not None
            else None
        ),
        "candidate_tools": [str(item) for item in context.get("candidate_tools", [])],
        "tool_schemas": canonical_json_value(context.get("tool_schemas", [])),
        "canonical_environment": canonical_json_value(
            context.get("canonical_environment", {})
        ),
        "completed_goal_slots": [
            str(item) for item in context.get("completed_goal_slots", [])
        ],
        "remaining_goal_slots": [
            str(item) for item in context.get("remaining_goal_slots", [])
        ],
        "observed_victim_prefix": [],
        "clean_solvability_prior": clean_prior,
        "attacker_action": attacker,
        "public_context": {
            "required_tool_depth": row.get("required_tool_depth"),
            "underspecification": row.get("underspecification"),
            "solvability_bin": row.get("solvability_bin"),
        },
        "source_audit": DecisionStateSourceAudit(
            source_kind="v2_manifest",
            ignored_outcome_paths=find_outcome_paths(row),
            trusted_task_context_supplied=bool(task_context),
        ),
        "state_fingerprint": "pending",
    }
    draft["state_fingerprint"] = stable_fingerprint(_semantic_state_payload(draft))
    return CanonicalDecisionState.model_validate(draft)


def step_to_victim_event(
    step: StepRecord | Mapping[str, Any], *, source: Literal["observed", "imagined"] = "observed"
) -> VictimActionEvent:
    """Convert an archival step to the action-only victim event representation."""

    record = step if isinstance(step, StepRecord) else StepRecord.model_validate(step)
    stop = record.selected_skill == "finish"
    return VictimActionEvent(
        event_index=record.step_id,
        tool_name=record.selected_tool,
        skill_name=record.selected_skill,
        arguments=canonical_json_value(record.skill_arguments),
        stop=stop,
        parse_valid=(record.selected_skill != "invalid_tool_call" and not record.tool_error),
        source=source,
    )

