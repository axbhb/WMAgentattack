"""Label-blind goal atoms and separated progress views for world-model agents."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_state import canonical_json_value, stable_fingerprint
from .state_storage_v2 import VisibilityScope
from .structured_ledger_v2 import StructuredEvidenceLedgerV2


GOAL_PROGRESS_SCHEMA_VERSION = "wmagentattack.goal_progress.v2"


class GoalAtomStatus(str, Enum):
    UNSUPPORTED = "UNSUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    AMBIGUOUS = "AMBIGUOUS"


class ProgressType(str, Enum):
    KNOWLEDGE = "KNOWLEDGE"
    ENVIRONMENT = "ENVIRONMENT"
    COMPLETION = "COMPLETION"


class GoalAtomKind(str, Enum):
    ENTITY_EXISTS = "ENTITY_EXISTS"
    ATTRIBUTE_KNOWN = "ATTRIBUTE_KNOWN"
    ATTRIBUTE_PREDICATE = "ATTRIBUTE_PREDICATE"
    MUTATION_CONFIRMED = "MUTATION_CONFIRMED"
    SAME_ENTITY = "SAME_ENTITY"
    COMPLETION_FIELD = "COMPLETION_FIELD"


class ComparisonOperator(str, Enum):
    KNOWN = "KNOWN"
    EQ = "EQ"
    NE = "NE"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    CONTAINS = "CONTAINS"


class GoalAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_id: str
    description: str
    progress_type: ProgressType
    kind: GoalAtomKind
    entity_type: str | None = None
    entity_key: dict[str, Any] = Field(default_factory=dict)
    attribute_name: str | None = None
    operator: ComparisonOperator | None = None
    target_value: Any = None
    context_requirements: dict[str, Any] = Field(default_factory=dict)
    member_atom_ids: tuple[str, ...] = ()
    required_for_completion: bool = True
    outcome_labels_present: bool = False

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind == GoalAtomKind.SAME_ENTITY:
            if len(self.member_atom_ids) < 2:
                raise ValueError("SAME_ENTITY requires at least two member atoms")
            if self.entity_type is not None or self.attribute_name is not None:
                raise ValueError("SAME_ENTITY cannot define an entity or attribute")
        elif self.member_atom_ids:
            raise ValueError("only SAME_ENTITY may reference member atoms")
        if self.kind in {
            GoalAtomKind.ENTITY_EXISTS,
            GoalAtomKind.ATTRIBUTE_KNOWN,
            GoalAtomKind.ATTRIBUTE_PREDICATE,
            GoalAtomKind.MUTATION_CONFIRMED,
        } and self.entity_type is None:
            raise ValueError(f"{self.kind.value} requires entity_type")
        if self.kind in {
            GoalAtomKind.ATTRIBUTE_KNOWN,
            GoalAtomKind.ATTRIBUTE_PREDICATE,
            GoalAtomKind.COMPLETION_FIELD,
        } and self.attribute_name is None:
            raise ValueError(f"{self.kind.value} requires attribute_name")
        if self.kind == GoalAtomKind.ATTRIBUTE_PREDICATE and self.operator in {
            None,
            ComparisonOperator.KNOWN,
        }:
            raise ValueError("ATTRIBUTE_PREDICATE requires a value comparator")
        if self.kind == GoalAtomKind.COMPLETION_FIELD and self.progress_type != ProgressType.COMPLETION:
            raise ValueError("COMPLETION_FIELD must use COMPLETION progress")
        if self.progress_type == ProgressType.COMPLETION and self.kind not in {
            GoalAtomKind.COMPLETION_FIELD,
            GoalAtomKind.SAME_ENTITY,
        }:
            raise ValueError("COMPLETION progress requires a completion-specific atom")
        return self


class GoalAtomPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = GOAL_PROGRESS_SCHEMA_VERSION
    task_id: str
    trusted_goal: str
    trusted_goal_fingerprint: str = Field(min_length=64, max_length=64)
    tool_schema_fingerprint: str = Field(min_length=64, max_length=64)
    compiler_id: Literal["manual_schema_grounded_v2"] = "manual_schema_grounded_v2"
    atoms: tuple[GoalAtom, ...]
    outcome_labels_present: bool = False

    @model_validator(mode="after")
    def validate_plan(self):
        if self.trusted_goal_fingerprint != stable_fingerprint(self.trusted_goal):
            raise ValueError("trusted goal fingerprint mismatch")
        by_id = {atom.atom_id: atom for atom in self.atoms}
        if len(by_id) != len(self.atoms):
            raise ValueError("goal plan contains duplicate atom IDs")
        for index, atom in enumerate(self.atoms):
            if atom.kind != GoalAtomKind.SAME_ENTITY:
                continue
            missing = sorted(set(atom.member_atom_ids) - set(by_id))
            if missing:
                raise ValueError(f"SAME_ENTITY references missing atoms: {missing}")
            prior = {row.atom_id for row in self.atoms[:index]}
            if not set(atom.member_atom_ids).issubset(prior):
                raise ValueError("SAME_ENTITY members must precede the relation atom")
        return self


class EnvironmentFact(BaseModel):
    """A task-relevant fact derived by the simulator from exact state."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    entity_type: str
    entity_key: dict[str, Any]
    entity_id: str
    attribute_name: str
    value: Any
    context: dict[str, Any] = Field(default_factory=dict)
    source_call_index: int = Field(ge=0)
    visibility_scope: Literal[VisibilityScope.PLANNER_PRIVILEGED] = (
        VisibilityScope.PLANNER_PRIVILEGED
    )
    outcome_labels_present: bool = False


class CompletionObservation(BaseModel):
    """Victim-produced stopping and answer fields, without checker outcomes."""

    model_config = ConfigDict(extra="forbid")

    stop_decision_observed: bool
    final_answer_present: bool
    answer_fields: dict[str, Any] = Field(default_factory=dict)
    visibility_scope: Literal[VisibilityScope.VICTIM_OBSERVED] = (
        VisibilityScope.VICTIM_OBSERVED
    )
    outcome_labels_present: bool = False


class AtomAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_id: str
    progress_type: ProgressType
    status: GoalAtomStatus
    status_probabilities: dict[str, float]
    supporting_entity_ids: tuple[str, ...] = ()
    candidate_entity_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rationale_code: str
    outcome_labels_present: bool = False

    @model_validator(mode="after")
    def validate_probabilities(self):
        if set(self.status_probabilities) != {status.value for status in GoalAtomStatus}:
            raise ValueError("atom belief must cover every status")
        if any(value < 0.0 or value > 1.0 for value in self.status_probabilities.values()):
            raise ValueError("atom belief probability outside [0, 1]")
        if abs(sum(self.status_probabilities.values()) - 1.0) > 1e-8:
            raise ValueError("atom belief probabilities must sum to one")
        if self.status_probabilities[self.status.value] != 1.0:
            raise ValueError("deterministic assessment must be one-hot at status")
        return self


class GoalProgressSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = GOAL_PROGRESS_SCHEMA_VERSION
    task_id: str
    assessments: tuple[AtomAssessment, ...]
    coverage_by_type: dict[str, float]
    required_atom_coverage: float = Field(ge=0.0, le=1.0)
    completion_ready: bool
    outcome_labels_present: bool = False


def compile_goal_plan(
    *,
    task_id: str,
    trusted_goal: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    atom_declarations: Sequence[GoalAtom | Mapping[str, Any]],
) -> GoalAtomPlan:
    """Compile a reviewed plan from trusted goal/schema inputs only."""

    atoms = tuple(
        atom if isinstance(atom, GoalAtom) else GoalAtom.model_validate(atom)
        for atom in atom_declarations
    )
    return GoalAtomPlan(
        task_id=task_id,
        trusted_goal=trusted_goal,
        trusted_goal_fingerprint=stable_fingerprint(trusted_goal),
        tool_schema_fingerprint=stable_fingerprint(
            canonical_json_value(list(tool_schemas))
        ),
        atoms=atoms,
    )


def build_environment_fact(
    *,
    entity_type: str,
    entity_key: Mapping[str, Any],
    attribute_name: str,
    value: Any,
    context: Mapping[str, Any],
    source_call_index: int,
) -> EnvironmentFact:
    canonical_key = canonical_json_value(dict(entity_key))
    entity_id = f"ENV_ENTITY::{entity_type}::{stable_fingerprint(canonical_key)[:24]}"
    fact_id = stable_fingerprint(
        {
            "entity_type": entity_type,
            "entity_key": canonical_key,
            "attribute_name": attribute_name,
            "value": canonical_json_value(value),
            "context": canonical_json_value(dict(context)),
            "source_call_index": source_call_index,
        }
    )
    return EnvironmentFact(
        fact_id=f"env_fact::{fact_id}",
        entity_type=entity_type,
        entity_key=canonical_key,
        entity_id=entity_id,
        attribute_name=attribute_name,
        value=canonical_json_value(value),
        context=canonical_json_value(dict(context)),
        source_call_index=source_call_index,
    )


def _one_hot(status: GoalAtomStatus) -> dict[str, float]:
    return {candidate.value: float(candidate == status) for candidate in GoalAtomStatus}


def _assessment(
    atom: GoalAtom,
    status: GoalAtomStatus,
    *,
    supporting_entity_ids: Sequence[str] = (),
    candidate_entity_ids: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    rationale_code: str,
) -> AtomAssessment:
    return AtomAssessment(
        atom_id=atom.atom_id,
        progress_type=atom.progress_type,
        status=status,
        status_probabilities=_one_hot(status),
        supporting_entity_ids=tuple(sorted(set(supporting_entity_ids))),
        candidate_entity_ids=tuple(sorted(set(candidate_entity_ids))),
        evidence_ids=tuple(sorted(set(evidence_ids))),
        rationale_code=rationale_code,
    )


def _contains(actual: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    return all(
        key in actual and canonical_json_value(actual[key]) == canonical_json_value(value)
        for key, value in required.items()
    )


def _compare(value: Any, operator: ComparisonOperator, target: Any) -> bool:
    if operator == ComparisonOperator.KNOWN:
        return value is not None
    if operator == ComparisonOperator.EQ:
        return canonical_json_value(value) == canonical_json_value(target)
    if operator == ComparisonOperator.NE:
        return canonical_json_value(value) != canonical_json_value(target)
    if operator == ComparisonOperator.CONTAINS:
        if isinstance(value, Mapping):
            return target in value
        if isinstance(value, (list, tuple, set, str)):
            return target in value
        return False
    try:
        left = float(value)
        right = float(target)
    except (TypeError, ValueError):
        return False
    if operator == ComparisonOperator.LT:
        return left < right
    if operator == ComparisonOperator.LE:
        return left <= right
    if operator == ComparisonOperator.GT:
        return left > right
    if operator == ComparisonOperator.GE:
        return left >= right
    raise AssertionError(operator)


def _ledger_atom(atom: GoalAtom, ledger: StructuredEvidenceLedgerV2) -> AtomAssessment:
    records = [
        record
        for record in ledger.records
        if record.observation_scope == VisibilityScope.VICTIM_OBSERVED
        and record.entity_type == atom.entity_type
        and _contains(record.entity_key, atom.entity_key)
        and _contains(record.context, atom.context_requirements)
        and record.execution_status == "success"
    ]
    if atom.kind == GoalAtomKind.ENTITY_EXISTS:
        relevant = [(record, True) for record in records]
    elif atom.kind == GoalAtomKind.MUTATION_CONFIRMED:
        relevant = []
        for record in records:
            acknowledged = any(
                attribute.name == "execution_acknowledged" and attribute.value is True
                for attribute in record.attributes
            )
            if record.state_provenance == "mutating":
                relevant.append((record, acknowledged))
    else:
        relevant = []
        operator = atom.operator or ComparisonOperator.KNOWN
        for record in records:
            for attribute in record.attributes:
                if attribute.name != atom.attribute_name:
                    continue
                relevant.append(
                    (record, _compare(attribute.value, operator, atom.target_value))
                )

    supporting = []
    candidates = []
    evidence = []
    unlinked_support = False
    observed_value = False
    for record, satisfied in relevant:
        observed_value = True
        evidence.append(record.record_id)
        if not satisfied:
            continue
        if record.resolved_entity_id is not None:
            supporting.append(record.resolved_entity_id)
        elif record.entity_candidates:
            candidates.extend(candidate.entity_id for candidate in record.entity_candidates)
        else:
            unlinked_support = True
    if supporting:
        return _assessment(
            atom,
            GoalAtomStatus.SUPPORTED,
            supporting_entity_ids=supporting,
            candidate_entity_ids=candidates,
            evidence_ids=evidence,
            rationale_code="victim_observed_unique_support",
        )
    if candidates:
        return _assessment(
            atom,
            GoalAtomStatus.AMBIGUOUS,
            candidate_entity_ids=candidates,
            evidence_ids=evidence,
            rationale_code="victim_observed_candidate_support",
        )
    if unlinked_support:
        return _assessment(
            atom,
            GoalAtomStatus.PARTIALLY_SUPPORTED,
            evidence_ids=evidence,
            rationale_code="victim_observed_unlinked_support",
        )
    if observed_value:
        return _assessment(
            atom,
            GoalAtomStatus.CONTRADICTED,
            evidence_ids=evidence,
            rationale_code="victim_observed_incompatible_value",
        )
    return _assessment(
        atom,
        GoalAtomStatus.UNSUPPORTED,
        rationale_code="no_matching_victim_observation",
    )


def _environment_atom(
    atom: GoalAtom, environment_facts: Sequence[EnvironmentFact]
) -> AtomAssessment:
    operator = atom.operator or ComparisonOperator.KNOWN
    relevant = [
        fact
        for fact in environment_facts
        if fact.visibility_scope == VisibilityScope.PLANNER_PRIVILEGED
        and fact.entity_type == atom.entity_type
        and _contains(fact.entity_key, atom.entity_key)
        and _contains(fact.context, atom.context_requirements)
        and (
            atom.kind == GoalAtomKind.ENTITY_EXISTS
            or fact.attribute_name == atom.attribute_name
        )
    ]
    supporting = [
        fact
        for fact in relevant
        if atom.kind == GoalAtomKind.ENTITY_EXISTS
        or _compare(fact.value, operator, atom.target_value)
    ]
    if supporting:
        return _assessment(
            atom,
            GoalAtomStatus.SUPPORTED,
            supporting_entity_ids=[fact.entity_id for fact in supporting],
            evidence_ids=[fact.fact_id for fact in relevant],
            rationale_code="simulator_derived_environment_support",
        )
    if relevant:
        return _assessment(
            atom,
            GoalAtomStatus.CONTRADICTED,
            evidence_ids=[fact.fact_id for fact in relevant],
            rationale_code="simulator_derived_incompatible_value",
        )
    return _assessment(
        atom,
        GoalAtomStatus.UNSUPPORTED,
        rationale_code="no_matching_environment_fact",
    )


def _completion_atom(
    atom: GoalAtom, observation: CompletionObservation | None
) -> AtomAssessment:
    if observation is None or not observation.final_answer_present:
        return _assessment(
            atom,
            GoalAtomStatus.UNSUPPORTED,
            rationale_code="no_final_answer_observation",
        )
    if atom.attribute_name not in observation.answer_fields:
        return _assessment(
            atom,
            GoalAtomStatus.UNSUPPORTED,
            rationale_code="answer_field_missing",
        )
    value = observation.answer_fields[atom.attribute_name]
    operator = atom.operator or ComparisonOperator.KNOWN
    status = (
        GoalAtomStatus.SUPPORTED
        if _compare(value, operator, atom.target_value)
        else GoalAtomStatus.CONTRADICTED
    )
    return _assessment(
        atom,
        status,
        evidence_ids=[f"answer_field::{atom.attribute_name}"],
        rationale_code=(
            "victim_answer_field_support"
            if status == GoalAtomStatus.SUPPORTED
            else "victim_answer_field_incompatible"
        ),
    )


def _same_entity_atom(
    atom: GoalAtom, assessments: Mapping[str, AtomAssessment]
) -> AtomAssessment:
    members = [assessments[atom_id] for atom_id in atom.member_atom_ids]
    support_sets = [set(member.supporting_entity_ids) for member in members]
    if all(support_sets):
        intersection = set.intersection(*support_sets)
        if intersection:
            return _assessment(
                atom,
                GoalAtomStatus.SUPPORTED,
                supporting_entity_ids=intersection,
                evidence_ids=[row for member in members for row in member.evidence_ids],
                rationale_code="same_resolved_entity_supports_all_members",
            )
        return _assessment(
            atom,
            GoalAtomStatus.CONTRADICTED,
            evidence_ids=[row for member in members for row in member.evidence_ids],
            rationale_code="member_support_split_across_entities",
        )
    possible_sets = [
        set(member.supporting_entity_ids) | set(member.candidate_entity_ids)
        for member in members
    ]
    if all(possible_sets) and set.intersection(*possible_sets):
        return _assessment(
            atom,
            GoalAtomStatus.AMBIGUOUS,
            candidate_entity_ids=set.intersection(*possible_sets),
            evidence_ids=[row for member in members for row in member.evidence_ids],
            rationale_code="candidate_entity_may_support_all_members",
        )
    if any(
        member.status
        in {
            GoalAtomStatus.SUPPORTED,
            GoalAtomStatus.PARTIALLY_SUPPORTED,
            GoalAtomStatus.AMBIGUOUS,
        }
        for member in members
    ):
        return _assessment(
            atom,
            GoalAtomStatus.PARTIALLY_SUPPORTED,
            evidence_ids=[row for member in members for row in member.evidence_ids],
            rationale_code="only_subset_of_relation_members_supported",
        )
    return _assessment(
        atom,
        GoalAtomStatus.UNSUPPORTED,
        rationale_code="relation_members_unsupported",
    )


def assess_goal_progress(
    plan: GoalAtomPlan,
    ledger: StructuredEvidenceLedgerV2,
    *,
    environment_facts: Sequence[EnvironmentFact] = (),
    completion_observation: CompletionObservation | None = None,
) -> GoalProgressSnapshot:
    """Assess current atoms without reading experts, future calls, or outcomes."""

    by_id: dict[str, AtomAssessment] = {}
    for atom in plan.atoms:
        if atom.kind == GoalAtomKind.SAME_ENTITY:
            assessment = _same_entity_atom(atom, by_id)
        elif atom.progress_type == ProgressType.KNOWLEDGE:
            assessment = _ledger_atom(atom, ledger)
        elif atom.progress_type == ProgressType.ENVIRONMENT:
            assessment = _environment_atom(atom, environment_facts)
        else:
            assessment = _completion_atom(atom, completion_observation)
        by_id[atom.atom_id] = assessment
    assessments = tuple(by_id[atom.atom_id] for atom in plan.atoms)
    coverage_by_type = {}
    for progress_type in ProgressType:
        selected = [row for row in assessments if row.progress_type == progress_type]
        coverage_by_type[progress_type.value] = (
            sum(row.status_probabilities[GoalAtomStatus.SUPPORTED.value] for row in selected)
            / len(selected)
            if selected
            else 0.0
        )
    required = [
        by_id[atom.atom_id] for atom in plan.atoms if atom.required_for_completion
    ]
    required_coverage = (
        sum(row.status_probabilities[GoalAtomStatus.SUPPORTED.value] for row in required)
        / len(required)
        if required
        else 1.0
    )
    completion_ready = bool(
        required
        and all(row.status == GoalAtomStatus.SUPPORTED for row in required)
        and completion_observation is not None
        and completion_observation.stop_decision_observed
        and completion_observation.final_answer_present
    )
    return GoalProgressSnapshot(
        task_id=plan.task_id,
        assessments=assessments,
        coverage_by_type=coverage_by_type,
        required_atom_coverage=required_coverage,
        completion_ready=completion_ready,
    )
