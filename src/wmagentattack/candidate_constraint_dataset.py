"""Causal candidate-by-constraint evidence dataset construction.

The pilot deliberately separates observed outcomes from unlabeled
counterfactual queries.  Proof contracts, future calls, final outcomes, and
hidden simulator state are not used as model features or target generators.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .decision_state import canonical_json_value, stable_fingerprint
from .hybrid_semantic_world_model import ExactObservedSemanticTransition
from .markov_sufficiency import validate_dataset_alignment
from .semantic_state_v3 import (
    StructuredSemanticStateV3,
    find_semantic_state_v3_leakage,
)


CANDIDATE_CONSTRAINT_SCHEMA_VERSION = (
    "wmagentattack.candidate_constraint_evidence.pilot.v1"
)
PROGRESS_OUTCOMES = (
    "ALREADY_SUPPORTED",
    "NEWLY_SUPPORTED",
    "UNCHANGED_UNSUPPORTED",
)


class ConstraintAtom(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_ref: str
    kind: Literal["goal_fact_term"] = "goal_fact_term"
    term: str
    source: Literal["trusted_goal.fact_terms"] = "trusted_goal.fact_terms"


class CandidateAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str
    arguments: dict[str, Any] | None
    argument_binding: Literal["OBSERVED", "UNBOUND_COUNTERFACTUAL"]


class EvidenceEvents(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_status: Literal["success", "error"]
    record_count_delta: int = Field(ge=0)
    conflict_count_delta: int = Field(ge=0)
    unresolved_entity_count_delta: int
    newly_matched_goal_terms: tuple[str, ...]


class ConstraintProgressTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    progress: Literal[
        "ALREADY_SUPPORTED",
        "NEWLY_SUPPORTED",
        "UNCHANGED_UNSUPPORTED",
    ]
    prior_status: Literal["SUPPORTED", "UNSUPPORTED"]
    training_role: Literal["PREDICTIVE", "STATE_CONSISTENCY_ONLY"]
    events: EvidenceEvents
    label_source: Literal["adjacent_observed_clean_state"] = (
        "adjacent_observed_clean_state"
    )


class RowMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    split: str
    suite: str
    difficulty: str
    archetype: str
    track: str
    run_seed: int
    transition_index: int = Field(ge=0)


class ObservedCandidateConstraintRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_ref: str
    transition_ref: str
    state_ref: str
    constraint_ref: str
    candidate_action: CandidateAction
    target: ConstraintProgressTarget
    metadata: RowMetadata


class UnlabeledCounterfactualQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_ref: str
    transition_ref: str
    state_ref: str
    constraint_ref: str
    candidate_action: CandidateAction
    label_status: Literal["UNLABELED_COUNTERFACTUAL"] = (
        "UNLABELED_COUNTERFACTUAL"
    )
    metadata: RowMetadata


def _constraint_ref(goal: str, term: str) -> str:
    return stable_fingerprint(
        {"kind": "goal_fact_term", "trusted_goal": goal, "term": term}
    )


def _transition_ref(episode: Mapping[str, Any], index: int) -> str:
    return stable_fingerprint(
        {
            "task_id": episode["task_id"],
            "track": episode["track"],
            "run_seed": episode["run_seed"],
            "transition_index": index,
        }
    )


def _row_ref(
    *, transition_ref: str, state_ref: str, candidate: str, constraint_ref: str
) -> str:
    return stable_fingerprint(
        {
            "transition_ref": transition_ref,
            "state_ref": state_ref,
            "candidate": candidate,
            "constraint_ref": constraint_ref,
        }
    )


def select_balanced_development_tasks(
    raw_dataset: Mapping[str, Any],
    *,
    split: str,
    track: str,
    suites: Sequence[str],
    difficulties: Sequence[str],
) -> tuple[str, ...]:
    """Select one lexicographically first task per metadata cell."""

    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    for episode in raw_dataset["episodes"]:
        if episode["split"] == split and episode["track"] == track:
            cells[(str(episode["suite"]), str(episode["task_difficulty"]))].add(
                str(episode["task_id"])
            )
    expected = {(suite, difficulty) for suite in suites for difficulty in difficulties}
    missing = sorted(expected - set(cells))
    if missing:
        raise ValueError(f"missing metadata selection cells: {missing}")
    extras = sorted(set(cells) - expected)
    if extras:
        raise ValueError(f"unexpected metadata selection cells: {extras}")
    return tuple(sorted(min(cells[cell]) for cell in sorted(expected)))


def _metadata(episode: Mapping[str, Any], index: int) -> RowMetadata:
    return RowMetadata(
        task_id=str(episode["task_id"]),
        split=str(episode["split"]),
        suite=str(episode["suite"]),
        difficulty=str(episode["task_difficulty"]),
        archetype=str(episode["task_archetype"]),
        track=str(episode["track"]),
        run_seed=int(episode["run_seed"]),
        transition_index=index,
    )


def _progress(term: str, current: set[str], following: set[str]) -> str:
    if term in current:
        return "ALREADY_SUPPORTED"
    if term in following:
        return "NEWLY_SUPPORTED"
    return "UNCHANGED_UNSUPPORTED"


def build_candidate_constraint_pilot(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    split: str,
    track: str,
    suites: Sequence[str],
    difficulties: Sequence[str],
) -> dict[str, Any]:
    validate_dataset_alignment(raw_dataset, semantic_dataset)
    selected = select_balanced_development_tasks(
        raw_dataset,
        split=split,
        track=track,
        suites=suites,
        difficulties=difficulties,
    )
    selected_set = set(selected)
    exact = ExactObservedSemanticTransition()
    state_catalog: dict[str, dict[str, Any]] = {}
    constraint_catalog: dict[str, dict[str, Any]] = {}
    observed: list[dict[str, Any]] = []
    counterfactual: list[dict[str, Any]] = []
    selected_episodes: list[dict[str, Any]] = []

    for raw_episode, semantic_episode in zip(
        raw_dataset["episodes"], semantic_dataset["episodes"]
    ):
        if (
            raw_episode["task_id"] not in selected_set
            or raw_episode["split"] != split
            or raw_episode["track"] != track
        ):
            continue
        selected_episodes.append(
            {
                "task_id": raw_episode["task_id"],
                "suite": raw_episode["suite"],
                "difficulty": raw_episode["task_difficulty"],
                "archetype": raw_episode["task_archetype"],
                "track": raw_episode["track"],
                "run_seed": raw_episode["run_seed"],
                "prefixes": len(raw_episode["prefixes"]),
            }
        )
        raw_prefixes = raw_episode["prefixes"]
        semantic_prefixes = semantic_episode["prefixes"]
        for index in range(len(raw_prefixes) - 1):
            raw_prefix = raw_prefixes[index]
            state_payload = semantic_prefixes[index]["features"]["semantic_state_v3"]
            next_payload = semantic_prefixes[index + 1]["features"][
                "semantic_state_v3"
            ]
            state = StructuredSemanticStateV3.model_validate(state_payload)
            following = StructuredSemanticStateV3.model_validate(next_payload)
            action_id = str(raw_prefix["targets"]["next_action"])
            if action_id == "STOP":
                raise ValueError("STOP cannot have an observed adjacent transition")
            exact.advance(state, following, executed_action_id=action_id)
            if action_id not in state.legal_actions:
                raise ValueError("observed candidate action is not legal")
            state_ref = stable_fingerprint(state.model_dump(mode="json"))
            state_catalog.setdefault(state_ref, state.model_dump(mode="json"))
            transition_ref = _transition_ref(raw_episode, index)
            metadata = _metadata(raw_episode, index)
            current_matched = set(state.goal_evidence.matched_fact_terms)
            next_matched = set(following.goal_evidence.matched_fact_terms)
            current_unresolved = (
                state.goal_evidence.ambiguous_entity_records
                + state.goal_evidence.unlinked_entity_records
            )
            next_unresolved = (
                following.goal_evidence.ambiguous_entity_records
                + following.goal_evidence.unlinked_entity_records
            )
            events = EvidenceEvents(
                execution_status=following.execution.last_status,
                record_count_delta=(
                    len(following.evidence_records) - len(state.evidence_records)
                ),
                conflict_count_delta=len(following.conflicts) - len(state.conflicts),
                unresolved_entity_count_delta=next_unresolved - current_unresolved,
                newly_matched_goal_terms=tuple(
                    sorted(next_matched - current_matched)
                ),
            )
            actual_arguments = canonical_json_value(
                following.execution.last_action.get("arguments", {})
            )
            actual_candidate = CandidateAction(
                tool_id=action_id,
                arguments=actual_arguments,
                argument_binding="OBSERVED",
            )
            for term in state.goal.fact_terms:
                constraint_ref = _constraint_ref(
                    state.goal.normalized_goal, term
                )
                constraint = ConstraintAtom(
                    constraint_ref=constraint_ref, term=term
                )
                constraint_catalog.setdefault(
                    constraint_ref, constraint.model_dump(mode="json")
                )
                observed.append(
                    ObservedCandidateConstraintRow(
                        row_ref=_row_ref(
                            transition_ref=transition_ref,
                            state_ref=state_ref,
                            candidate=action_id,
                            constraint_ref=constraint_ref,
                        ),
                        transition_ref=transition_ref,
                        state_ref=state_ref,
                        constraint_ref=constraint_ref,
                        candidate_action=actual_candidate,
                        target=ConstraintProgressTarget(
                            progress=_progress(
                                term, current_matched, next_matched
                            ),
                            prior_status=(
                                "SUPPORTED"
                                if term in current_matched
                                else "UNSUPPORTED"
                            ),
                            training_role=(
                                "STATE_CONSISTENCY_ONLY"
                                if term in current_matched
                                else "PREDICTIVE"
                            ),
                            events=events,
                        ),
                        metadata=metadata,
                    ).model_dump(mode="json")
                )
                for candidate_id in state.legal_actions:
                    if candidate_id == action_id:
                        continue
                    counterfactual.append(
                        UnlabeledCounterfactualQuery(
                            query_ref=_row_ref(
                                transition_ref=transition_ref,
                                state_ref=state_ref,
                                candidate=candidate_id,
                                constraint_ref=constraint_ref,
                            ),
                            transition_ref=transition_ref,
                            state_ref=state_ref,
                            constraint_ref=constraint_ref,
                            candidate_action=CandidateAction(
                                tool_id=candidate_id,
                                arguments=None,
                                argument_binding="UNBOUND_COUNTERFACTUAL",
                            ),
                            metadata=metadata,
                        ).model_dump(mode="json")
                    )

    if len(selected_episodes) != len(selected):
        raise ValueError("selected development task does not have exactly one episode")
    output = {
        "schema_version": CANDIDATE_CONSTRAINT_SCHEMA_VERSION,
        "scope": "clean-only development schema pilot",
        "selection": {
            "split": split,
            "track": track,
            "suites": list(suites),
            "difficulties": list(difficulties),
            "rule": "lexicographically_first_task_id_per_suite_difficulty_cell",
            "selected_task_ids": list(selected),
        },
        "loader_contract": {
            "resolve_references_before_encoding": True,
            "never_embed_reference_or_metadata_ids": True,
            "observed_rows_are_labeled": True,
            "unlabeled_counterfactual_queries_are_never_negative_targets": True,
            "proof_contract_inputs": False,
            "confirmation_inputs": False,
        },
        "tool_catalog": canonical_json_value(raw_dataset["tool_catalog"]),
        "selected_episodes": sorted(
            selected_episodes, key=lambda row: str(row["task_id"])
        ),
        "state_catalog": dict(sorted(state_catalog.items())),
        "constraint_catalog": dict(sorted(constraint_catalog.items())),
        "observed_rows": sorted(observed, key=lambda row: str(row["row_ref"])),
        "unlabeled_counterfactual_queries": sorted(
            counterfactual, key=lambda row: str(row["query_ref"])
        ),
    }
    return output


def audit_candidate_constraint_pilot(
    dataset: Mapping[str, Any],
    *,
    expected: Mapping[str, int],
    schema_gate: Mapping[str, int],
    readiness_gate: Mapping[str, float | int],
) -> dict[str, Any]:
    states = dataset["state_catalog"]
    constraints = dataset["constraint_catalog"]
    observed = dataset["observed_rows"]
    counterfactual = dataset["unlabeled_counterfactual_queries"]
    episodes = dataset["selected_episodes"]
    leakage = {
        state_ref: list(find_semantic_state_v3_leakage(payload))
        for state_ref, payload in states.items()
        if find_semantic_state_v3_leakage(payload)
    }
    progress = Counter(row["target"]["progress"] for row in observed)
    training_roles = Counter(
        row["target"]["training_role"] for row in observed
    )
    progress_tasks: dict[str, set[str]] = defaultdict(set)
    progress_suites: dict[str, Counter[str]] = defaultdict(Counter)
    progress_difficulties: dict[str, Counter[str]] = defaultdict(Counter)
    for row in observed:
        name = str(row["target"]["progress"])
        progress_tasks[name].add(str(row["metadata"]["task_id"]))
        progress_suites[str(row["metadata"]["suite"])][name] += 1
        progress_difficulties[str(row["metadata"]["difficulty"])][name] += 1
    transition_refs = {row["transition_ref"] for row in observed}
    event_positive = Counter()
    for transition_ref in sorted(transition_refs):
        row = next(row for row in observed if row["transition_ref"] == transition_ref)
        events = row["target"]["events"]
        event_positive["execution_error"] += int(
            events["execution_status"] == "error"
        )
        event_positive["conflict_added"] += int(
            int(events["conflict_count_delta"]) > 0
        )
        event_positive["ambiguous_or_unlinked_added"] += int(
            int(events["unresolved_entity_count_delta"]) > 0
        )
    all_refs_valid = all(
        row["state_ref"] in states and row["constraint_ref"] in constraints
        for row in (*observed, *counterfactual)
    )
    no_counterfactual_targets = all(
        "target" not in row
        and row["label_status"] == "UNLABELED_COUNTERFACTUAL"
        and row["candidate_action"]["argument_binding"]
        == "UNBOUND_COUNTERFACTUAL"
        for row in counterfactual
    )
    observed_pairs = {
        (row["transition_ref"], row["constraint_ref"], row["candidate_action"]["tool_id"])
        for row in observed
    }
    counterfactual_pairs = {
        (row["transition_ref"], row["constraint_ref"], row["candidate_action"]["tool_id"])
        for row in counterfactual
    }
    query_total = len(observed) + len(counterfactual)
    observed_fraction = len(observed) / max(1, query_total)
    expected_cells = {
        (suite, difficulty)
        for suite in dataset["selection"]["suites"]
        for difficulty in dataset["selection"]["difficulties"]
    }
    observed_cells = {
        (str(row["suite"]), str(row["difficulty"])) for row in episodes
    }
    exact_checks = {
        "tasks": len({row["task_id"] for row in episodes})
        == int(expected["tasks"]),
        "episodes": len(episodes) == int(expected["episodes"]),
        "states": len(states) == int(expected["states"]),
        "constraints": len(constraints) == int(expected["constraints"]),
        "transitions": len(transition_refs) == int(expected["transitions"]),
        "observed_rows": len(observed) == int(expected["observed_rows"]),
        "counterfactual_queries": len(counterfactual)
        == int(expected["counterfactual_queries"]),
        "total_query_space": query_total == int(expected["total_query_space"]),
    }
    schema_checks = {
        **exact_checks,
        "training_split_only": dataset["selection"]["split"] == "training"
        and all(row["metadata"]["split"] == "training" for row in observed)
        and all(row["metadata"]["track"] == "deterministic_greedy" for row in observed),
        "zero_state_leakage": not leakage,
        "all_references_valid": all_refs_valid,
        "complete_suite_difficulty_cells": observed_cells == expected_cells,
        "unique_observed_row_refs": len({row["row_ref"] for row in observed})
        == len(observed),
        "unique_counterfactual_query_refs": len(
            {row["query_ref"] for row in counterfactual}
        )
        == len(counterfactual),
        "observed_and_counterfactual_disjoint": not (
            observed_pairs & counterfactual_pairs
        ),
        "counterfactuals_unlabeled": no_counterfactual_targets,
        "all_progress_classes_present": set(progress) == set(PROGRESS_OUTCOMES),
        "minimum_rows_per_progress": min(progress.values())
        >= int(schema_gate["minimum_rows_per_progress"]),
        "minimum_tasks_per_progress": min(
            len(tasks) for tasks in progress_tasks.values()
        )
        >= int(schema_gate["minimum_tasks_per_progress"]),
    }
    readiness_checks = {
        "minimum_observed_candidate_fraction": observed_fraction
        >= float(readiness_gate["minimum_observed_candidate_fraction"]),
        "minimum_execution_errors": event_positive["execution_error"]
        >= int(readiness_gate["minimum_execution_errors"]),
        "minimum_conflicts": event_positive["conflict_added"]
        >= int(readiness_gate["minimum_conflicts"]),
        "minimum_ambiguity_events": event_positive[
            "ambiguous_or_unlinked_added"
        ]
        >= int(readiness_gate["minimum_ambiguity_events"]),
    }
    schema_pass = all(schema_checks.values())
    readiness_pass = all(readiness_checks.values())
    if schema_pass and readiness_pass:
        decision = "GO__CANDIDATE_CONSTRAINT_DATA_READY_FOR_MODEL_PROBE"
    elif schema_pass:
        decision = "GO_SCHEMA__NO_GO_TRAINING__COUNTERFACTUAL_COLLECTION_REQUIRED"
    else:
        decision = "NO_GO__CANDIDATE_CONSTRAINT_SCHEMA_PILOT_FAILED"
    return {
        "schema_version": CANDIDATE_CONSTRAINT_SCHEMA_VERSION,
        "counts": {
            "tasks": len({row["task_id"] for row in episodes}),
            "episodes": len(episodes),
            "states": len(states),
            "constraints": len(constraints),
            "transitions": len(transition_refs),
            "observed_rows": len(observed),
            "counterfactual_queries": len(counterfactual),
            "total_query_space": query_total,
            "observed_candidate_fraction": observed_fraction,
        },
        "progress_rows": dict(sorted(progress.items())),
        "training_role_rows": dict(sorted(training_roles.items())),
        "progress_task_counts": {
            name: len(tasks) for name, tasks in sorted(progress_tasks.items())
        },
        "progress_by_suite": {
            name: dict(sorted(rows.items()))
            for name, rows in sorted(progress_suites.items())
        },
        "progress_by_difficulty": {
            name: dict(sorted(rows.items()))
            for name, rows in sorted(progress_difficulties.items())
        },
        "event_positive_transitions": dict(sorted(event_positive.items())),
        "state_leakage": leakage,
        "schema_checks": schema_checks,
        "training_readiness_checks": readiness_checks,
        "schema_pass": schema_pass,
        "training_ready": readiness_pass,
        "decision": decision,
    }
