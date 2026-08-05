"""Outcome-blind manifest construction for clean sandbox counterfactuals.

The manifest is selected before any counterfactual tool is executed.  It uses
only training metadata, causal Semantic-State-v3 prefixes, legal tool schemas,
and argument payloads from already-observed clean training calls.  Donor tool
outputs and all utility/security/final labels are outside this module.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_state import canonical_json_value, stable_fingerprint
from .markov_sufficiency import validate_dataset_alignment
from .semantic_state_v3 import StructuredSemanticStateV3


COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION = (
    "wmagentattack.counterfactual_evidence_manifest.v1"
)
BINDING_SOURCES = (
    "SCHEMA_EMPTY",
    "CROSS_TASK_OBSERVED_DONOR",
    "SAME_TASK_CLEAN_DONOR",
)


@dataclass(frozen=True)
class ToolBindingSpec:
    candidate_id: str
    tool_name: str
    suite: str
    required_fields: tuple[str, ...]
    mutating: bool
    validator: Callable[[Mapping[str, Any]], dict[str, Any]]

    def validate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return canonical_json_value(self.validator(arguments))


class QueryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    episode_id: str
    split: Literal["training"]
    suite: str
    difficulty: str
    archetype: str
    track: Literal["deterministic_greedy"]
    run_seed: int
    prefix_index: int = Field(ge=0)


class CandidateQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_ref: str
    decision_ref: str
    state_ref: str
    candidate_id: str
    tool_name: str
    mutation_class: Literal["read_only", "mutating"]
    current_victim_decision: str
    constraint_refs: tuple[str, ...]
    metadata: QueryMetadata


class ArgumentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal[
        "SCHEMA_EMPTY",
        "CROSS_TASK_OBSERVED_DONOR",
        "SAME_TASK_CLEAN_DONOR",
    ]
    arguments: dict[str, Any]
    donor_task_id: str | None = None
    donor_episode_id: str | None = None
    donor_track: str | None = None
    donor_transition_index: int | None = Field(default=None, ge=0)
    donor_outcomes_read: Literal[False] = False

    @model_validator(mode="after")
    def validate_provenance(self):
        donor_fields = (
            self.donor_task_id,
            self.donor_episode_id,
            self.donor_track,
            self.donor_transition_index,
        )
        if self.source == "SCHEMA_EMPTY" and any(
            value is not None for value in donor_fields
        ):
            raise ValueError("schema-empty binding cannot carry donor metadata")
        if self.source != "SCHEMA_EMPTY" and any(
            value is None for value in donor_fields
        ):
            raise ValueError("observed-donor binding requires complete provenance")
        return self


class CounterfactualManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_row_ref: str
    query: CandidateQuery
    binding: ArgumentBinding


def build_tool_binding_specs(
    suites: Mapping[str, Any], *, mutating_tools: set[str]
) -> dict[str, ToolBindingSpec]:
    specs: dict[str, ToolBindingSpec] = {}
    for suite_name, suite in sorted(suites.items()):
        for tool in sorted(suite.tools, key=lambda row: str(row.name)):
            candidate_id = f"{suite_name}::{tool.name}"
            schema = tool.parameters.model_json_schema()

            def validate(
                arguments: Mapping[str, Any], *, model=tool.parameters
            ) -> dict[str, Any]:
                return model.model_validate(arguments).model_dump(mode="json")

            specs[candidate_id] = ToolBindingSpec(
                candidate_id=candidate_id,
                tool_name=str(tool.name),
                suite=str(suite_name),
                required_fields=tuple(
                    sorted(str(value) for value in schema.get("required", ()))
                ),
                mutating=str(tool.name) in mutating_tools,
                validator=validate,
            )
    return specs


def _decision_ref(episode: Mapping[str, Any], prefix_index: int) -> str:
    return stable_fingerprint(
        {
            "task_id": episode["task_id"],
            "track": episode["track"],
            "run_seed": episode["run_seed"],
            "prefix_index": prefix_index,
        }
    )


def _constraint_ref(goal: str, term: str) -> str:
    return stable_fingerprint(
        {"kind": "goal_fact_term", "trusted_goal": goal, "term": term}
    )


def _rank(seed: str, *parts: Any) -> str:
    serialized = [
        str(part)
        if part is None or isinstance(part, (str, int, float, bool))
        else stable_fingerprint(canonical_json_value(part))
        for part in parts
    ]
    payload = "|".join([seed, *serialized])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_query_universe(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    selected_task_ids: Sequence[str],
    tool_specs: Mapping[str, ToolBindingSpec],
) -> dict[str, Any]:
    """Build all executable state/action queries, including terminal prefixes."""

    validate_dataset_alignment(raw_dataset, semantic_dataset)
    selected = set(selected_task_ids)
    states: dict[str, dict[str, Any]] = {}
    constraints: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_episodes = 0
    observed_transition_rows = 0

    for raw_episode, semantic_episode in zip(
        raw_dataset["episodes"], semantic_dataset["episodes"], strict=True
    ):
        if (
            str(raw_episode["task_id"]) not in selected
            or raw_episode["split"] != "training"
            or raw_episode["track"] != "deterministic_greedy"
        ):
            continue
        selected_episodes += 1
        if len(raw_episode["prefixes"]) != len(semantic_episode["prefixes"]):
            raise ValueError("raw and semantic prefix counts disagree")
        for prefix_index, (raw_prefix, semantic_prefix) in enumerate(
            zip(
                raw_episode["prefixes"],
                semantic_episode["prefixes"],
                strict=True,
            )
        ):
            state = StructuredSemanticStateV3.model_validate(
                semantic_prefix["features"]["semantic_state_v3"]
            )
            state_payload = state.model_dump(mode="json")
            state_ref = stable_fingerprint(state_payload)
            states.setdefault(state_ref, state_payload)
            decision_ref = _decision_ref(raw_episode, prefix_index)
            current_decision = str(raw_prefix["targets"]["next_action"])
            constraint_refs = []
            for term in state.goal.fact_terms:
                ref = _constraint_ref(state.goal.normalized_goal, term)
                constraints.setdefault(
                    ref,
                    {
                        "constraint_ref": ref,
                        "kind": "goal_fact_term",
                        "term": term,
                        "source": "trusted_goal.fact_terms",
                    },
                )
                constraint_refs.append(ref)
            if current_decision != "STOP":
                observed_transition_rows += len(constraint_refs)
            decisions.append(
                {
                    "decision_ref": decision_ref,
                    "state_ref": state_ref,
                    "task_id": raw_episode["task_id"],
                    "prefix_index": prefix_index,
                    "current_victim_decision": current_decision,
                    "terminal_decision": current_decision == "STOP",
                    "constraint_refs": sorted(constraint_refs),
                }
            )
            metadata = QueryMetadata(
                task_id=str(raw_episode["task_id"]),
                episode_id=str(raw_episode["episode_id"]),
                split="training",
                suite=str(raw_episode["suite"]),
                difficulty=str(raw_episode["task_difficulty"]),
                archetype=str(raw_episode["task_archetype"]),
                track="deterministic_greedy",
                run_seed=int(raw_episode["run_seed"]),
                prefix_index=prefix_index,
            )
            for candidate_id in state.legal_actions:
                if candidate_id in {"STOP", current_decision}:
                    continue
                spec = tool_specs.get(candidate_id)
                if spec is None:
                    raise KeyError(f"no binding spec for legal candidate {candidate_id}")
                if spec.suite != metadata.suite:
                    raise ValueError("cross-suite legal candidate")
                query_ref = stable_fingerprint(
                    {
                        "decision_ref": decision_ref,
                        "state_ref": state_ref,
                        "candidate_id": candidate_id,
                    }
                )
                queries.append(
                    CandidateQuery(
                        query_ref=query_ref,
                        decision_ref=decision_ref,
                        state_ref=state_ref,
                        candidate_id=candidate_id,
                        tool_name=spec.tool_name,
                        mutation_class=(
                            "mutating" if spec.mutating else "read_only"
                        ),
                        current_victim_decision=current_decision,
                        constraint_refs=tuple(sorted(constraint_refs)),
                        metadata=metadata,
                    ).model_dump(mode="json")
                )

    if selected_episodes != len(selected):
        raise ValueError("selected task does not map to exactly one greedy episode")
    if len({row["query_ref"] for row in queries}) != len(queries):
        raise ValueError("duplicate executable query reference")
    cells = {
        (row["metadata"]["suite"], row["metadata"]["difficulty"])
        for row in queries
    }
    return {
        "state_catalog": dict(sorted(states.items())),
        "constraint_catalog": dict(sorted(constraints.items())),
        "decisions": sorted(decisions, key=lambda row: row["decision_ref"]),
        "queries": sorted(queries, key=lambda row: row["query_ref"]),
        "audit": {
            "tasks": selected_episodes,
            "states": len(states),
            "decisions": len(decisions),
            "terminal_decisions": sum(row["terminal_decision"] for row in decisions),
            "observed_transitions": sum(
                not row["terminal_decision"] for row in decisions
            ),
            "observed_constraint_rows": observed_transition_rows,
            "constraints": len(constraints),
            "executable_action_queries": len(queries),
            "candidate_constraint_queries": sum(
                len(row["constraint_refs"]) for row in queries
            ),
            "query_suite_difficulty_cells": len(cells),
        },
    }


def _argument_donors(
    raw_dataset: Mapping[str, Any],
    tool_specs: Mapping[str, ToolBindingSpec],
) -> dict[str, list[dict[str, Any]]]:
    donors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in raw_dataset["episodes"]:
        if episode["split"] != "training":
            continue
        for index in range(len(episode["prefixes"]) - 1):
            candidate_id = str(episode["prefixes"][index]["targets"]["next_action"])
            if candidate_id == "STOP" or candidate_id not in tool_specs:
                continue
            raw_arguments = episode["prefixes"][index + 1]["features"][
                "last_action"
            ].get("arguments", {})
            try:
                arguments = tool_specs[candidate_id].validate(raw_arguments)
            except Exception:
                continue
            donors[candidate_id].append(
                {
                    "task_id": str(episode["task_id"]),
                    "episode_id": str(episode["episode_id"]),
                    "track": str(episode["track"]),
                    "transition_index": index,
                    "arguments": arguments,
                }
            )
    return donors


def _choose_donor(
    pool: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    query: Mapping[str, Any],
) -> Mapping[str, Any]:
    return min(
        pool,
        key=lambda donor: _rank(
            seed,
            query["state_ref"],
            query["candidate_id"],
            donor["task_id"],
            donor["episode_id"],
            donor["track"],
            donor["transition_index"],
            donor["arguments"],
        ),
    )


def bind_query(
    query: Mapping[str, Any],
    *,
    tool_specs: Mapping[str, ToolBindingSpec],
    donors: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: str,
) -> ArgumentBinding | None:
    spec = tool_specs[str(query["candidate_id"])]
    if not spec.required_fields:
        return ArgumentBinding(source="SCHEMA_EMPTY", arguments=spec.validate({}))
    pool = list(donors.get(spec.candidate_id, ()))
    cross_task = [
        row for row in pool if row["task_id"] != query["metadata"]["task_id"]
    ]
    same_task = [
        row for row in pool if row["task_id"] == query["metadata"]["task_id"]
    ]
    if cross_task:
        source = "CROSS_TASK_OBSERVED_DONOR"
        donor = _choose_donor(cross_task, seed=seed, query=query)
    elif same_task:
        source = "SAME_TASK_CLEAN_DONOR"
        donor = _choose_donor(same_task, seed=seed, query=query)
    else:
        return None
    return ArgumentBinding(
        source=source,
        arguments=spec.validate(donor["arguments"]),
        donor_task_id=str(donor["task_id"]),
        donor_episode_id=str(donor["episode_id"]),
        donor_track=str(donor["track"]),
        donor_transition_index=int(donor["transition_index"]),
    )


def build_counterfactual_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    selected_task_ids: Sequence[str],
    suites: Sequence[str],
    difficulties: Sequence[str],
    tool_specs: Mapping[str, ToolBindingSpec],
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    universe = build_query_universe(
        raw_dataset,
        semantic_dataset,
        selected_task_ids=selected_task_ids,
        tool_specs=tool_specs,
    )
    donors = _argument_donors(raw_dataset, tool_specs)
    eligible = []
    ineligible = Counter()
    for query in universe["queries"]:
        binding = bind_query(
            query, tool_specs=tool_specs, donors=donors, seed=seed
        )
        if binding is None:
            ineligible["NO_CAUSAL_ARGUMENT_BINDING"] += 1
            continue
        eligible.append(
            CounterfactualManifestRow(
                manifest_row_ref=stable_fingerprint(
                    {
                        "query_ref": query["query_ref"],
                        "arguments": binding.arguments,
                    }
                ),
                query=CandidateQuery.model_validate(query),
                binding=binding,
            ).model_dump(mode="json")
        )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        metadata = row["query"]["metadata"]
        groups[
            (
                str(metadata["suite"]),
                str(metadata["difficulty"]),
                str(row["query"]["mutation_class"]),
            )
        ].append(row)
    selected = []
    for suite in suites:
        for difficulty in difficulties:
            used_states: set[str] = set()
            for mutation_class in ("read_only", "mutating"):
                pool = groups.get((suite, difficulty, mutation_class), [])
                if not pool:
                    raise ValueError(
                        "no eligible query for "
                        f"{suite}/{difficulty}/{mutation_class}"
                    )
                distinct = [
                    row
                    for row in pool
                    if row["query"]["state_ref"] not in used_states
                ]
                candidates = distinct or pool
                chosen = min(
                    candidates,
                    key=lambda row: _rank(
                        seed,
                        suite,
                        difficulty,
                        mutation_class,
                        row["query"]["state_ref"],
                        row["query"]["candidate_id"],
                    ),
                )
                selected.append(chosen)
                used_states.add(str(chosen["query"]["state_ref"]))
    selected.sort(key=lambda row: row["manifest_row_ref"])
    selection_cells = Counter(
        (
            row["query"]["metadata"]["suite"],
            row["query"]["metadata"]["difficulty"],
            row["query"]["mutation_class"],
        )
        for row in selected
    )
    manifest = {
        "schema_version": COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
        "scope": "clean-only AgentDojo synthetic sandbox counterfactual pilot",
        "selection_seed": seed,
        "selection_contract": {
            "outcome_blind": True,
            "training_split_only": True,
            "terminal_prefixes_included": True,
            "stop_is_never_executed_as_a_tool": True,
            "per_suite_difficulty": {"read_only": 1, "mutating": 1},
            "argument_binding_precedence": list(BINDING_SOURCES),
            "donor_outputs_or_outcomes_read": False,
        },
        "universe_audit": universe["audit"],
        "rows": selected,
    }
    audit = {
        **universe["audit"],
        "eligible_action_queries": len(eligible),
        "ineligible_action_queries": dict(sorted(ineligible.items())),
        "selected_action_queries": len(selected),
        "selected_unique_tasks": len(
            {row["query"]["metadata"]["task_id"] for row in selected}
        ),
        "selected_unique_states": len(
            {row["query"]["state_ref"] for row in selected}
        ),
        "selected_unique_tools": len(
            {row["query"]["candidate_id"] for row in selected}
        ),
        "selected_mutation_class": dict(
            sorted(Counter(row["query"]["mutation_class"] for row in selected).items())
        ),
        "selected_binding_source": dict(
            sorted(Counter(row["binding"]["source"] for row in selected).items())
        ),
        "selected_constraint_rows": sum(
            len(row["query"]["constraint_refs"]) for row in selected
        ),
        "selection_cells": {
            "::".join(key): value for key, value in sorted(selection_cells.items())
        },
        "selection_uses_outcomes": False,
        "stop_selected": any(
            row["query"]["candidate_id"] == "STOP" for row in selected
        ),
    }
    return manifest, audit
