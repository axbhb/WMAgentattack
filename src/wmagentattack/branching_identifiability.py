"""Outcome-blind paired branch selection and causal-effect auditing.

The selector deliberately creates several legal actions from the same clean
AgentDojo prefix.  Selection may inspect only causal semantic state, legal tool
schemas, and already-observed clean argument donors.  It never reads a fresh
counterfactual outcome, task utility, security labels, or expert future calls.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .clean_state_instrumentation import infer_argument_entity_links
from .counterfactual_evidence import (
    ArgumentBinding,
    CandidateQuery,
    CounterfactualManifestRow,
    ToolBindingSpec,
    _argument_donors,
    _rank,
    build_query_universe,
)
from .decision_state import canonical_json_value, stable_fingerprint


BRANCH_MANIFEST_SCHEMA_VERSION = "wmagentattack.paired_branch_manifest.v17"


def _binding_candidates(
    query: Mapping[str, Any],
    *,
    spec: ToolBindingSpec,
    donors: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ArgumentBinding]:
    if not spec.required_fields:
        return [ArgumentBinding(source="SCHEMA_EMPTY", arguments=spec.validate({}))]
    rows = []
    seen = set()
    for donor in donors.get(spec.candidate_id, ()):
        arguments = spec.validate(donor["arguments"])
        fingerprint = stable_fingerprint(arguments)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        same_task = str(donor["task_id"]) == str(query["metadata"]["task_id"])
        rows.append(
            ArgumentBinding(
                source=(
                    "SAME_TASK_CLEAN_DONOR"
                    if same_task
                    else "CROSS_TASK_OBSERVED_DONOR"
                ),
                arguments=arguments,
                donor_task_id=str(donor["task_id"]),
                donor_episode_id=str(donor["episode_id"]),
                donor_track=str(donor["track"]),
                donor_transition_index=int(donor["transition_index"]),
            )
        )
    return rows


def _resolution_counts(arguments: Mapping[str, Any], state: Mapping[str, Any]) -> Counter:
    return Counter(
        row.resolution for row in infer_argument_entity_links(arguments, state)
    )


def _binding_rank(
    binding: ArgumentBinding,
    *,
    query: Mapping[str, Any],
    state: Mapping[str, Any],
    previous_rows: set[str],
    seed: str,
) -> tuple[Any, ...]:
    counts = _resolution_counts(binding.arguments, state)
    row_ref = stable_fingerprint(
        {"query_ref": query["query_ref"], "arguments": binding.arguments}
    )
    novel = row_ref not in previous_rows
    cross_task = binding.source == "CROSS_TASK_OBSERVED_DONOR"
    if query["mutation_class"] == "mutating":
        structural = (counts["no_match"], counts["ambiguous"], counts["unique"])
    else:
        structural = (counts["ambiguous"], counts["no_match"], counts["unique"])
    # ``min`` is used by callers, hence negate quantities that should be large.
    return (
        not novel,
        *(-value for value in structural),
        not cross_task,
        _rank(seed, query["state_ref"], query["candidate_id"], binding.arguments),
    )


def _best_bound_query(
    query: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    tool_specs: Mapping[str, ToolBindingSpec],
    donors: Mapping[str, Sequence[Mapping[str, Any]]],
    previous_rows: set[str],
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    spec = tool_specs[str(query["candidate_id"])]
    bindings = _binding_candidates(query, spec=spec, donors=donors)
    if not bindings:
        return None
    binding = min(
        bindings,
        key=lambda row: _binding_rank(
            row,
            query=query,
            state=state,
            previous_rows=previous_rows,
            seed=seed,
        ),
    )
    counts = _resolution_counts(binding.arguments, state)
    manifest_row = CounterfactualManifestRow(
        manifest_row_ref=stable_fingerprint(
            {"query_ref": query["query_ref"], "arguments": binding.arguments}
        ),
        query=CandidateQuery.model_validate(query),
        binding=binding,
    ).model_dump(mode="json")
    audit = {
        "manifest_row_ref": manifest_row["manifest_row_ref"],
        "decision_ref": query["decision_ref"],
        "state_ref": query["state_ref"],
        "task_id": query["metadata"]["task_id"],
        "prefix_index": query["metadata"]["prefix_index"],
        "candidate_id": query["candidate_id"],
        "mutation_class": query["mutation_class"],
        "binding_source": binding.source,
        "argument_resolution": dict(sorted(counts.items())),
        "novel_vs_0805_exact_row": manifest_row["manifest_row_ref"]
        not in previous_rows,
    }
    return manifest_row, audit


def build_paired_branch_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    selected_task_ids: Sequence[str],
    tool_specs: Mapping[str, ToolBindingSpec],
    seed: str,
    actions_per_class: int,
    previous_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose one anchor and several legal action branches per task."""

    if actions_per_class < 1:
        raise ValueError("actions_per_class must be positive")
    universe = build_query_universe(
        raw_dataset,
        semantic_dataset,
        selected_task_ids=selected_task_ids,
        tool_specs=tool_specs,
    )
    donors = _argument_donors(raw_dataset, tool_specs)
    previous_rows = {
        str(row["manifest_row_ref"])
        for row in (previous_manifest or {}).get("rows", ())
    }
    by_decision: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for query in universe["queries"]:
        state = universe["state_catalog"][query["state_ref"]]
        bound = _best_bound_query(
            query,
            state=state,
            tool_specs=tool_specs,
            donors=donors,
            previous_rows=previous_rows,
            seed=seed,
        )
        if bound is not None:
            by_decision[str(query["decision_ref"])].append(bound)

    anchors_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision_ref, bound_rows in by_decision.items():
        classes = Counter(
            row[0]["query"]["mutation_class"] for row in bound_rows
        )
        if any(classes[kind] < actions_per_class for kind in ("read_only", "mutating")):
            continue
        sample = bound_rows[0][0]["query"]
        state = universe["state_catalog"][sample["state_ref"]]
        evidence = state.get("goal_evidence", {})
        anchors_by_task[str(sample["metadata"]["task_id"])].append(
            {
                "decision_ref": decision_ref,
                "state_ref": sample["state_ref"],
                "prefix_index": int(sample["metadata"]["prefix_index"]),
                "evidence_records": len(state.get("evidence_records", ())),
                "ambiguous_records": int(evidence.get("ambiguous_entity_records", 0)),
                "conflicts": int(evidence.get("conflict_count", 0)),
                "bound_rows": bound_rows,
            }
        )

    selected_rows = []
    selected_audit = []
    anchor_audit = []
    for task_id in selected_task_ids:
        anchors = anchors_by_task.get(str(task_id), [])
        if not anchors:
            raise ValueError(f"no four-branch anchor for {task_id}")
        anchor = min(
            anchors,
            key=lambda row: (
                -row["conflicts"],
                -row["ambiguous_records"],
                -row["evidence_records"],
                -row["prefix_index"],
                _rank(seed, task_id, row["decision_ref"]),
            ),
        )
        anchor_audit.append(
            {key: value for key, value in anchor.items() if key != "bound_rows"}
        )
        for mutation_class in ("read_only", "mutating"):
            candidates = [
                row
                for row in anchor["bound_rows"]
                if row[0]["query"]["mutation_class"] == mutation_class
            ]
            candidates.sort(
                key=lambda pair: (
                    not pair[1]["novel_vs_0805_exact_row"],
                    -pair[1]["argument_resolution"].get(
                        "ambiguous" if mutation_class == "read_only" else "no_match",
                        0,
                    ),
                    -pair[1]["argument_resolution"].get("no_match", 0),
                    _rank(
                        seed,
                        task_id,
                        anchor["state_ref"],
                        pair[0]["query"]["candidate_id"],
                    ),
                )
            )
            chosen = candidates[:actions_per_class]
            selected_rows.extend(row for row, _ in chosen)
            selected_audit.extend(audit for _, audit in chosen)

    selected_rows.sort(key=lambda row: row["manifest_row_ref"])
    expected_rows = len(selected_task_ids) * actions_per_class * 2
    if len(selected_rows) != expected_rows:
        raise ValueError("paired branch budget mismatch")
    manifest = {
        "schema_version": BRANCH_MANIFEST_SCHEMA_VERSION,
        "scope": "clean-only same-prefix AgentDojo paired branch pilot",
        "selection_seed": seed,
        "selection_contract": {
            "outcome_blind": True,
            "training_split_only": True,
            "one_anchor_per_task": True,
            "actions_per_anchor": actions_per_class * 2,
            "actions_per_class": actions_per_class,
            "argument_donor_outputs_or_outcomes_read": False,
            "previous_exact_rows_are_deprioritized": True,
        },
        "anchor_audit_only": anchor_audit,
        "row_selection_audit_only": selected_audit,
        "rows": selected_rows,
    }
    per_anchor = Counter(row["query"]["decision_ref"] for row in selected_rows)
    per_class = Counter(row["query"]["mutation_class"] for row in selected_rows)
    audit = {
        **universe["audit"],
        "eligible_bound_queries": sum(len(rows) for rows in by_decision.values()),
        "selected_rows": len(selected_rows),
        "selected_anchors": len(per_anchor),
        "selected_tasks": len(
            {row["query"]["metadata"]["task_id"] for row in selected_rows}
        ),
        "selected_suite_difficulty_cells": len(
            {
                (
                    row["query"]["metadata"]["suite"],
                    row["query"]["metadata"]["difficulty"],
                )
                for row in selected_rows
            }
        ),
        "selected_per_class": dict(sorted(per_class.items())),
        "all_anchors_have_exact_budget": all(
            count == actions_per_class * 2 for count in per_anchor.values()
        ),
        "novel_exact_rows": sum(
            row["novel_vs_0805_exact_row"] for row in selected_audit
        ),
        "selection_uses_outcomes": False,
        "expected_prior_replay_executions_two_replicas": sum(
            int(row["query"]["metadata"]["prefix_index"]) * 2
            for row in selected_rows
        ),
    }
    return canonical_json_value(manifest), canonical_json_value(audit)


def _effect_projection(outcome: Mapping[str, Any]) -> dict[str, Any]:
    visible = outcome["model_visible"]
    next_state = visible["next_semantic_state"]
    evidence = next_state["goal_evidence"]
    execution = next_state["execution"]
    return canonical_json_value(
        {
            "execution_status": visible["events"]["execution_status"],
            "cumulative_errors": execution["cumulative_errors"],
            "matched_fact_terms": evidence["matched_fact_terms"],
            "unmatched_fact_terms": evidence["unmatched_fact_terms"],
            "unique_entity_records": evidence["unique_entity_records"],
            "ambiguous_entity_records": evidence["ambiguous_entity_records"],
            "unlinked_entity_records": evidence["unlinked_entity_records"],
            "conflict_count": evidence["conflict_count"],
            "observed_entity_types": evidence["observed_entity_types"],
            "state_changed": outcome["simulator_audit_only"]["state_changed"],
            "state_delta_roots": outcome["simulator_audit_only"][
                "state_delta_roots"
            ],
        }
    )


def audit_paired_branch_effects(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    outcomes = {
        row["manifest_row_ref"]: row for row in dataset["counterfactual_outcomes"]
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for manifest_row in manifest["rows"]:
        outcome = outcomes.get(manifest_row["manifest_row_ref"])
        if outcome is not None:
            grouped[str(manifest_row["query"]["decision_ref"])].append(outcome)

    anchor_rows = []
    pairwise_total = 0
    pairwise_different = 0
    for decision_ref, rows in sorted(grouped.items()):
        projections = [_effect_projection(row) for row in rows]
        fingerprints = [stable_fingerprint(row) for row in projections]
        for left in range(len(fingerprints)):
            for right in range(left + 1, len(fingerprints)):
                pairwise_total += 1
                pairwise_different += fingerprints[left] != fingerprints[right]
        anchor_rows.append(
            {
                "decision_ref": decision_ref,
                "actions": len(rows),
                "distinct_effect_projections": len(set(fingerprints)),
                "execution_statuses": sorted(
                    {
                        row["model_visible"]["events"]["execution_status"]
                        for row in rows
                    }
                ),
                "state_changed_values": sorted(
                    {row["simulator_audit_only"]["state_changed"] for row in rows}
                ),
            }
        )
    boundary = {
        "execution_error": sum(
            row["model_visible"]["events"]["execution_status"] == "error"
            for row in outcomes.values()
        ),
        "conflict": sum(
            row["model_visible"]["events"]["conflict_count_delta"] > 0
            for row in outcomes.values()
        ),
        "ambiguity": sum(
            row["model_visible"]["events"]["unresolved_entity_count_delta"] > 0
            for row in outcomes.values()
        ),
    }
    return {
        "anchors": len(anchor_rows),
        "complete_four_action_anchors": sum(row["actions"] == 4 for row in anchor_rows),
        "anchors_with_two_effects": sum(
            row["distinct_effect_projections"] >= 2 for row in anchor_rows
        ),
        "anchors_with_three_effects": sum(
            row["distinct_effect_projections"] >= 3 for row in anchor_rows
        ),
        "anchors_with_status_or_state_change_diversity": sum(
            len(row["execution_statuses"]) > 1 or len(row["state_changed_values"]) > 1
            for row in anchor_rows
        ),
        "pairwise_effect_difference_fraction": pairwise_different
        / max(1, pairwise_total),
        "pairwise_effect_comparisons": pairwise_total,
        "boundary_events": boundary,
        "boundary_event_total": sum(boundary.values()),
        "boundary_event_types_present": sum(value > 0 for value in boundary.values()),
        "anchor_rows": anchor_rows,
    }
