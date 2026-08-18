"""Minimal paired parameter interventions for AgentDojo action boundaries."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .counterfactual_evidence import (
    CandidateQuery,
    CounterfactualManifestRow,
    ToolBindingSpec,
    _argument_donors,
    bind_query,
    build_query_universe,
)
from .decision_state import canonical_json_value, stable_fingerprint


PARAMETER_INTERVENTION_SCHEMA_VERSION = (
    "wmagentattack.parameter_intervention_manifest.v18"
)

TARGET_TOOL_BY_SUITE = {
    "banking": "update_scheduled_transaction",
    "slack": "add_user_to_channel",
    "travel": "reserve_hotel",
    "workspace": "share_file",
}

CORRUPTION_BY_SUITE = {
    "banking": ("id", 2_147_483_647),
    "slack": ("user", "__WMAGENTATTACK_MISSING_USER_V18__"),
    "travel": ("start_day", "not-a-date-v18"),
    "workspace": ("file_id", "__WMAGENTATTACK_MISSING_FILE_V18__"),
}


def _row(query: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    return CounterfactualManifestRow(
        manifest_row_ref=stable_fingerprint(
            {"query_ref": query["query_ref"], "arguments": binding["arguments"]}
        ),
        query=CandidateQuery.model_validate(query),
        binding=binding,
    ).model_dump(mode="json")


def build_parameter_intervention_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    selected_task_ids: Sequence[str],
    tool_specs: Mapping[str, ToolBindingSpec],
    seed: str,
    base_branch_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pair one valid control with one schema-valid precondition corruption."""

    universe = build_query_universe(
        raw_dataset,
        semantic_dataset,
        selected_task_ids=selected_task_ids,
        tool_specs=tool_specs,
    )
    donors = _argument_donors(raw_dataset, tool_specs)
    base_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_branch_manifest["rows"]:
        base_by_task[str(row["query"]["metadata"]["task_id"])].append(row)

    rows = []
    pairs = []
    for task_id in selected_task_ids:
        base_rows = base_by_task.get(str(task_id), [])
        if len(base_rows) != 4:
            raise ValueError(f"v17 base anchor is incomplete for {task_id}")
        decision_refs = {str(row["query"]["decision_ref"]) for row in base_rows}
        if len(decision_refs) != 1:
            raise ValueError("v17 rows do not share one root")
        decision_ref = next(iter(decision_refs))
        suite = str(base_rows[0]["query"]["metadata"]["suite"])
        target_tool = TARGET_TOOL_BY_SUITE[suite]
        candidates = [
            query
            for query in universe["queries"]
            if str(query["decision_ref"]) == decision_ref
            and str(query["tool_name"]) == target_tool
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one {target_tool} query at v17 root for {task_id}"
            )
        query = candidates[0]
        spec = tool_specs[str(query["candidate_id"])]
        base_match = [
            row for row in base_rows if row["query"]["query_ref"] == query["query_ref"]
        ]
        if base_match:
            valid_binding = canonical_json_value(base_match[0]["binding"])
        else:
            selected = bind_query(
                query,
                tool_specs=tool_specs,
                donors=donors,
                seed=seed,
            )
            if selected is None:
                raise ValueError(f"no valid control binding for {task_id}")
            valid_binding = selected.model_dump(mode="json")
        valid_binding["arguments"] = spec.validate(valid_binding["arguments"])
        field, corrupted_value = CORRUPTION_BY_SUITE[suite]
        if field not in valid_binding["arguments"]:
            raise ValueError(f"control binding lacks intervention field {field}")
        corrupt_binding = copy.deepcopy(valid_binding)
        corrupt_binding["arguments"][field] = corrupted_value
        corrupt_binding["arguments"] = spec.validate(corrupt_binding["arguments"])

        changed_fields = [
            key
            for key in sorted(
                set(valid_binding["arguments"]) | set(corrupt_binding["arguments"])
            )
            if valid_binding["arguments"].get(key)
            != corrupt_binding["arguments"].get(key)
        ]
        if changed_fields != [field]:
            raise ValueError("parameter intervention must change exactly one field")
        control = _row(query, valid_binding)
        corrupted = _row(query, corrupt_binding)
        pair_ref = stable_fingerprint(
            {"decision_ref": decision_ref, "query_ref": query["query_ref"], "field": field}
        )
        rows.extend((control, corrupted))
        pairs.append(
            {
                "pair_ref": pair_ref,
                "task_id": task_id,
                "suite": suite,
                "difficulty": query["metadata"]["difficulty"],
                "decision_ref": decision_ref,
                "state_ref": query["state_ref"],
                "prefix_index": query["metadata"]["prefix_index"],
                "candidate_id": query["candidate_id"],
                "tool_name": target_tool,
                "intervention_field": field,
                "control_row_ref": control["manifest_row_ref"],
                "corrupted_row_ref": corrupted["manifest_row_ref"],
                "changed_fields": changed_fields,
                "schema_valid_before_execution": True,
            }
        )

    rows.sort(key=lambda row: row["manifest_row_ref"])
    pairs.sort(key=lambda row: row["pair_ref"])
    manifest = {
        "schema_version": PARAMETER_INTERVENTION_SCHEMA_VERSION,
        "scope": "clean-only same-state same-tool parameter intervention pilot",
        "selection_seed": seed,
        "selection_contract": {
            "outcome_blind": True,
            "same_state_and_tool_within_pair": True,
            "exactly_one_argument_field_changed": True,
            "schema_valid_corruption": True,
            "simulator_precondition_outcome_unread": True,
            "task_utility_security_and_future_calls_unread": True,
        },
        "pair_audit_only": pairs,
        "rows": rows,
    }
    audit = {
        **universe["audit"],
        "pairs": len(pairs),
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in pairs}),
        "suite_difficulty_cells": len(
            {(row["suite"], row["difficulty"]) for row in pairs}
        ),
        "target_tools": dict(sorted(Counter(row["tool_name"] for row in pairs).items())),
        "intervention_fields": dict(
            sorted(Counter(row["intervention_field"] for row in pairs).items())
        ),
        "all_pairs_change_exactly_one_field": all(
            len(row["changed_fields"]) == 1 for row in pairs
        ),
        "selection_uses_outcomes": False,
        "expected_prior_replay_executions_two_replicas": sum(
            int(row["prefix_index"]) * 4 for row in pairs
        ),
    }
    return canonical_json_value(manifest), canonical_json_value(audit)


def audit_parameter_interventions(
    dataset: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    outcomes = {
        row["manifest_row_ref"]: row for row in dataset["counterfactual_outcomes"]
    }
    pair_rows = []
    for pair in manifest["pair_audit_only"]:
        control = outcomes.get(pair["control_row_ref"])
        corrupted = outcomes.get(pair["corrupted_row_ref"])
        if control is None or corrupted is None:
            continue
        control_status = control["model_visible"]["events"]["execution_status"]
        corrupt_status = corrupted["model_visible"]["events"]["execution_status"]
        control_effect = {
            "status": control_status,
            "state_changed": control["simulator_audit_only"]["state_changed"],
            "delta_roots": control["simulator_audit_only"]["state_delta_roots"],
            "matched": control["model_visible"]["next_semantic_state"]["goal_evidence"][
                "matched_fact_terms"
            ],
        }
        corrupt_effect = {
            "status": corrupt_status,
            "state_changed": corrupted["simulator_audit_only"]["state_changed"],
            "delta_roots": corrupted["simulator_audit_only"]["state_delta_roots"],
            "matched": corrupted["model_visible"]["next_semantic_state"][
                "goal_evidence"
            ]["matched_fact_terms"],
        }
        pair_rows.append(
            {
                "pair_ref": pair["pair_ref"],
                "task_id": pair["task_id"],
                "suite": pair["suite"],
                "tool_name": pair["tool_name"],
                "control_status": control_status,
                "corrupted_status": corrupt_status,
                "paired_status_flip": control_status == "success"
                and corrupt_status == "error",
                "effect_changed": stable_fingerprint(control_effect)
                != stable_fingerprint(corrupt_effect),
                "corrupted_error_type": corrupted["simulator_audit_only"][
                    "tool_error_type"
                ],
            }
        )
    return {
        "complete_pairs": len(pair_rows),
        "control_successes": sum(row["control_status"] == "success" for row in pair_rows),
        "corrupted_errors": sum(row["corrupted_status"] == "error" for row in pair_rows),
        "paired_status_flips": sum(row["paired_status_flip"] for row in pair_rows),
        "pairs_with_effect_change": sum(row["effect_changed"] for row in pair_rows),
        "suites_with_status_flip": len(
            {row["suite"] for row in pair_rows if row["paired_status_flip"]}
        ),
        "error_types": dict(
            sorted(
                Counter(
                    row["corrupted_error_type"]
                    for row in pair_rows
                    if row["corrupted_error_type"] is not None
                ).items()
            )
        ),
        "pair_rows": pair_rows,
    }
