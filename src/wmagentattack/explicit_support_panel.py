"""Outcome-blind rare-mechanism support branches for the v25 data gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .branching_identifiability import _best_bound_query
from .compositional_effect_world_model import parse_effect_token
from .counterfactual_evidence import _argument_donors, _rank, build_query_universe
from .decision_state import canonical_json_value, stable_fingerprint
from .intervention_union import effect_descriptor, effect_tokens, normalized_action_descriptor
from .semantic_state_v3 import find_semantic_state_v3_leakage


SCHEMA_VERSION = "wmagentattack.explicit_atom_support.v25"


def effect_slot_atoms(token: str) -> list[str]:
    slots = parse_effect_token(token)
    atoms = [f"category::{slots['category']}"]
    for key in ("entity", "field", "kind", "value"):
        value = slots[key]
        if value:
            atoms.append(f"{key}::{value}")
    return sorted(set(atoms))


def build_explicit_support_manifest(
    raw_dataset: Mapping[str, Any],
    semantic_dataset: Mapping[str, Any],
    *,
    target_tools_by_task: Mapping[str, Sequence[str]],
    tool_specs: Mapping[str, Any],
    seed: str,
    anchors_per_task: int,
    confirmation_task_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose multiple roots per support task without reading branch outcomes."""

    if anchors_per_task < 1:
        raise ValueError("anchors_per_task must be positive")
    selected_tasks = sorted(target_tools_by_task)
    if set(selected_tasks) & set(confirmation_task_ids):
        raise ValueError("support and confirmation tasks overlap")
    universe = build_query_universe(
        raw_dataset,
        semantic_dataset,
        selected_task_ids=selected_tasks,
        tool_specs=tool_specs,
    )
    donors = _argument_donors(raw_dataset, tool_specs)
    wanted = {
        task: tuple(sorted(str(tool) for tool in tools))
        for task, tools in target_tools_by_task.items()
    }
    by_task_decision: dict[str, dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for query in universe["queries"]:
        task = str(query["metadata"]["task_id"])
        candidate = str(query["candidate_id"])
        if candidate not in wanted.get(task, ()):
            continue
        state = universe["state_catalog"][query["state_ref"]]
        bound = _best_bound_query(
            query,
            state=state,
            tool_specs=tool_specs,
            donors=donors,
            previous_rows=set(),
            seed=seed,
        )
        if bound is not None:
            by_task_decision[task][str(query["decision_ref"])][candidate] = bound

    selected_rows: list[dict[str, Any]] = []
    anchor_audit = []
    for task in selected_tasks:
        required = set(wanted[task])
        anchors = []
        for decision_ref, candidates in by_task_decision[task].items():
            if set(candidates) != required:
                continue
            sample = next(iter(candidates.values()))[0]["query"]
            state = universe["state_catalog"][sample["state_ref"]]
            anchors.append(
                {
                    "decision_ref": decision_ref,
                    "state_ref": sample["state_ref"],
                    "prefix_index": int(sample["metadata"]["prefix_index"]),
                    "evidence_records": len(state.get("evidence_records", ())),
                    "candidates": candidates,
                }
            )
        anchors.sort(
            key=lambda row: (
                -row["evidence_records"],
                -row["prefix_index"],
                _rank(seed, task, row["decision_ref"]),
            )
        )
        if len(anchors) < anchors_per_task:
            raise ValueError(f"insufficient target-complete anchors for {task}")
        for anchor in anchors[:anchors_per_task]:
            anchor_audit.append(
                {key: value for key, value in anchor.items() if key != "candidates"}
                | {"task_id": task, "target_tools": sorted(required)}
            )
            for candidate in sorted(required):
                selected_rows.append(anchor["candidates"][candidate][0])

    selected_rows.sort(key=lambda row: row["manifest_row_ref"])
    per_tool = Counter(row["query"]["candidate_id"] for row in selected_rows)
    per_task = Counter(row["query"]["metadata"]["task_id"] for row in selected_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "clean-only AgentDojo rare-mechanism atom-support branches",
        "selection_seed": seed,
        "selection_contract": {
            "outcome_blind": True,
            "support_tasks_never_enter_confirmation": True,
            "fresh_branch_outcomes_not_read": True,
            "composite_effect_labels_are_audit_only": True,
            "model_training_receives_slot_atoms_only": True,
        },
        "anchor_audit_only": anchor_audit,
        "rows": selected_rows,
    }
    audit = {
        **universe["audit"],
        "selected_rows": len(selected_rows),
        "selected_tasks": len(per_task),
        "selected_anchors": len(anchor_audit),
        "selected_per_tool": dict(sorted(per_tool.items())),
        "selected_per_task": dict(sorted(per_task.items())),
        "support_confirmation_task_overlap": sorted(
            set(per_task) & set(confirmation_task_ids)
        ),
        "selection_uses_outcomes": False,
        "expected_prior_replay_executions_two_replicas": sum(
            int(row["query"]["metadata"]["prefix_index"]) * 2
            for row in selected_rows
        ),
    }
    return canonical_json_value(manifest), canonical_json_value(audit)


def _hard_unseen_occurrences(
    hard_dataset: Mapping[str, Any], fold: int
) -> list[tuple[Mapping[str, Any], str]]:
    rows = hard_dataset["transitions"]
    observed = {
        token
        for row in rows
        if int(row["confirmation_fold"]) != fold
        for token in row["model_target"]["effect_tokens"]
    }
    return [
        (row, token)
        for row in rows
        if int(row["confirmation_fold"]) == fold
        for token in row["model_target"]["effect_tokens"]
        if token not in observed
    ]


def build_atom_support_dataset_and_gate(
    execution_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hard_dataset: Mapping[str, Any],
    *,
    confirmation_task_ids: Sequence[str],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_by_ref = {row["manifest_row_ref"]: row for row in manifest["rows"]}
    support_rows = []
    leakage: dict[str, list[str]] = {}
    for outcome in execution_dataset["counterfactual_outcomes"]:
        manifest_row = manifest_by_ref[outcome["manifest_row_ref"]]
        current_ref = outcome["model_visible"]["current_state_ref"]
        current = execution_dataset["current_state_catalog"][current_ref]
        following = outcome["model_visible"]["next_semantic_state"]
        effect = effect_descriptor(current, following)
        composites = effect_tokens(effect)
        atoms = sorted({atom for token in composites for atom in effect_slot_atoms(token)})
        action = outcome["model_visible"]["candidate_action"]
        normalized_action = normalized_action_descriptor(action, current)
        for state in (current, following):
            findings = find_semantic_state_v3_leakage(state)
            if findings:
                leakage[stable_fingerprint(state)] = list(findings)
        support_rows.append(
            canonical_json_value(
                {
                    "support_ref": stable_fingerprint(
                        {"manifest_row_ref": outcome["manifest_row_ref"], "atoms": atoms}
                    ),
                    "task_id_split_only": outcome["metadata"]["task_id"],
                    "suite_split_only": outcome["metadata"]["suite"],
                    "model_input": {
                        "current_semantic_state": current,
                        "normalized_action": normalized_action,
                    },
                    "model_target": {
                        "effect_slot_atoms": atoms,
                        "execution_error": int(effect["execution_status"] == "error"),
                    },
                    "audit_only": {
                        "composite_effect_tokens": composites,
                        "full_effect": effect,
                        "manifest_row_ref": outcome["manifest_row_ref"],
                        "replicas_identical": next(
                            row["identical"]
                            for row in execution_dataset["replica_verification"]
                            if row["manifest_row_ref"] == outcome["manifest_row_ref"]
                        ),
                    },
                }
            )
        )
    support_rows.sort(key=lambda row: row["support_ref"])

    support_atoms = {
        atom for row in support_rows for atom in row["model_target"]["effect_slot_atoms"]
    }
    support_tools = Counter(
        row["model_input"]["normalized_action"]["tool_id"] for row in support_rows
    )
    fold_rows = []
    totals = Counter()
    hits = Counter()
    scalar_unseen = Counter()
    for fold in range(3):
        occurrences = _hard_unseen_occurrences(hard_dataset, fold)
        train_atoms = {
            atom
            for row in hard_dataset["transitions"]
            if int(row["confirmation_fold"]) != fold
            for token in row["model_target"]["effect_tokens"]
            for atom in effect_slot_atoms(token)
        } | support_atoms
        fold_counts = Counter()
        fold_hits = Counter()
        for row, token in occurrences:
            slots = parse_effect_token(token)
            focused = bool(slots["entity"] or (slots["category"] == "attribute" and slots["field"]))
            for slot in ("entity", "field", "kind"):
                value = slots[slot]
                if focused and value:
                    fold_counts[slot] += 1
                    totals[slot] += 1
                    if f"{slot}::{value}" in train_atoms:
                        fold_hits[slot] += 1
                        hits[slot] += 1
            if focused:
                tool_id = row["model_input"]["normalized_action"]["tool_id"]
                fold_counts["operation"] += 1
                totals["operation"] += 1
                if tool_id in support_tools:
                    fold_hits["operation"] += 1
                    hits["operation"] += 1
            else:
                scalar_unseen[token] += 1
        fold_rows.append(
            {
                "fold": fold,
                "unseen_positive_occurrences": len(occurrences),
                "focused_entity_or_attribute_occurrences": fold_counts["operation"],
                "coverage": {
                    key: fold_hits[key] / max(1, fold_counts[key])
                    for key in ("entity", "field", "kind", "operation")
                },
            }
        )
    coverage = {
        key: hits[key] / max(1, totals[key])
        for key in ("entity", "field", "kind", "operation")
    }
    support_tasks = {row["task_id_split_only"] for row in support_rows}
    success_rows = sum(
        row["model_target"]["execution_error"] == 0 for row in support_rows
    )
    rows_per_tool = dict(sorted(support_tools.items()))
    gate_checks = {
        "exact_support_rows": len(support_rows) == int(thresholds["required_support_rows"]),
        "exact_support_tasks": len(support_tasks) == int(thresholds["required_support_tasks"]),
        "all_target_tools_present": len(support_tools) == int(thresholds["required_target_tools"]),
        "minimum_rows_per_tool": min(support_tools.values(), default=0)
        >= int(thresholds["minimum_rows_per_tool"]),
        "all_support_executions_successful": success_rows == len(support_rows),
        "zero_support_confirmation_task_overlap": not (
            support_tasks & set(confirmation_task_ids)
        ),
        "zero_semantic_leakage": not leakage,
        "entity_atom_coverage": coverage["entity"]
        >= float(thresholds["minimum_entity_coverage"]),
        "field_atom_coverage": coverage["field"]
        >= float(thresholds["minimum_field_coverage"]),
        "kind_atom_coverage": coverage["kind"]
        >= float(thresholds["minimum_kind_coverage"]),
        "operation_coverage": coverage["operation"]
        >= float(thresholds["minimum_operation_coverage"]),
        "unseen_confirmation_signal_retained": sum(
            row["unseen_positive_occurrences"] for row in fold_rows
        ) >= int(thresholds["minimum_unseen_positive_occurrences"]),
        "composite_effect_labels_disabled": True,
    }
    passed = all(gate_checks.values())
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "scope": "clean-only task-disjoint explicit rare-mechanism atom support",
        "loader_contract": {
            "composite_effect_tokens_are_audit_only": True,
            "effect_head_must_not_read_audit_only": True,
            "support_task_ids_are_split_only": True,
            "support_rows_never_enter_confirmation": True,
            "utility_security_attack_and_final_outcomes_absent": True,
        },
        "atom_vocabulary": sorted(support_atoms),
        "support_rows": support_rows,
    }
    gate = {
        "decision": "GO_SUPPORT_CONDITIONED_MODEL_V25" if passed else "NO_GO_SUPPORT_DATA_V25",
        "gate_checks": gate_checks,
        "coverage": coverage,
        "coverage_denominators": dict(totals),
        "fold_diagnostics": fold_rows,
        "scalar_unseen_counterevidence": dict(sorted(scalar_unseen.items())),
        "support_rows": len(support_rows),
        "support_tasks": len(support_tasks),
        "support_tools": rows_per_tool,
        "successful_support_rows": success_rows,
        "semantic_leakage": leakage,
    }
    return canonical_json_value(dataset), canonical_json_value(gate)
