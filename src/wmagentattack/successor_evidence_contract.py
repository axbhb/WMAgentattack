"""Typed successor-evidence targets and deterministic rendering for v27.

The learned target is a bound evidence-state delta, not a canonical effect
label.  Canonical effects, including matched_count, are rendered exactly from
the predicted structured delta plus the explicit action.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .compositional_effect_world_model import parse_effect_token
from .decision_state import canonical_json_value, stable_fingerprint
from .intervention_union import effect_descriptor, effect_tokens, normalized_action_descriptor
from .semantic_state_v3 import find_semantic_state_v3_leakage


SCHEMA_VERSION = "wmagentattack.successor_evidence_contract.v27"
FORBIDDEN_MODEL_INPUT_KEYS = {
    "task_id", "suite", "difficulty", "confirmation_fold", "source_versions",
    "utility", "security", "attack", "final_outcome", "audit_only",
}


def _insert(catalog: dict[str, Any], ref: str, state: Mapping[str, Any]) -> None:
    payload = canonical_json_value(state)
    if stable_fingerprint(payload) != ref:
        raise ValueError(f"semantic state reference mismatch: {ref}")
    if ref in catalog and catalog[ref] != payload:
        raise ValueError(f"semantic state hash collision: {ref}")
    catalog[ref] = payload


def reconstruct_state_catalog(
    v17: Mapping[str, Any], v18: Mapping[str, Any], v19: Mapping[str, Any]
) -> dict[str, Any]:
    catalog: dict[str, Any] = {}
    for dataset in (v17, v18):
        for field in ("current_state_catalog", "next_state_catalog"):
            for ref, state in dataset[field].items():
                _insert(catalog, str(ref), state)
    for sequence in v19["sequences"]:
        root_ref = str(sequence["model_visible"]["root_state_ref"])
        if root_ref not in catalog:
            raise ValueError(f"v19 root absent from v17/v18 catalogs: {root_ref}")
        expected_step = int(catalog[root_ref]["step_index"]) + 1
        for step in sorted(sequence["model_visible"]["steps"], key=lambda row: int(row["step_index"])):
            state = step["next_semantic_state"]
            if int(state["step_index"]) != expected_step:
                raise ValueError("v19 semantic states are not adjacent")
            _insert(catalog, str(step["next_state_ref"]), state)
            expected_step += 1
    return catalog


def structured_successor_delta(
    current: Mapping[str, Any], following: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    effect = effect_descriptor(current, following)
    goal_terms = list(current.get("goal", {}).get("fact_terms", ()))
    index = {str(term): position for position, term in enumerate(goal_terms)}
    current_matched = set(current.get("goal_evidence", {}).get("matched_fact_terms", ()))
    next_matched = set(following.get("goal_evidence", {}).get("matched_fact_terms", ()))
    new_terms = sorted(next_matched - current_matched)
    missing = sorted(term for term in new_terms if term not in index)
    pointers = sorted(index[term] for term in new_terms if term in index)
    records = []
    for record in effect["added_records"]:
        records.append(
            {
                "entity_type": record["entity_type"],
                "link_status": record["link_status"],
                "attributes": [
                    {"name": name, "kind": kind}
                    for name, kind in record["attributes"]
                ],
            }
        )
    target = canonical_json_value(
        {
            "execution_status": effect["execution_status"],
            "delta_bits": effect["delta_bits"],
            "added_evidence_records": records,
            "added_conflicts": effect["added_conflicts"],
            "newly_matched_goal_term_indices": pointers,
        }
    )
    audit = {
        "newly_matched_goal_terms": new_terms,
        "missing_goal_pointer_terms": missing,
        "full_effect": effect,
    }
    return target, canonical_json_value(audit)


def render_effect_tokens(
    target: Mapping[str, Any], normalized_action: Mapping[str, Any]
) -> list[str]:
    tokens = [
        f"delta_bit_{index}={int(value)}"
        for index, value in enumerate(target["delta_bits"])
    ]
    tokens.extend(
        (
            f"execution={target['execution_status']}",
            f"matched_count={min(len(target['newly_matched_goal_term_indices']), 3)}",
        )
    )
    source = str(normalized_action["tool_id"]).rsplit("::", 1)[-1]
    for record in target["added_evidence_records"]:
        tokens.extend(
            (
                f"entity={record['entity_type']}",
                f"link={record['link_status']}",
                f"source={source}",
            )
        )
        for attribute in record["attributes"]:
            tokens.append(
                f"attribute={record['entity_type']}::{attribute['name']}::{attribute['kind']}"
            )
    for conflict in target["added_conflicts"]:
        tokens.append(f"conflict={conflict['attribute_name']}")
    return sorted(set(tokens))


def _relation_atoms(target: Mapping[str, Any]) -> set[str]:
    output = set()
    for record in target["added_evidence_records"]:
        entity = str(record["entity_type"])
        output.add(f"entity::{entity}")
        for attribute in record["attributes"]:
            output.add(f"attribute::{entity}::{attribute['name']}::{attribute['kind']}")
    return output


def _confirmation_row(
    union_row: Mapping[str, Any], hard_row: Mapping[str, Any], following: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = union_row["model_input"]["current_semantic_state"]
    target, target_audit = structured_successor_delta(current, following)
    normalized_action = union_row["model_input"]["normalized_action"]
    rendered = render_effect_tokens(target, normalized_action)
    hard_rendered = sorted(token for token in rendered if not token.startswith("source="))
    row = canonical_json_value(
        {
            "transition_ref": union_row["transition_ref"],
            "task_id_split_only": union_row["task_id"],
            "suite_split_only": union_row["suite"],
            "difficulty_split_only": union_row["difficulty"],
            "confirmation_fold_split_only": union_row["confirmation_fold"],
            "model_input": union_row["model_input"],
            "model_target": {"structured_successor_delta": target},
            "audit_only": {
                "current_state_ref": union_row["current_state_ref"],
                "next_state_ref": union_row["next_state_ref"],
                "source_versions": union_row["source_versions"],
                "rendered_full_effect_tokens": rendered,
                "rendered_hard_effect_tokens": hard_rendered,
                "target_audit": target_audit,
                "v20_target_matches": rendered == union_row["model_target"]["effect_tokens"],
                "v21_target_matches": hard_rendered == hard_row["model_target"]["effect_tokens"],
            },
        }
    )
    return row, {"full": rendered, "hard": hard_rendered}


def _support_row(outcome: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    following = outcome["model_visible"]["next_semantic_state"]
    normalized_action = normalized_action_descriptor(
        outcome["model_visible"]["candidate_action"], current
    )
    target, target_audit = structured_successor_delta(current, following)
    return canonical_json_value(
        {
            "support_ref": outcome["manifest_row_ref"],
            "task_id_split_only": outcome["metadata"]["task_id"],
            "suite_split_only": outcome["metadata"]["suite"],
            "model_input": {
                "current_semantic_state": current,
                "normalized_action": normalized_action,
            },
            "model_target": {"structured_successor_delta": target},
            "audit_only": {
                "current_state_ref": outcome["model_visible"]["current_state_ref"],
                "next_state_ref": outcome["model_visible"]["next_state_ref"],
                "rendered_full_effect_tokens": render_effect_tokens(target, normalized_action),
                "target_audit": target_audit,
            },
        }
    )


def _unseen_effect_occurrences(
    hard_rows: Sequence[Mapping[str, Any]], fold: int
) -> list[tuple[Mapping[str, Any], str]]:
    observed = {
        token
        for row in hard_rows
        if int(row["confirmation_fold"]) != fold
        for token in row["model_target"]["effect_tokens"]
    }
    return [
        (row, token)
        for row in hard_rows
        if int(row["confirmation_fold"]) == fold
        for token in row["model_target"]["effect_tokens"]
        if token not in observed
    ]


def build_successor_evidence_dataset(
    v17: Mapping[str, Any],
    v18: Mapping[str, Any],
    v19: Mapping[str, Any],
    union: Mapping[str, Any],
    hard: Mapping[str, Any],
    support_execution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = reconstruct_state_catalog(v17, v18, v19)
    hard_by_ref = {row["transition_ref"]: row for row in hard["transitions"]}
    confirmation_rows = []
    missing_next_refs = []
    for union_row in union["transitions"]:
        next_ref = str(union_row["next_state_ref"])
        if next_ref not in catalog:
            missing_next_refs.append(next_ref)
            continue
        row, _ = _confirmation_row(union_row, hard_by_ref[union_row["transition_ref"]], catalog[next_ref])
        confirmation_rows.append(row)
    confirmation_rows.sort(key=lambda row: row["transition_ref"])

    support_rows = []
    for outcome in support_execution["counterfactual_outcomes"]:
        current_ref = str(outcome["model_visible"]["current_state_ref"])
        current = support_execution["current_state_catalog"][current_ref]
        support_rows.append(_support_row(outcome, current))
    support_rows.sort(key=lambda row: row["support_ref"])

    support_tasks = {row["task_id_split_only"] for row in support_rows}
    confirmation_tasks = {row["task_id_split_only"] for row in confirmation_rows}
    support_relations = {
        relation
        for row in support_rows
        for relation in _relation_atoms(row["model_target"]["structured_successor_delta"])
    }
    support_tools = {
        row["model_input"]["normalized_action"]["tool_id"] for row in support_rows
    }
    fold_diagnostics = []
    relation_totals = Counter()
    relation_hits = Counter()
    operation_total = operation_hits = 0
    hard_rows = hard["transitions"]
    confirmation_by_ref = {row["transition_ref"]: row for row in confirmation_rows}
    for fold in range(3):
        train_relations = set(support_relations)
        for row in confirmation_rows:
            if int(row["confirmation_fold_split_only"]) != fold:
                train_relations |= _relation_atoms(row["model_target"]["structured_successor_delta"])
        fold_counts = Counter()
        fold_hits = Counter()
        occurrences = _unseen_effect_occurrences(hard_rows, fold)
        for hard_row, token in occurrences:
            slots = parse_effect_token(token)
            relation = None
            kind = None
            if slots["category"] == "entity":
                relation = f"entity::{slots['entity']}"
                kind = "entity"
            elif slots["category"] == "attribute":
                relation = f"attribute::{slots['entity']}::{slots['field']}::{slots['kind']}"
                kind = "attribute"
            if relation is not None:
                fold_counts[kind] += 1
                relation_totals[kind] += 1
                if relation in train_relations:
                    fold_hits[kind] += 1
                    relation_hits[kind] += 1
                operation_total += 1
                if hard_row["model_input"]["normalized_action"]["tool_id"] in support_tools:
                    operation_hits += 1
        train_count_values = {
            min(len(row["model_target"]["structured_successor_delta"]["newly_matched_goal_term_indices"]), 3)
            for row in confirmation_rows
            if int(row["confirmation_fold_split_only"]) != fold
        } | {
            min(len(row["model_target"]["structured_successor_delta"]["newly_matched_goal_term_indices"]), 3)
            for row in support_rows
        }
        fold_diagnostics.append(
            {
                "fold": fold,
                "unseen_positive_occurrences": len(occurrences),
                "relation_occurrences": dict(fold_counts),
                "relation_coverage": {
                    key: fold_hits[key] / max(1, fold_counts[key])
                    for key in ("entity", "attribute")
                },
                "available_matched_count_values": sorted(train_count_values),
            }
        )

    all_rows = confirmation_rows + support_rows
    input_leakage = {}
    semantic_leakage = {}
    pointer_errors = {}
    binding_errors = {}
    adjacency_errors = []
    for row in all_rows:
        ref = row.get("transition_ref", row.get("support_ref"))
        bad_keys = sorted(FORBIDDEN_MODEL_INPUT_KEYS & set(row["model_input"]))
        if bad_keys:
            input_leakage[str(ref)] = bad_keys
        findings = find_semantic_state_v3_leakage(row["model_input"]["current_semantic_state"])
        if findings:
            semantic_leakage[str(ref)] = list(findings)
        target = row["model_target"]["structured_successor_delta"]
        if row["audit_only"]["target_audit"]["missing_goal_pointer_terms"]:
            pointer_errors[str(ref)] = row["audit_only"]["target_audit"]["missing_goal_pointer_terms"]
        for record in target["added_evidence_records"]:
            if not record["entity_type"] or not record["link_status"]:
                binding_errors.setdefault(str(ref), []).append(record)
        current_step = int(row["model_input"]["current_semantic_state"]["step_index"])
        next_state = (
            catalog.get(row["audit_only"]["next_state_ref"])
            if "transition_ref" in row
            else support_execution["next_state_catalog"].get(row["audit_only"]["next_state_ref"])
        )
        if next_state is None or int(next_state["step_index"]) != current_step + 1:
            adjacency_errors.append(str(ref))

    support_count_values = Counter(
        min(len(row["model_target"]["structured_successor_delta"]["newly_matched_goal_term_indices"]), 3)
        for row in support_rows
    )
    audit = {
        "confirmation_rows": len(confirmation_rows),
        "support_rows": len(support_rows),
        "tasks": len(confirmation_tasks),
        "support_tasks": len(support_tasks),
        "reconstructed_state_catalog": len(catalog),
        "missing_next_state_refs": sorted(set(missing_next_refs)),
        "full_render_matches": sum(row["audit_only"]["v20_target_matches"] for row in confirmation_rows),
        "hard_render_matches": sum(row["audit_only"]["v21_target_matches"] for row in confirmation_rows),
        "support_rendered_rows": len(support_rows),
        "support_confirmation_task_overlap": sorted(support_tasks & confirmation_tasks),
        "semantic_input_leakage": semantic_leakage,
        "model_input_key_leakage": input_leakage,
        "goal_pointer_errors": pointer_errors,
        "record_binding_errors": binding_errors,
        "adjacency_errors": adjacency_errors,
        "relation_coverage": {
            key: relation_hits[key] / max(1, relation_totals[key])
            for key in ("entity", "attribute")
        },
        "relation_coverage_denominators": dict(relation_totals),
        "operation_coverage": operation_hits / max(1, operation_total),
        "operation_coverage_denominator": operation_total,
        "support_matched_count_values": dict(sorted(support_count_values.items())),
        "fold_diagnostics": fold_diagnostics,
        "structured_targets_exclude_composite_effect_tokens": all(
            "effect_tokens" not in row["model_target"]["structured_successor_delta"]
            for row in all_rows
        ),
    }
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "scope": "clean-only typed successor-evidence delta identification",
        "loader_contract": {
            "task_suite_difficulty_and_folds_are_split_only": True,
            "model_input_contains_only_current_semantic_state_and_normalized_action": True,
            "model_target_contains_only_typed_successor_delta": True,
            "canonical_effect_tokens_are_audit_only": True,
            "matched_count_is_deterministically_rendered_from_goal_term_pointers": True,
            "source_tool_is_deterministically_rendered_from_explicit_action": True,
            "no_utility_security_attack_or_final_outcome_labels": True,
        },
        "split_manifest": hard["split_manifest"],
        "confirmation_rows": confirmation_rows,
        "support_rows": support_rows,
    }
    return canonical_json_value(dataset), canonical_json_value(audit)
