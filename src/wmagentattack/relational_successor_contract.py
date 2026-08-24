"""Privacy-safe record-to-goal relation targets and static candidates for v29."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .bound_successor_world_model import record_signature
from .decision_state import canonical_json_value
from .semantic_state_v3 import find_semantic_state_v3_leakage
from .successor_evidence_contract import reconstruct_state_catalog, render_effect_tokens


SCHEMA_VERSION = "wmagentattack.relational_successor_contract.v29"
_FIELD = re.compile(r"[^a-z0-9]+")
_FORBIDDEN_TARGET_KEYS = {"value", "matched_goal_terms", "fact_terms", "effect_tokens"}


def _field_name(value: str) -> str:
    return _FIELD.sub("_", str(value).lower()).strip("_") or "value"


def _attributes(record: Mapping[str, Any]) -> list[dict[str, str]]:
    return sorted(
        [
            {"name": str(row["name"]), "kind": str(row.get("kind", "UNKNOWN"))}
            for row in record.get("attributes", ())
        ],
        key=lambda row: (row["name"], row["kind"]),
    )


def relational_successor_delta(
    current: Mapping[str, Any], following: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_records = list(current.get("evidence_records", ()))
    next_records = list(following.get("evidence_records", ()))
    if len(next_records) < len(current_records):
        raise ValueError("successor evidence ledger shrank")
    current_identity = [
        canonical_json_value({k: v for k, v in row.items() if k != "matched_goal_terms"})
        for row in current_records
    ]
    next_prefix = [
        canonical_json_value({k: v for k, v in row.items() if k != "matched_goal_terms"})
        for row in next_records[: len(current_records)]
    ]
    if current_identity != next_prefix:
        raise ValueError("successor evidence ledger is not prefix-causal")

    terms = [str(value) for value in current.get("goal", {}).get("fact_terms", ())]
    term_index = {value: index for index, value in enumerate(terms)}
    current_matched = set(current.get("goal_evidence", {}).get("matched_fact_terms", ()))
    following_matched = set(following.get("goal_evidence", {}).get("matched_fact_terms", ()))
    new_terms = following_matched - current_matched
    missing = sorted(value for value in new_terms if value not in term_index)
    if missing:
        raise ValueError(f"new goal terms absent from current goal frame: {missing}")

    records = []
    covered_terms: set[str] = set()
    raw_audit = []
    for raw in next_records[len(current_records) :]:
        matched = set(str(value) for value in raw.get("matched_goal_terms", ())) & new_terms
        covered_terms |= matched
        record = {
            "entity_type": str(raw["entity_type"]),
            "link_status": str(raw["link_status"]),
            "attributes": _attributes(raw),
            "newly_matched_goal_term_indices": sorted(term_index[value] for value in matched),
        }
        records.append(record)
        raw_audit.append({
            "source_tool": str(raw.get("source_tool", "<UNKNOWN>")),
            "matched_goal_terms": sorted(matched),
        })
    records.sort(key=lambda row: (
        row["entity_type"], row["link_status"],
        json.dumps(row["attributes"], sort_keys=True),
        row["newly_matched_goal_term_indices"],
    ))
    uncovered = sorted(new_terms - covered_terms)
    target = canonical_json_value({
        "execution_status": str(following.get("execution", {}).get("last_status", "unknown")),
        "added_evidence_records": records,
        "newly_matched_goal_term_indices": sorted(term_index[value] for value in new_terms),
    })
    return target, canonical_json_value({
        "raw_added_record_relations": raw_audit,
        "newly_matched_goal_terms": sorted(new_terms),
        "uncovered_new_goal_terms": uncovered,
    })


def _adapter_map(base: Mapping[str, Any], extension: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(base["adapters"])
    duplicate = set(output) & set(extension["additional_adapters"])
    if duplicate:
        raise ValueError(f"duplicate adapter definitions: {sorted(duplicate)}")
    output.update(extension["additional_adapters"])
    return output


def _template_attributes(spec: Mapping[str, Any], schemas: Mapping[str, Any]) -> list[dict[str, str]]:
    mode = str(spec["mode"])
    if mode == "VALUE":
        return [{
            "name": _field_name(str(spec["attribute_name"])),
            "kind": str(spec.get("attribute_kind", "SINGLE_VALUED")),
        }]
    if mode == "SCALAR_LIST":
        return [{"name": "observed", "kind": "SINGLE_VALUED"}]
    if mode in {"NAME_LIST_TEXT", "ENTITY_MAP"}:
        return [{"name": str(spec["attribute_name"]), "kind": str(spec["attribute_kind"])}]
    if mode == "PRICE_RANGE_MAP":
        return [{"name": "price_range", "kind": "RANGE"}]
    if mode == "RATING_REVIEWS_MAP":
        return [
            {"name": "rating", "kind": "SINGLE_VALUED"},
            {"name": "reviews", "kind": "SET_VALUED"},
        ]
    if mode == "FLIGHT_LINES":
        return [
            {"name": "departure_time", "kind": "TIME_SCOPED"},
            {"name": "arrival_time", "kind": "TIME_SCOPED"},
            {"name": "price", "kind": "SINGLE_VALUED"},
            {"name": "contact_information", "kind": "SINGLE_VALUED"},
        ]
    if mode == "MUTATION_ACK":
        return [
            {"name": "execution_acknowledged", "kind": "SINGLE_VALUED"},
            {"name": "observation", "kind": "SINGLE_VALUED"},
        ]
    if mode in {"OBJECT", "OBJECT_LIST", "USER_FIELDS"}:
        entity = str(spec["entity_type"])
        if entity not in schemas["object_output_fields"]:
            raise ValueError(f"missing static output schema for {entity}")
        return list(schemas["object_output_fields"][entity])
    raise ValueError(f"unsupported adapter mode: {mode}")


def static_record_candidates(
    base: Mapping[str, Any], extension: Mapping[str, Any], schemas: Mapping[str, Any]
) -> tuple[list[str], dict[str, list[str]]]:
    by_tool: dict[str, list[str]] = {}
    all_candidates: set[str] = set()
    error = record_signature({
        "entity_type": "execution_error", "link_status": "UNLINKED",
        "attributes": [{"name": "error_type", "kind": "SINGLE_VALUED"}],
    })
    for tool, spec in sorted(_adapter_map(base, extension).items()):
        attributes = _template_attributes(spec, schemas)
        candidates = {error}
        for link in ("UNIQUE", "AMBIGUOUS", "UNLINKED"):
            candidates.add(record_signature({
                "entity_type": spec["entity_type"], "link_status": link,
                "attributes": attributes,
            }))
        by_tool[tool] = sorted(candidates)
        all_candidates |= candidates
    return sorted(all_candidates), by_tool


def _target_has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(_FORBIDDEN_TARGET_KEYS & set(value)) or any(
            _target_has_forbidden_key(child) for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_target_has_forbidden_key(child) for child in value)
    return False


def _row(
    source: Mapping[str, Any], current: Mapping[str, Any], following: Mapping[str, Any], ref_key: str
) -> dict[str, Any]:
    target, audit = relational_successor_delta(current, following)
    return canonical_json_value({
        ref_key: source[ref_key],
        "task_id_split_only": source["task_id_split_only"],
        "suite_split_only": source["suite_split_only"],
        "model_input": source["model_input"],
        "model_target": {"relational_successor_delta": target},
        "audit_only": {**source["audit_only"], "relational_audit": audit},
    })


def _coverage(rows: Sequence[Mapping[str, Any]], refs: Sequence[str], candidates: set[str]) -> dict[str, Any]:
    wanted = set(refs)
    occurrences = []
    for row in rows:
        if row["transition_ref"] not in wanted:
            continue
        occurrences.extend(
            record_signature(record)
            for record in row["model_target"]["relational_successor_delta"]["added_evidence_records"]
        )
    unique = set(occurrences)
    return {
        "unique_total": len(unique), "occurrence_total": len(occurrences),
        "unique_coverage": 1.0 if not unique else len(unique & candidates) / len(unique),
        "occurrence_coverage": 1.0 if not occurrences else sum(value in candidates for value in occurrences) / len(occurrences),
        "missing_unique_signatures": sorted(unique - candidates),
    }


def build_relational_dataset(
    v17: Mapping[str, Any], v18: Mapping[str, Any], v19: Mapping[str, Any],
    hard: Mapping[str, Any], structured: Mapping[str, Any],
    support_execution: Mapping[str, Any],
    base_adapters: Mapping[str, Any], extension_adapters: Mapping[str, Any],
    output_schemas: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = reconstruct_state_catalog(v17, v18, v19)
    confirmation = []
    relation_errors = {}
    for source in structured["confirmation_rows"]:
        ref = source["transition_ref"]
        current = source["model_input"]["current_semantic_state"]
        following = catalog[source["audit_only"]["next_state_ref"]]
        row = _row(source, current, following, "transition_ref")
        target = row["model_target"]["relational_successor_delta"]
        union = sorted({
            index for record in target["added_evidence_records"]
            for index in record["newly_matched_goal_term_indices"]
        })
        if union != target["newly_matched_goal_term_indices"]:
            relation_errors[ref] = {"record_union": union, "global": target["newly_matched_goal_term_indices"]}
        confirmation.append(row)
    confirmation.sort(key=lambda row: row["transition_ref"])

    support = []
    source_support = {row["support_ref"]: row for row in structured["support_rows"]}
    for outcome in support_execution["counterfactual_outcomes"]:
        source = source_support[outcome["manifest_row_ref"]]
        current = support_execution["current_state_catalog"][outcome["model_visible"]["current_state_ref"]]
        following = outcome["model_visible"]["next_semantic_state"]
        target, relation_audit = relational_successor_delta(current, following)
        union = sorted({
            index for record in target["added_evidence_records"]
            for index in record["newly_matched_goal_term_indices"]
        })
        if union != target["newly_matched_goal_term_indices"]:
            relation_errors[source["support_ref"]] = {
                "record_union": union, "global": target["newly_matched_goal_term_indices"]
            }
        support.append(canonical_json_value({
            "support_ref": source["support_ref"],
            "task_id_split_only": source["task_id_split_only"],
            "suite_split_only": source["suite_split_only"],
            "model_input": source["model_input"],
            "model_target": {"relational_successor_delta": target},
            "audit_only": {**source["audit_only"], "relational_audit": relation_audit},
        }))
    support.sort(key=lambda row: row["support_ref"])

    candidates, by_tool = static_record_candidates(base_adapters, extension_adapters, output_schemas)
    candidate_set = set(candidates)
    split_coverage = {}
    for suite_name, suite in hard["split_manifest"].items():
        split_coverage[suite_name] = {
            name: _coverage(confirmation, split["test_refs"], candidate_set)
            for name, split in sorted(suite.items())
        }
    all_rows = confirmation + support
    leakage = {
        str(row.get("transition_ref", row.get("support_ref"))): list(find_semantic_state_v3_leakage(row["model_input"]["current_semantic_state"]))
        for row in all_rows
        if find_semantic_state_v3_leakage(row["model_input"]["current_semantic_state"])
    }
    target_leakage = [
        str(row.get("transition_ref", row.get("support_ref"))) for row in all_rows
        if _target_has_forbidden_key(row["model_target"])
    ]
    links = [
        len(record["newly_matched_goal_term_indices"])
        for row in confirmation
        for record in row["model_target"]["relational_successor_delta"]["added_evidence_records"]
    ]
    audit = canonical_json_value({
        "confirmation_rows": len(confirmation), "support_rows": len(support),
        "confirmation_tasks": len({row["task_id_split_only"] for row in confirmation}),
        "support_tasks": len({row["task_id_split_only"] for row in support}),
        "support_confirmation_task_overlap": sorted(
            {row["task_id_split_only"] for row in confirmation}
            & {row["task_id_split_only"] for row in support}
        ),
        "record_goal_relation_errors": relation_errors,
        "records_with_goal_links": sum(value > 0 for value in links),
        "maximum_goal_links_per_record": max(links, default=0),
        "static_candidate_count": len(candidates),
        "static_candidate_tools": len(by_tool),
        "webpage_candidate_present": any(
            json.loads(value)["entity_type"] == "webpage" for value in candidates
        ),
        "split_candidate_coverage": split_coverage,
        "semantic_input_leakage": leakage,
        "model_target_leakage": target_leakage,
        "static_registry_outcome_labels_present": bool(
            base_adapters.get("outcome_labels_present", False)
            or extension_adapters.get("outcome_labels_present", False)
            or output_schemas.get("outcome_labels_present", False)
        ),
    })
    dataset = canonical_json_value({
        "schema_version": SCHEMA_VERSION,
        "scope": "clean-only relational successor evidence data sufficiency",
        "loader_contract": {
            "record_goal_edges_are_bound_inside_each_record": True,
            "raw_goal_terms_and_values_are_audit_only": True,
            "candidate_inventory_uses_only_static_tool_contracts": True,
            "no_utility_security_attack_or_final_outcome_labels": True,
        },
        "split_manifest": hard["split_manifest"],
        "static_record_candidates": candidates,
        "static_candidates_by_tool": by_tool,
        "confirmation_rows": confirmation,
        "support_rows": support,
    })
    return dataset, audit
