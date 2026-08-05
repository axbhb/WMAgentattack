from copy import deepcopy

import pytest

from wmagentattack.candidate_constraint_dataset import (
    PROGRESS_OUTCOMES,
    audit_candidate_constraint_pilot,
    build_candidate_constraint_pilot,
    select_balanced_development_tasks,
)
from wmagentattack.semantic_state_v3 import semantic_state_v3_payload


def _record(*, call_index: int, source_tool: str, attributes):
    return {
        "record_id": f"runtime-{call_index}",
        "entity_type": "hotel",
        "entity_key": {"name": "Hotel A"},
        "entity_candidates": [{"name": "Hotel A"}],
        "link_status": "UNIQUE",
        "attributes": attributes,
        "context": {},
        "source_tool": source_tool,
        "source_arguments": {"city": "London"},
        "call_index": call_index,
        "execution_status": "success",
        "state_provenance": "hidden-oracle-derived",
    }


def _features(step: int):
    records = []
    receipts = []
    last_action = {"function": "<START>", "arguments": {}}
    observation = ""
    receipt = {"status": "start", "error_type": None, "output_type": None}
    if step >= 1:
        records.append(
            _record(
                call_index=0,
                source_tool="lookup",
                attributes=[
                    {"name": "city", "value": "London", "kind": "SINGLE_VALUED"}
                ],
            )
        )
        receipts.append(
            {"call_index": 0, "tool_name": "lookup", "execution_status": "success"}
        )
        last_action = {"function": "lookup", "arguments": {"city": "London"}}
        observation = "Hotel A in London"
        receipt = {"status": "success", "error_type": None, "output_type": "str"}
    if step >= 2:
        records.append(
            _record(
                call_index=1,
                source_tool="get_price",
                attributes=[
                    {"name": "price", "value": 250, "kind": "SINGLE_VALUED"}
                ],
            )
        )
        receipts.append(
            {
                "call_index": 1,
                "tool_name": "get_price",
                "execution_status": "success",
            }
        )
        last_action = {"function": "get_price", "arguments": {"name": "Hotel A"}}
        observation = "Hotel A costs 250 USD"
    return {
        "trusted_goal": "Find the London hotel price.",
        "track": "deterministic_greedy",
        "prefix_index": step,
        "legal_tools": ["STOP", "travel::lookup", "travel::get_price"],
        "last_action": last_action,
        "last_observation": observation,
        "execution_receipt": receipt,
        "causal_state_summary": {"hidden": step},
        "ledger_v2": {
            "records": records,
            "conflicts": [],
            "execution_receipts": receipts,
        },
    }


def _raw_prefix(step: int, action: str):
    return {
        "prefix_index": step,
        "features": _features(step),
        "targets": {
            "next_action": action,
            "argument_keys": [],
            "stop": action == "STOP",
            "evidence_obligations": [],
        },
    }


def _datasets():
    actions = ["travel::lookup", "travel::get_price", "STOP"]
    raw_prefixes = [_raw_prefix(index, action) for index, action in enumerate(actions)]
    semantic_prefixes = [
        {
            "prefix_index": prefix["prefix_index"],
            "features": {
                "semantic_state_v3": semantic_state_v3_payload(prefix["features"])
            },
            "targets": deepcopy(prefix["targets"]),
        }
        for prefix in raw_prefixes
    ]
    metadata = {
        "episode_id": "greedy::task-a",
        "task_id": "task-a",
        "suite": "travel",
        "split": "training",
        "track": "deterministic_greedy",
        "run_seed": 293,
        "task_difficulty": "L1",
        "task_archetype": "dependent_lookup",
    }
    catalog = {
        "STOP": {"name": "STOP", "description": "stop"},
        "travel::lookup": {"name": "lookup", "description": "lookup hotel"},
        "travel::get_price": {"name": "get_price", "description": "get price"},
    }
    raw = {
        "episodes": [{**metadata, "prefixes": raw_prefixes}],
        "tool_catalog": catalog,
        "argument_key_vocab": [],
    }
    semantic = {
        "episodes": [{**metadata, "prefixes": semantic_prefixes}],
        "tool_catalog": catalog,
        "argument_key_vocab": [],
    }
    return raw, semantic


def test_balanced_selection_is_metadata_only_and_lexicographic():
    raw, _ = _datasets()
    later = deepcopy(raw["episodes"][0])
    later["task_id"] = "task-z"
    later["episode_id"] = "greedy::task-z"
    sampled = deepcopy(raw["episodes"][0])
    sampled["task_id"] = "task-0"
    sampled["episode_id"] = "sampled::task-0"
    sampled["track"] = "stochastic_policy"
    calibration = deepcopy(raw["episodes"][0])
    calibration["task_id"] = "task-00"
    calibration["episode_id"] = "calibration::task-00"
    calibration["split"] = "calibration"
    raw["episodes"].extend((later, sampled, calibration))
    selected = select_balanced_development_tasks(
        raw,
        split="training",
        track="deterministic_greedy",
        suites=("travel",),
        difficulties=("L1",),
    )
    assert selected == ("task-a",)


def test_builder_creates_observed_labels_and_separate_unlabeled_queries():
    raw, semantic = _datasets()
    dataset = build_candidate_constraint_pilot(
        raw,
        semantic,
        split="training",
        track="deterministic_greedy",
        suites=("travel",),
        difficulties=("L1",),
    )
    assert len(dataset["state_catalog"]) == 2
    assert len(dataset["constraint_catalog"]) == 3
    assert len(dataset["observed_rows"]) == 6
    assert len(dataset["unlabeled_counterfactual_queries"]) == 12
    assert {row["target"]["progress"] for row in dataset["observed_rows"]} == set(
        PROGRESS_OUTCOMES
    )
    assert all(
        row["target"]["training_role"] == "STATE_CONSISTENCY_ONLY"
        for row in dataset["observed_rows"]
        if row["target"]["progress"] == "ALREADY_SUPPORTED"
    )
    assert all(
        row["target"]["training_role"] == "PREDICTIVE"
        for row in dataset["observed_rows"]
        if row["target"]["progress"] != "ALREADY_SUPPORTED"
    )
    assert all(
        row["candidate_action"]["argument_binding"] == "OBSERVED"
        for row in dataset["observed_rows"]
    )
    assert all(
        "target" not in row
        and row["label_status"] == "UNLABELED_COUNTERFACTUAL"
        and row["candidate_action"]["arguments"] is None
        for row in dataset["unlabeled_counterfactual_queries"]
    )


def test_audit_passes_schema_but_blocks_training_without_counterfactual_events():
    raw, semantic = _datasets()
    dataset = build_candidate_constraint_pilot(
        raw,
        semantic,
        split="training",
        track="deterministic_greedy",
        suites=("travel",),
        difficulties=("L1",),
    )
    audit = audit_candidate_constraint_pilot(
        dataset,
        expected={
            "tasks": 1,
            "episodes": 1,
            "states": 2,
            "constraints": 3,
            "transitions": 2,
            "observed_rows": 6,
            "counterfactual_queries": 12,
            "total_query_space": 18,
        },
        schema_gate={
            "minimum_rows_per_progress": 1,
            "minimum_tasks_per_progress": 1,
        },
        readiness_gate={
            "minimum_observed_candidate_fraction": 0.25,
            "minimum_execution_errors": 1,
            "minimum_conflicts": 1,
            "minimum_ambiguity_events": 1,
        },
    )
    assert audit["schema_pass"]
    assert not audit["training_ready"]
    assert audit["decision"] == (
        "GO_SCHEMA__NO_GO_TRAINING__COUNTERFACTUAL_COLLECTION_REQUIRED"
    )
    assert audit["state_leakage"] == {}


def test_future_or_proof_fields_in_semantic_state_fail_closed():
    raw, semantic = _datasets()
    semantic["episodes"][0]["prefixes"][0]["features"]["semantic_state_v3"][
        "proof_contract"
    ] = {"required_calls": ["lookup"]}
    with pytest.raises(Exception):
        build_candidate_constraint_pilot(
            raw,
            semantic,
            split="training",
            track="deterministic_greedy",
            suites=("travel",),
            difficulties=("L1",),
        )
