from copy import deepcopy

import pytest

from wmagentattack.semantic_state_v3 import (
    SEMANTIC_STATE_V3_SCHEMA_VERSION,
    build_structured_semantic_state_v3,
    find_semantic_state_v3_leakage,
    semantic_state_v3_fingerprint,
    semantic_state_v3_payload,
)


def _features():
    return {
        "trusted_goal": (
            "Find all London hotels under $300, compare their ratings, and report "
            "the cheapest unique option."
        ),
        "track": "deterministic_greedy",
        "prefix_index": 2,
        "legal_tools": ["STOP", "travel::get_hotels", "travel::get_rating"],
        "last_action": {
            "function": "get_hotels",
            "arguments": {"location": "London"},
        },
        "last_observation": "Hotel A\nHotel B",
        "execution_receipt": {
            "status": "success",
            "error_type": None,
            "output_type": "str",
        },
        "causal_state_summary": {
            "last_state_changed": True,
            "delta_roots": {"hidden_bookings": 99},
        },
        "ledger_v2": {
            "records": [
                {
                    "record_id": "episode-specific-a",
                    "entity_type": "hotel",
                    "entity_key": {"name": "Hotel A"},
                    "entity_candidates": [{"name": "Hotel A"}],
                    "link_status": "UNIQUE",
                    "attributes": [
                        {
                            "fact_id": "fact-a",
                            "name": "location",
                            "value": "London",
                            "kind": "SINGLE_VALUED",
                        },
                        {
                            "name": "price",
                            "value": 250,
                            "kind": "SINGLE_VALUED",
                        },
                    ],
                    "context": {"currency": "USD"},
                    "source_tool": "get_hotels",
                    "source_arguments": {"location": "London"},
                    "call_index": 0,
                    "execution_status": "success",
                    "state_provenance": "mutating",
                },
                {
                    "record_id": "episode-specific-b",
                    "entity_type": "hotel",
                    "entity_key": {"name": "Hotel B"},
                    "entity_candidates": [{"name": "Hotel B"}],
                    "link_status": "UNIQUE",
                    "attributes": [
                        {
                            "name": "location",
                            "value": "London",
                            "kind": "SINGLE_VALUED",
                        },
                        {
                            "name": "price",
                            "value": 280,
                            "kind": "SINGLE_VALUED",
                        },
                    ],
                    "context": {"currency": "USD"},
                    "source_tool": "get_hotels",
                    "source_arguments": {"location": "London"},
                    "call_index": 0,
                    "execution_status": "success",
                    "state_provenance": "read_only",
                },
            ],
            "conflicts": [],
            "execution_receipts": [
                {
                    "episode_id": "secret-episode",
                    "call_index": 0,
                    "tool_name": "get_hotels",
                    "arguments_fingerprint": "secret-args",
                    "observation_fingerprint": "secret-output",
                    "execution_status": "success",
                },
                {
                    "call_index": 1,
                    "tool_name": "get_hotels",
                    "execution_status": "success",
                },
            ],
        },
    }


def test_state_is_deterministic_entity_preserving_and_goal_linked():
    first = build_structured_semantic_state_v3(_features())
    second = build_structured_semantic_state_v3(deepcopy(_features()))
    assert first == second
    assert first.schema_version == SEMANTIC_STATE_V3_SCHEMA_VERSION
    assert len(first.evidence_records) == 2
    assert [row.entity_key["name"] for row in first.evidence_records] == [
        "Hotel A",
        "Hotel B",
    ]
    assert {"london", "hotels"} & set(first.goal_evidence.matched_fact_terms)
    assert first.goal.requires_set_coverage
    assert first.goal.requires_uniqueness
    assert first.goal.has_comparison


def test_runtime_ids_and_hidden_simulator_summaries_are_invariant():
    baseline = _features()
    changed = deepcopy(baseline)
    changed["causal_state_summary"] = {
        "last_state_changed": False,
        "delta_roots": {"different_hidden_state": 1},
    }
    changed["ledger_v2"]["records"][0]["record_id"] = "different-id"
    changed["ledger_v2"]["records"][0]["state_provenance"] = "read_only"
    changed["ledger_v2"]["execution_receipts"][0]["episode_id"] = "other"
    assert semantic_state_v3_fingerprint(baseline) == semantic_state_v3_fingerprint(
        changed
    )
    emitted = semantic_state_v3_payload(changed)
    assert find_semantic_state_v3_leakage(emitted) == ()
    serialized = str(emitted)
    assert "different_hidden_state" not in serialized
    assert "different-id" not in serialized
    assert "state_provenance" not in serialized


@pytest.mark.parametrize(
    "path,value",
    [
        (("utility",), True),
        (("future_calls",), ["send_money"]),
        (("expert_calls",), ["get_hotels"]),
        (("ledger_v2", "final_output"), "answer"),
        (("ledger_v2", "records", 0, "task_success"), True),
    ],
)
def test_future_expert_and_outcome_fields_fail_closed(path, value):
    features = _features()
    cursor = features
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match="leakage|unknown"):
        build_structured_semantic_state_v3(features)


def test_future_ledger_event_is_rejected_instead_of_silently_truncated():
    features = _features()
    future = deepcopy(features["ledger_v2"]["records"][0])
    future["call_index"] = features["prefix_index"]
    features["ledger_v2"]["records"].append(future)
    with pytest.raises(ValueError, match="prefix causality"):
        build_structured_semantic_state_v3(features)


def test_error_and_retry_summary_uses_observed_receipts_only():
    features = _features()
    features["prefix_index"] = 3
    features["last_action"] = {
        "function": "get_hotels",
        "arguments": {"location": "London"},
    }
    features["execution_receipt"] = {
        "status": "error",
        "error_type": "ValidationError",
        "output_type": None,
    }
    features["last_observation"] = "Validation error: invalid location"
    features["ledger_v2"]["execution_receipts"].append(
        {
            "call_index": 2,
            "tool_name": "get_hotels",
            "execution_status": "error",
        }
    )
    state = build_structured_semantic_state_v3(features)
    assert state.execution.cumulative_errors == 1
    assert state.execution.consecutive_errors == 1
    assert state.execution.repeated_last_tool_count == 3
    assert state.execution.observation_has_error_lexeme
