from copy import deepcopy

import numpy as np
import pytest
import torch

from wmagentattack.hybrid_semantic_world_model import (
    EVIDENCE_DELTA_TARGETS,
    ExactObservedSemanticTransition,
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
    evidence_delta_target,
    semantic_state_v3_feature_size,
    semantic_state_v3_feature_vector,
    tool_candidate_vector,
)
from wmagentattack.semantic_state_v3 import semantic_state_v3_payload


def _features(step: int):
    features = {
        "trusted_goal": "Find the London hotel and report its price.",
        "track": "deterministic_greedy",
        "prefix_index": step,
        "legal_tools": ["STOP", "travel::lookup"],
        "last_action": {"function": "<START>", "arguments": {}},
        "last_observation": "",
        "execution_receipt": {
            "status": "start",
            "error_type": None,
            "output_type": None,
        },
        "causal_state_summary": {"hidden": "must-not-enter"},
        "ledger_v2": {
            "records": [],
            "conflicts": [],
            "execution_receipts": [],
        },
    }
    if step == 0:
        return features
    features["last_action"] = {
        "function": "lookup",
        "arguments": {"city": "London"},
    }
    features["last_observation"] = "Hotel A costs 250 USD"
    features["execution_receipt"] = {
        "status": "success",
        "error_type": None,
        "output_type": "str",
    }
    features["ledger_v2"] = {
        "records": [
            {
                "record_id": "runtime-id",
                "entity_type": "hotel",
                "entity_key": {"name": "Hotel A"},
                "entity_candidates": [{"name": "Hotel A"}],
                "link_status": "UNIQUE",
                "attributes": [
                    {
                        "name": "city",
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
                "source_tool": "lookup",
                "source_arguments": {"city": "London"},
                "call_index": 0,
                "execution_status": "success",
                "state_provenance": "hidden-oracle-derived",
            }
        ],
        "conflicts": [],
        "execution_receipts": [
            {
                "call_index": 0,
                "tool_name": "lookup",
                "execution_status": "success",
            }
        ],
    }
    return features


def _state(step: int):
    return semantic_state_v3_payload(_features(step))


def test_feature_vector_is_deterministic_and_has_frozen_size():
    state = _state(1)
    first = semantic_state_v3_feature_vector(state, hash_dimension=16)
    second = semantic_state_v3_feature_vector(deepcopy(state), hash_dimension=16)
    assert np.array_equal(first, second)
    assert first.shape == (semantic_state_v3_feature_size(16),)
    assert np.isfinite(first).all()


def test_feature_vector_revalidates_schema_and_rejects_leakage():
    state = _state(1)
    state["utility"] = True
    with pytest.raises(ValueError, match="leakage"):
        semantic_state_v3_feature_vector(state, hash_dimension=8)


def test_exact_transition_accepts_one_observed_append():
    following, audit = ExactObservedSemanticTransition().advance(
        _state(0), _state(1), executed_action_id="travel::lookup"
    )
    assert following.step_index == 1
    assert audit.records_added == 1
    assert audit.matched_goal_terms_added >= 1
    assert audit.conflicts_added == 0


@pytest.mark.parametrize("mutation", ["goal", "history", "action"])
def test_exact_transition_rejects_rewrites_and_action_mismatch(mutation):
    current = _state(0)
    following = _state(1)
    action = "travel::lookup"
    if mutation == "goal":
        following["goal"]["normalized_goal"] = "A different goal"
    elif mutation == "history":
        following["execution"]["history"].append(
            {
                "call_index": 0,
                "tool_name": "lookup",
                "execution_status": "success",
            }
        )
    else:
        action = "travel::different_tool"
    with pytest.raises(ValueError):
        ExactObservedSemanticTransition().advance(
            current, following, executed_action_id=action
        )


def test_evidence_delta_uses_adjacent_observed_states():
    target = evidence_delta_target(_state(0), _state(1))
    assert target.shape == (len(EVIDENCE_DELTA_TARGETS),)
    assert target.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_hybrid_heads_have_expected_candidate_conditional_shapes_and_gradients():
    states = torch.as_tensor(
        np.stack(
            [
                semantic_state_v3_feature_vector(_state(0), hash_dimension=8),
                semantic_state_v3_feature_vector(_state(1), hash_dimension=8),
            ]
        ),
        dtype=torch.float32,
    )
    candidates = torch.as_tensor(
        np.stack(
            [
                tool_candidate_vector("STOP", hash_dimension=8),
                tool_candidate_vector(
                    {"name": "travel::lookup", "description": "lookup"},
                    hash_dimension=8,
                ),
                tool_candidate_vector(
                    {"name": "travel::other", "description": "other"},
                    hash_dimension=8,
                ),
            ]
        ),
        dtype=torch.float32,
    )
    model = HybridSemanticWorldModel(
        state_size=states.shape[1],
        candidate_size=candidates.shape[1],
        argument_keys=4,
        hidden_size=12,
        dropout=0.0,
    )
    assert_no_planning_or_value_heads(model)
    action, arguments, evidence = model(states, candidates)
    assert action.shape == (2, 3)
    assert arguments.shape == (2, 4)
    assert evidence.shape == (2, 3, len(EVIDENCE_DELTA_TARGETS))
    (action.mean() + arguments.mean() + evidence.mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_legal_mask_assigns_zero_probability_to_illegal_candidates():
    states = torch.zeros(2, semantic_state_v3_feature_size(4))
    candidates = torch.zeros(3, 4)
    model = HybridSemanticWorldModel(
        state_size=states.shape[1],
        candidate_size=4,
        argument_keys=1,
        hidden_size=8,
        dropout=0.0,
    )
    legal = torch.tensor([[True, True, False], [False, True, False]])
    probabilities = model.action_probabilities(states, candidates, legal)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(probabilities[~legal], torch.zeros_like(probabilities[~legal]))


def test_tool_candidate_vector_is_stable():
    descriptor = {"name": "travel::lookup", "description": "lookup hotel"}
    assert np.array_equal(
        tool_candidate_vector(descriptor, hash_dimension=16),
        tool_candidate_vector(deepcopy(descriptor), hash_dimension=16),
    )
