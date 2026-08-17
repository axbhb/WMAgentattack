from __future__ import annotations

import numpy as np
import pytest
import torch

from wmagentattack.factorized_belief_world_model import (
    FactorizedBeliefWorldModel,
    TYPED_STATE_NODES,
    assert_factorized_scope,
    masked_action_probabilities,
    typed_state_node_features,
)


def _causal() -> dict:
    return {
        "source": "agentdojo",
        "track": "agentdojo:travel",
        "trusted_goal": "Find a London hotel under 160 on March 15.",
        "visible_observation": "Hotel A costs 140 and has rating 4.8.",
        "visible_prior_tool": "hotel_read",
        "legal_tool_names": ["hotel_read", "hotel_generate"],
        "tool_schemas": [
            {"function": {"name": "hotel_read", "description": "read hotels"}},
            {"function": {"name": "hotel_generate", "description": "book hotel"}},
        ],
    }


def test_typed_nodes_are_deterministic_and_field_sensitive() -> None:
    first = typed_state_node_features(_causal(), hash_dimension=32)
    second = typed_state_node_features(_causal(), hash_dimension=32)
    assert np.array_equal(first, second)
    assert first.shape == (len(TYPED_STATE_NODES), 40)
    changed = _causal()
    changed["visible_observation"] = "Hotel B failed with an execution error."
    third = typed_state_node_features(changed, hash_dimension=32)
    observation = TYPED_STATE_NODES.index("visible_observation")
    goal = TYPED_STATE_NODES.index("trusted_goal")
    assert not np.array_equal(first[observation], third[observation])
    assert np.array_equal(first[goal], third[goal])


def test_factorized_model_respects_legal_interface() -> None:
    model = FactorizedBeliefWorldModel(
        structured_state_size=24,
        node_feature_size=40,
        node_count=len(TYPED_STATE_NODES),
        candidate_size=16,
        hidden_size=32,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    )
    assert_factorized_scope(model)
    logits, outcomes, belief = model.one_step(
        torch.randn(3, 24),
        torch.randn(3, len(TYPED_STATE_NODES), 40),
        torch.randn(3, 16),
        torch.randn(5, 16),
    )
    assert logits.shape == (3, 5)
    assert outcomes.shape == (3, 3)
    assert belief.shape == (3, 32)
    legal = torch.tensor(
        [[True, False, True, False, False]] * 3, dtype=torch.bool
    )
    probabilities = masked_action_probabilities(logits, legal)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
    assert torch.equal(probabilities[:, ~legal[0]], torch.zeros(3, 3))


def test_empty_legal_interface_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one legal action"):
        masked_action_probabilities(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
