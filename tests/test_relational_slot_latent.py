import copy
import numpy as np
import torch

from wmagentattack.relational_slot_latent import (
    GroundedPredictiveSlotResidual,
    RelationalSlotEncoder,
    build_interface_affordance_state,
    build_relational_slot_state,
)


def causal():
    return {
        "source": "agentdojo",
        "track": "agentdojo:travel",
        "trusted_goal": "Find the cheapest hotel in Paris on 2026-09-01 under $200.",
        "visible_observation": "Paris options for 2026-09-01 include 'Hotel A' at $180.",
        "visible_prior_tool": "hotel_read",
        "legal_tool_names": ["hotel_read", "hotel_generate"],
        "tool_schemas": [],
    }


def test_entity_renaming_is_invariant():
    left = build_relational_slot_state(causal())
    changed = copy.deepcopy(causal())
    changed["trusted_goal"] = changed["trusted_goal"].replace("Paris", "Berlin")
    changed["visible_observation"] = changed["visible_observation"].replace("Paris", "Berlin").replace("Hotel A", "Hotel Z")
    right = build_relational_slot_state(changed)
    np.testing.assert_array_equal(left.features, right.features)
    np.testing.assert_array_equal(left.relations, right.relations)


def test_semantic_constraint_changes_features():
    left = build_relational_slot_state(causal())
    changed = copy.deepcopy(causal())
    changed["trusted_goal"] = changed["trusted_goal"].replace("cheapest", "highest")
    right = build_relational_slot_state(changed)
    assert not np.array_equal(left.features, right.features)


def test_encoder_is_permutation_equivariant_at_pool():
    state = build_relational_slot_state(causal())
    n = state.audit["node_count"]
    permutation = np.arange(state.features.shape[0])
    permutation[:n] = permutation[:n][::-1]
    encoder = RelationalSlotEncoder(
        feature_size=state.features.shape[1], hidden_size=16, layers=2, dropout=0.0
    ).eval()
    def run(features, types, relations):
        with torch.no_grad():
            return encoder(
                torch.tensor(features[None]), torch.tensor(types[None]),
                torch.tensor(relations[None]),
                torch.tensor((np.arange(len(types)) < n)[None]),
            )
    original = run(state.features, state.node_types, state.relations)
    permuted = run(
        state.features[permutation], state.node_types[permutation],
        state.relations[np.ix_(permutation, permutation)],
    )
    torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)


def test_grounded_predictive_shapes():
    model = GroundedPredictiveSlotResidual(
        candidate_size=12, slot_feature_size=26, hidden_size=16,
        slot_layers=1, grounding_size=12, dropout=0.0,
    )
    hidden = torch.randn(4, 16)
    predicted = model.predict_slot_latent(hidden)
    assert predicted.shape == (4, 16)
    assert model.static_grounding(predicted).shape == (4, 12)
    assert model.transition_grounding(hidden, predicted).shape == (4, 12)


def affordance_causal():
    value = causal()
    value["tool_schemas"] = [
        {
            "type": "function",
            "function": {
                "name": "hotel_read",
                "description": "Search hotels and return hotel rating, address, and price.",
                "parameters": {"type": "object", "properties": {"city": {}, "budget": {}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hotel_generate",
                "description": "Book a selected hotel for the requested date.",
                "parameters": {"type": "object", "properties": {"hotel": {}, "date": {}}},
            },
        },
    ]
    return value


def test_affordance_state_ignores_unmatched_entity_values():
    left = build_interface_affordance_state(affordance_causal())
    changed = copy.deepcopy(affordance_causal())
    changed["trusted_goal"] = changed["trusted_goal"].replace("Paris", "Reykjavik")
    changed["visible_observation"] = changed["visible_observation"].replace("Paris", "Reykjavik").replace("Hotel A", "Aurora Place")
    right = build_interface_affordance_state(changed)
    np.testing.assert_array_equal(left.features, right.features)
    np.testing.assert_array_equal(left.relations, right.relations)
    assert left.audit["unmatched_text_tokens_encoded"] == 0


def test_affordance_state_preserves_interface_aligned_intent():
    left = build_interface_affordance_state(affordance_causal())
    changed = copy.deepcopy(affordance_causal())
    changed["trusted_goal"] = "Find and book a hotel on 2026-09-01."
    right = build_interface_affordance_state(changed)
    assert not np.array_equal(left.features, right.features)
    assert left.audit["encoded_interface_concepts"] > 0
    assert left.audit["interface_only_lexical_encoding"]
