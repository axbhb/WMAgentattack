import numpy as np
import torch

from wmagentattack.compositional_effect_world_model import (
    CompositionalEffectTransition,
    IndependentLabelEffectTransition,
    effect_token_feature_vector,
    effect_vocabulary_features,
    normalized_action_feature_vector,
    parse_effect_token,
)


def test_effect_token_parser_preserves_compositional_slots() -> None:
    assert parse_effect_token("attribute=cloud_file::content::SINGLE_VALUED") == {
        "category": "attribute",
        "entity": "cloud_file",
        "field": "content",
        "kind": "SINGLE_VALUED",
        "value": "",
    }
    assert parse_effect_token("delta_bit_2=1")["category"] == "delta_bit"


def test_effect_features_are_deterministic_and_share_structure() -> None:
    left = effect_token_feature_vector("attribute=cloud_file::content::SINGLE_VALUED")
    same = effect_token_feature_vector("attribute=cloud_file::content::SINGLE_VALUED")
    related = effect_token_feature_vector("attribute=document::content::SINGLE_VALUED")
    unrelated = effect_token_feature_vector("execution=error")
    assert np.array_equal(left, same)
    assert float(left @ related) > float(left @ unrelated)


def test_action_features_are_label_blind_and_deterministic() -> None:
    action = {
        "tool_id": "workspace::share_file",
        "arguments": [{"field": "permission", "value_class": {"type": "string"}}],
    }
    assert np.array_equal(
        normalized_action_feature_vector(action), normalized_action_feature_vector(action)
    )


def test_candidate_models_score_an_arbitrary_label_matrix() -> None:
    labels = torch.tensor(effect_vocabulary_features([
        "entity=cloud_file", "attribute=cloud_file::content::SINGLE_VALUED"
    ]))
    state = torch.zeros(3, 8)
    action = torch.zeros(3, 6)
    semantic = torch.zeros(3, 4)
    compositional = CompositionalEffectTransition(8, 6, 4, labels.shape[1], 12)
    independent = IndependentLabelEffectTransition(8, 6, 4, 2, 12)
    assert compositional(state, action, semantic, labels)[0].shape == (3, 2)
    assert independent(state, action, semantic, labels)[0].shape == (3, 2)

