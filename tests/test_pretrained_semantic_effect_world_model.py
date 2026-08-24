import json

import numpy as np
import torch

from wmagentattack.pretrained_semantic_effect_world_model import (
    PretrainedSemanticEffectTransition,
    apply_unseen_calibration,
    calibration_label_mask,
    effect_token_description,
    normalized_action_description,
    select_unseen_calibration,
    semantic_hard_negative_loss,
)


def test_descriptions_are_semantic_and_deterministic():
    label = effect_token_description("attribute=bank_account::balance::SINGLE_VALUED")
    action = normalized_action_description({
        "tool_id": "banking::get_balance",
        "arguments": [{"field": "account", "value_class": {"type": "string"}}],
    })
    assert label == "passage: effect type attribute; entity bank account; field balance; kind SINGLE VALUED."
    assert "banking get balance" in action and "argument account" in action


def test_label_calibration_partition_is_stable():
    vocabulary = [f"entity=item_{index}" for index in range(50)]
    first = calibration_label_mask(vocabulary, 5)
    assert np.array_equal(first, calibration_label_mask(vocabulary, 5))
    assert 3 <= int(first.sum()) <= 17


def test_model_and_hard_negative_loss_are_finite():
    model = PretrainedSemanticEffectTransition(16, 8, 12, 12)
    states = torch.randn(4, 16)
    actions = torch.randn(4, 8)
    action_semantics = torch.nn.functional.normalize(torch.randn(4, 12), dim=-1)
    labels = torch.nn.functional.normalize(torch.randn(7, 12), dim=-1)
    logits, execution = model(states, actions, action_semantics, labels)
    target = torch.zeros(4, 7)
    target[:, 0] = 1
    loss = semantic_hard_negative_loss(logits, target, torch.ones(7, dtype=torch.bool), labels, 2, 0.5)
    assert logits.shape == (4, 7) and execution.shape == (4,)
    assert torch.isfinite(loss)


def test_train_only_unseen_calibration_is_json_safe_and_changes_only_unseen():
    logits = np.asarray([[0.0, -2.0], [0.0, 1.0]])
    target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    heldout = np.asarray([False, True])
    selected = select_unseen_calibration(logits, target, heldout, [1.0, 2.0], [0.0, 1.0])
    json.dumps(selected)
    changed = apply_unseen_calibration(logits, heldout, selected["temperature"], selected["bias"])
    assert np.array_equal(changed[:, 0], logits[:, 0])
