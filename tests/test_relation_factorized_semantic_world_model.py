import numpy as np
import torch

from wmagentattack.relation_factorized_semantic_world_model import (
    action_relation_descriptions,
    aggregate_channels,
    effect_relation_descriptions,
    relation_kernel,
    select_support_set_rule,
    similarity_distribution_loss,
    support_fused_probabilities,
)


def test_descriptions_are_relation_only():
    effect, effect_mask = effect_relation_descriptions(
        "attribute=hotel::price_range::RANGE"
    )
    action, action_mask = action_relation_descriptions({
        "tool_id": "travel::reserve_hotel",
        "arguments": [{
            "field": "hotel_name",
            "value_class": {"type": "string", "category": "text"},
        }],
    })
    text = " ".join(effect + action).lower()
    assert effect_mask.tolist() == [True, True, True, True, True]
    assert action_mask.tolist() == [True, True, True]
    assert "hotel" in text and "price range" in text
    for forbidden in ("task_id", "source_versions", "utility", "security"):
        assert forbidden not in text


def test_aggregation_and_kernel_are_normalized_and_symmetric():
    rng = np.random.default_rng(7)
    channels = rng.normal(size=(4, 5, 8))
    channels /= np.linalg.norm(channels, axis=-1, keepdims=True)
    mask = np.asarray([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1],
    ], dtype=bool)
    weights = [0.35, 0.1, 0.25, 0.2, 0.1]
    aggregate = aggregate_channels(channels, mask, weights)
    kernel = relation_kernel(channels, mask, weights, 0.15)
    assert np.allclose(np.linalg.norm(aggregate, axis=1), 1.0, atol=1e-6)
    assert np.allclose(kernel, kernel.T, atol=1e-7)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.isfinite(kernel).all()


def test_support_diffusion_consumes_only_fitted_sources():
    logits = np.asarray([[3.0, -3.0, 8.0]])
    kernel = np.asarray([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.8],
        [0.1, 0.8, 1.0],
    ])
    source = np.asarray([True, False, False])
    candidate = np.asarray([False, True, False])
    fused = support_fused_probabilities(
        logits, source, candidate, kernel, 0.75, top_k=2
    )
    raw = 1.0 / (1.0 + np.exp(-logits))
    assert fused[0, 1] > raw[0, 1]
    assert fused[0, 0] == raw[0, 0]
    assert fused[0, 2] == raw[0, 2]


def test_distribution_loss_is_finite_and_differentiable():
    logits = torch.tensor([[2.0, -1.0, 0.2], [-0.5, 1.5, 0.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    fitted = torch.tensor([True, True, False])
    relation = torch.tensor([
        [1.0, 0.1, 0.8],
        [0.1, 1.0, 0.7],
        [0.8, 0.7, 1.0],
    ])
    loss = similarity_distribution_loss(logits, targets, fitted, relation, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_support_rule_respects_frozen_open_set_constraints():
    logits = np.asarray([
        [4.0, 2.0, -3.0],
        [4.0, -3.0, 2.0],
        [-4.0, -3.0, -3.0],
        [-4.0, -3.0, -3.0],
    ])
    targets = np.asarray([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    fitted = np.asarray([True, False, False])
    heldout = np.asarray([False, True, True])
    kernel = np.asarray([
        [1.0, 0.8, 0.8],
        [0.8, 1.0, 0.2],
        [0.8, 0.2, 1.0],
    ])
    rule = select_support_set_rule(
        logits,
        targets,
        fitted,
        heldout,
        kernel,
        support_weights=[0.0, 0.25],
        thresholds=[0.5, 0.7],
        top_k=2,
        maximum_false_positive_rate=0.05,
        maximum_set_size_multiplier=2.0,
        set_size_offset=0.5,
    )
    assert rule["false_positive_rate"] <= 0.05
    assert rule["mean_predicted_set_size"] <= rule["selection_set_limit"]
    assert rule["positive_nll"] <= rule["raw_positive_nll"] + 0.1
