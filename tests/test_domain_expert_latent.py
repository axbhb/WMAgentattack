import copy

import numpy as np
import pytest
import torch

from wmagentattack.domain_expert_latent import (
    DOMAIN_NAMES,
    DenseCapacityAffordanceResidual,
    DomainExpertAffordanceResidual,
    domain_index_from_causal,
    routed_parameter_gap_fraction,
)


def causal(domain="travel"):
    return {
        "source": "agentdojo",
        "track": f"agentdojo:{domain}",
        "trusted_goal": "Find a hotel with the highest rating under $200.",
        "visible_observation": "A hotel result has a rating and price.",
        "visible_prior_tool": "hotel_read",
        "legal_tool_names": ["hotel_read", "hotel_generate"],
        "tool_schemas": [],
    }


def models():
    common = dict(
        candidate_size=128, slot_feature_size=40, hidden_size=96,
        slot_layers=2, dropout=0.0,
    )
    return (
        DenseCapacityAffordanceResidual(**common, dense_bottleneck_size=96),
        DomainExpertAffordanceResidual(**common, expert_bottleneck_size=24),
    )


def test_router_uses_only_visible_track_not_text_or_task_id():
    left = causal("travel")
    changed = copy.deepcopy(left)
    changed["trusted_goal"] = "Completely different arbitrary request."
    changed["task_name"] = "secret_task_id"
    assert domain_index_from_causal(left) == domain_index_from_causal(changed) == 2
    assert tuple(domain_index_from_causal(causal(name)) for name in DOMAIN_NAMES) == (0, 1, 2, 3)


def test_unknown_track_is_rejected():
    with pytest.raises(ValueError):
        domain_index_from_causal(causal("unknown"))


def test_dense_and_expert_capacities_are_within_two_percent():
    dense, expert = models()
    assert routed_parameter_gap_fraction(dense, expert) <= 0.02


def test_zero_gates_exactly_reproduce_context():
    dense, expert = models()
    batch = 8
    context = torch.randn(batch, 96)
    features = torch.randn(batch, 12, 40)
    types = torch.zeros(batch, 12, dtype=torch.long)
    relations = torch.zeros(batch, 12, 12, dtype=torch.long)
    mask = torch.ones(batch, 12, dtype=torch.bool)
    domains = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    with torch.no_grad():
        dense_hidden, _ = dense.initial_hidden(context, features, types, relations, mask, domains)
        expert_hidden, _ = expert.initial_hidden(context, features, types, relations, mask, domains)
    torch.testing.assert_close(dense_hidden, context)
    torch.testing.assert_close(expert_hidden, context)


def test_each_row_uses_exactly_one_expert():
    _, model = models()
    with torch.no_grad():
        model.expert_gates.fill_(0.7)
    calls = [0, 0, 0, 0]
    hooks = []
    for index, expert in enumerate(model.experts):
        hooks.append(expert.register_forward_hook(lambda _m, inputs, _o, i=index: calls.__setitem__(i, calls[i] + len(inputs[0]))))
    batch = 8
    model.initial_hidden(
        torch.randn(batch, 96), torch.randn(batch, 12, 40),
        torch.zeros(batch, 12, dtype=torch.long),
        torch.zeros(batch, 12, 12, dtype=torch.long),
        torch.ones(batch, 12, dtype=torch.bool),
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    for hook in hooks:
        hook.remove()
    assert calls == [2, 2, 2, 2]
