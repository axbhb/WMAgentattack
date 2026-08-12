from __future__ import annotations

import numpy as np
import torch

from wmagentattack.source_residual_adapter import (
    SourceResidualActionModel,
    source_indices,
)


def test_source_indices_are_frozen_and_deterministic() -> None:
    rows = [{"source": "agentdojo"}, {"source": "tool_sandbox"}]
    candidates = ["a", "b", "c"]
    catalog = {
        "a": {"source": "agentdojo"},
        "b": {"source": "injecagent"},
        "c": {"source": "tool_sandbox"},
    }
    row_values, candidate_values = source_indices(rows, candidates, catalog)
    assert np.array_equal(row_values, [0, 2])
    assert np.array_equal(candidate_values, [0, 1, 2])


def test_source_adapter_preserves_legal_mask() -> None:
    model = SourceResidualActionModel(
        state_size=6,
        candidate_size=4,
        hidden_size=8,
        bottleneck_size=3,
        source_count=3,
        residual_scale=0.25,
        dropout=0.0,
    )
    legal = torch.tensor(
        [[True, False, True, False], [False, True, False, True]], dtype=torch.bool
    )
    probabilities = model.action_probabilities(
        torch.randn(2, 6),
        torch.randn(4, 4),
        torch.tensor([0, 2]),
        torch.tensor([0, 1, 2, 2]),
        legal,
    )
    assert probabilities.shape == legal.shape
    assert torch.allclose(probabilities[~legal], torch.zeros(4))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_source_adapter_parameter_blocks_are_distinct() -> None:
    model = SourceResidualActionModel(
        state_size=6,
        candidate_size=4,
        hidden_size=8,
        bottleneck_size=3,
        source_count=3,
        residual_scale=0.25,
        dropout=0.0,
    )
    assert model.state_adapters[0].down.weight.data_ptr() != model.state_adapters[1].down.weight.data_ptr()
    assert model.candidate_adapters[1].up.weight.data_ptr() != model.candidate_adapters[2].up.weight.data_ptr()
