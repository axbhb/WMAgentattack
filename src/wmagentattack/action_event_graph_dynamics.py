"""Zero-initialized v6 residual conditioned on explicit action-event graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .structured_residual_dynamics import StructuredResidualDynamics


class ActionEventGraphDynamics(nn.Module):
    def __init__(self, *, graph_size: int, candidate_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.base = StructuredResidualDynamics(
            candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.graph_gate = nn.Parameter(torch.zeros(()))

    def condition(self, hidden: Tensor, graph: Tensor) -> Tensor:
        return hidden + torch.tanh(self.graph_gate) * self.graph_encoder(graph)

    def advance(self, hidden: Tensor, action_inputs: Tensor, graph: Tensor) -> Tensor:
        return self.condition(self.base.advance(hidden, action_inputs), graph)

    def one_step_delta_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.one_step_delta_logits(hidden, candidates)

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.rollout_logits(hidden, candidates)

    def projected_context(self, hidden: Tensor) -> Tensor:
        return self.base.projected_context(hidden)

    def joint_logits(self, hidden: Tensor) -> Tensor:
        return self.base.joint_logits(hidden)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
