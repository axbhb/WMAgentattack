"""Zero-initialized recurrent residual dynamics over a frozen structured model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


class StructuredResidualDynamics(nn.Module):
    def __init__(self, *, candidate_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.dynamics = nn.GRUCell(hidden_size, hidden_size)
        self.one_step_delta = nn.Linear(hidden_size, 1)
        self.rollout_head = nn.Linear(hidden_size, 1)
        self.future_context_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(dropout)
        )
        self.future_joint_head = nn.Linear(hidden_size, len(JOINT_OUTCOME_CLASSES))
        nn.init.zeros_(self.one_step_delta.weight); nn.init.zeros_(self.one_step_delta.bias)
        nn.init.zeros_(self.rollout_head.weight); nn.init.zeros_(self.rollout_head.bias)

    def one_step_delta_logits(self, context: Tensor, candidates: Tensor) -> Tensor:
        candidate = self.candidate_encoder(candidates)
        return self.one_step_delta(torch.tanh(context[:, None, :] + candidate[None, :, :])).squeeze(-1)

    def advance(self, hidden: Tensor, action_inputs: Tensor) -> Tensor:
        return self.dynamics(self.action_encoder(action_inputs), hidden)

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        candidate = self.candidate_encoder(candidates)
        return self.rollout_head(torch.tanh(hidden[:, None, :] + candidate[None, :, :])).squeeze(-1)

    def projected_context(self, hidden: Tensor) -> Tensor:
        return self.future_context_projection(hidden)

    def joint_logits(self, hidden: Tensor) -> Tensor:
        return self.future_joint_head(hidden)
