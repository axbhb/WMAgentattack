"""Tiny action-conditioned transition models for the frozen v20 comparison.

The models predict semantic transition effects only.  They intentionally do
not expose reward, utility, security, actor, critic, or planner heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectStructuredTransition(nn.Module):
    """Structured Markov v3 transition probe with no recurrent state."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int, targets: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.effect_head = nn.Linear(hidden_size, targets)
        self.execution_head = nn.Linear(hidden_size, 1)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(torch.cat((state, action), dim=-1))
        return self.effect_head(hidden), self.execution_head(hidden).squeeze(-1)


class RecurrentResidualTransition(nn.Module):
    """v6-style recurrent transition baseline with a zero-start residual gate."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int, targets: int) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.effect_head = nn.Linear(hidden_size, targets)
        self.execution_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance(self, hidden: Tensor, action: Tensor) -> Tensor:
        proposal = self.cell(self.action_encoder(action), hidden)
        return hidden + torch.tanh(self.residual_gate) * (proposal - hidden)

    def predict_hidden(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.effect_head(hidden), self.execution_head(hidden).squeeze(-1)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.predict_hidden(self.advance(self.initial_hidden(state), action))


class InterventionModularTransition(nn.Module):
    """Factorized execution/effect model with zero-init latent residual dynamics.

    A learned execution gate mixes success- and error-conditional effect
    experts.  Pair and sequence supervision are applied by the training
    protocol, while this module remains free of task and group identifiers.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int, targets: int) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.next_norm = nn.LayerNorm(hidden_size)
        self.execution_head = nn.Linear(hidden_size * 2, 1)
        self.shared_effect_head = nn.Linear(hidden_size, targets)
        self.success_effect_head = nn.Linear(hidden_size, targets)
        self.error_effect_head = nn.Linear(hidden_size, targets)

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(self, hidden: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        action_hidden = self.action_encoder(action)
        joint = torch.cat((hidden, action_hidden), dim=-1)
        execution_logit = self.execution_head(joint).squeeze(-1)
        following = self.next_norm(hidden + self.residual(joint))
        return following, execution_logit

    def predict_hidden(self, hidden: Tensor, execution_logit: Tensor) -> Tensor:
        error_probability = torch.sigmoid(execution_logit)[:, None]
        conditional = (
            (1.0 - error_probability) * self.success_effect_head(hidden)
            + error_probability * self.error_effect_head(hidden)
        )
        return self.shared_effect_head(hidden) + conditional

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        hidden, execution_logit = self.advance_with_execution(
            self.initial_hidden(state), action
        )
        return self.predict_hidden(hidden, execution_logit), execution_logit


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_transition_only(model: nn.Module) -> None:
    forbidden = {"actor", "critic", "planner", "reward_head", "utility_head", "value_head"}
    present = sorted(forbidden & set(dict(model.named_modules())))
    if present:
        raise ValueError(f"forbidden modules enabled: {present}")
