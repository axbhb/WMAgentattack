"""Zero-initialized exact/evidence late fusion with an action-relevant latent residual."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .structured_residual_dynamics import StructuredResidualDynamics


class LateFusionDistilledDynamics(nn.Module):
    def __init__(self, *, graph_size: int, candidate_size: int, hidden_size: int, latent_size: int, dropout: float):
        super().__init__()
        self.base = StructuredResidualDynamics(candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout)
        self.exact_encoder = nn.Sequential(nn.Linear(graph_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.evidence_encoder = nn.Sequential(nn.Linear(graph_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(dropout)
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, latent_size), nn.Tanh()
        )
        self.latent_decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.exact_gate = nn.Parameter(torch.zeros(()))
        self.evidence_gate = nn.Parameter(torch.zeros(()))
        self.interaction_gate = nn.Parameter(torch.zeros(()))
        self.latent_gate = nn.Parameter(torch.zeros(()))

    def condition(self, hidden: Tensor, exact_graph: Tensor, evidence_graph: Tensor) -> Tensor:
        exact = self.exact_encoder(exact_graph); evidence = self.evidence_encoder(evidence_graph)
        interaction = self.interaction(torch.cat((exact, evidence, exact * evidence), dim=-1))
        return (hidden + torch.tanh(self.exact_gate) * exact + torch.tanh(self.evidence_gate) * evidence
                + torch.tanh(self.interaction_gate) * interaction)

    def advance_latent(self, hidden: Tensor, action_inputs: Tensor) -> tuple[Tensor, Tensor]:
        action = self.base.action_encoder(action_inputs)
        latent = self.latent_predictor(torch.cat((hidden, action), dim=-1))
        advanced = self.base.advance(hidden, action_inputs) + torch.tanh(self.latent_gate) * self.latent_decoder(latent)
        return advanced, latent

    def one_step_delta_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.one_step_delta_logits(hidden, candidates)

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.rollout_logits(hidden, candidates)

    def projected_context(self, hidden: Tensor) -> Tensor:
        return self.base.projected_context(hidden)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
