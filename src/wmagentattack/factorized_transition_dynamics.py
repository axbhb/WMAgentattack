"""v6 residual dynamics conditioned on predicted semantic-transition factors."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .factorized_transition_labels import FACTOR_CLASSES
from .structured_residual_dynamics import StructuredResidualDynamics


class FactorizedSemanticTransitionDynamics(nn.Module):
    def __init__(
        self, *, candidate_size: int, hidden_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.base = StructuredResidualDynamics(
            candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout
        )
        self.factor_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, len(classes))
            for name, classes in FACTOR_CLASSES.items()
        })
        self.factor_projections = nn.ModuleDict({
            name: nn.Linear(len(classes), hidden_size, bias=False)
            for name, classes in FACTOR_CLASSES.items()
        })
        self.factor_norm = nn.LayerNorm(hidden_size)
        self.factor_gate = nn.Parameter(torch.zeros(()))

    def factor_logits(self, hidden: Tensor) -> dict[str, Tensor]:
        return {name: head(hidden) for name, head in self.factor_heads.items()}

    def _factor_latent_from_probabilities(
        self, probabilities: Mapping[str, Tensor]
    ) -> Tensor:
        values = [self.factor_projections[name](probabilities[name]) for name in FACTOR_CLASSES]
        return self.factor_norm(torch.stack(values).mean(0))

    def condition_predicted(self, hidden: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self.factor_logits(hidden)
        probabilities = {name: torch.softmax(value, dim=-1) for name, value in logits.items()}
        latent = self._factor_latent_from_probabilities(probabilities)
        return hidden + torch.tanh(self.factor_gate) * latent, logits

    def condition_oracle(self, hidden: Tensor, factor_indices: Tensor) -> Tensor:
        if factor_indices.ndim != 2 or factor_indices.shape[1] != len(FACTOR_CLASSES):
            raise ValueError("one oracle index per factor is required")
        probabilities = {}
        for column, (name, classes) in enumerate(FACTOR_CLASSES.items()):
            probabilities[name] = torch.nn.functional.one_hot(
                factor_indices[:, column], num_classes=len(classes)
            ).to(hidden.dtype)
        latent = self._factor_latent_from_probabilities(probabilities)
        return hidden + torch.tanh(self.factor_gate) * latent

    def initial_hidden(self, context: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.condition_predicted(context)

    def advance(self, hidden: Tensor, action_inputs: Tensor) -> Tensor:
        advanced = self.base.advance(hidden, action_inputs)
        return self.condition_predicted(advanced)[0]

    def advance_base(self, hidden: Tensor, action_inputs: Tensor) -> Tensor:
        return self.base.advance(hidden, action_inputs)

    def one_step_delta_logits(self, context: Tensor, candidates: Tensor) -> Tensor:
        return self.base.one_step_delta_logits(context, candidates)

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.rollout_logits(hidden, candidates)

    def projected_context(self, hidden: Tensor) -> Tensor:
        return self.base.projected_context(hidden)

    def joint_logits(self, hidden: Tensor) -> Tensor:
        return self.base.joint_logits(hidden)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
