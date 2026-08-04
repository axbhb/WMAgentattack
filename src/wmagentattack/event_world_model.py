"""Factorized Event Transformer for victim-policy world modelling.

Unlike the historical Dreamer adaptation, this model does not reconstruct raw
text observations and has no actor/critic.  It predicts the stochastic victim
action (skill, argument signature, stop) while exact tool/state transitions
remain in :mod:`wmagentattack.exact_simulator`.  Repeated utility/security
outcomes are fitted with a four-cell Dirichlet-multinomial likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


JOINT_OUTCOME_ORDER = (
    "attack0_utility0",
    "attack0_utility1",
    "attack1_utility0",
    "attack1_utility1",
)


@dataclass(frozen=True)
class EventWorldModelConfig:
    num_tools: int
    num_attack_contexts: int
    num_domains: int
    num_argument_signatures: int
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_size: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 64
    pad_tool_id: int = 0
    minimum_concentration: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "num_tools",
            "num_attack_contexts",
            "num_domains",
            "num_argument_signatures",
            "hidden_size",
            "num_layers",
            "num_heads",
            "feedforward_size",
            "max_sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.minimum_concentration <= 0.0:
            raise ValueError("minimum_concentration must be positive")


class FactorizedEventWorldModel(nn.Module):
    """Causal victim-event model with factorized utility/risk heads."""

    def __init__(self, config: EventWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.tool_embedding = nn.Embedding(
            config.num_tools, hidden, padding_idx=config.pad_tool_id
        )
        self.attack_embedding = nn.Embedding(config.num_attack_contexts, hidden)
        self.domain_embedding = nn.Embedding(config.num_domains, hidden)
        self.position_embedding = nn.Embedding(config.max_sequence_length, hidden)
        self.clean_prior_projection = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(hidden)

        self.next_tool_head = nn.Linear(hidden, config.num_tools)
        self.argument_signature_head = nn.Linear(
            hidden, config.num_argument_signatures
        )
        self.stop_head = nn.Linear(hidden, 1)
        self.progress_delta_head = nn.Linear(hidden, 1)
        self.utility_residual_head = nn.Linear(hidden, 1)
        self.joint_concentration_head = nn.Linear(hidden, len(JOINT_OUTCOME_ORDER))

    def forward(
        self,
        tool_ids: Tensor,
        attack_context_ids: Tensor,
        domain_ids: Tensor,
        clean_utility_prior: Tensor,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if tool_ids.ndim != 2:
            raise ValueError("tool_ids must have shape [batch, time]")
        batch, length = tool_ids.shape
        if length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured max_sequence_length")
        if attack_context_ids.shape != (batch,) or domain_ids.shape != (batch,):
            raise ValueError("context and domain ids must have shape [batch]")
        if clean_utility_prior.shape not in {(batch,), (batch, 1)}:
            raise ValueError("clean_utility_prior must have shape [batch] or [batch, 1]")
        if attention_mask is None:
            attention_mask = tool_ids.ne(self.config.pad_tool_id)
        attention_mask = attention_mask.bool()
        if attention_mask.shape != tool_ids.shape:
            raise ValueError("attention_mask shape must match tool_ids")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("every sequence needs at least one unmasked event")

        positions = torch.arange(length, device=tool_ids.device).unsqueeze(0)
        prior = clean_utility_prior.reshape(batch, 1).to(tool_ids.device, torch.float32)
        hidden = self.tool_embedding(tool_ids)
        hidden = hidden + self.position_embedding(positions)
        hidden = hidden + self.attack_embedding(attack_context_ids).unsqueeze(1)
        hidden = hidden + self.domain_embedding(domain_ids).unsqueeze(1)
        hidden = hidden + self.clean_prior_projection(prior).unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones(length, length, device=tool_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
        )
        encoded = self.final_norm(encoded)
        last_indices = attention_mask.long().sum(dim=1) - 1
        pooled = encoded[torch.arange(batch, device=tool_ids.device), last_indices]
        concentration = F.softplus(self.joint_concentration_head(pooled))
        concentration = concentration + self.config.minimum_concentration
        return {
            "next_tool_logits": self.next_tool_head(encoded),
            "argument_signature_logits": self.argument_signature_head(encoded),
            "stop_logits": self.stop_head(encoded).squeeze(-1),
            "progress_delta": self.progress_delta_head(encoded).squeeze(-1),
            "utility_logit_residual": self.utility_residual_head(pooled).squeeze(-1),
            "joint_concentration": concentration,
        }

    @staticmethod
    def dirichlet_multinomial_nll(
        concentration: Tensor,
        counts: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        """Negative log likelihood for repeated four-cell outcomes."""

        if concentration.shape != counts.shape or concentration.shape[-1] != 4:
            raise ValueError("concentration and counts must both have shape [batch, 4]")
        if torch.any(counts < 0):
            raise ValueError("joint counts must be non-negative")
        counts = counts.to(concentration.dtype)
        trials = counts.sum(dim=-1)
        valid = trials > 0
        if not torch.any(valid):
            return concentration.sum() * 0.0
        alpha = concentration[valid]
        observed = counts[valid]
        n = trials[valid]
        log_probability = torch.lgamma(n + 1.0) - torch.lgamma(observed + 1.0).sum(-1)
        log_probability = log_probability + torch.lgamma(alpha.sum(-1))
        log_probability = log_probability - torch.lgamma(alpha.sum(-1) + n)
        log_probability = log_probability + (
            torch.lgamma(alpha + observed) - torch.lgamma(alpha)
        ).sum(-1)
        losses = -log_probability
        if sample_weight is None:
            return losses.mean()
        weights = sample_weight.to(losses.dtype)[valid]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)

    @staticmethod
    def outcome_probabilities(concentration: Tensor) -> dict[str, Tensor]:
        joint = concentration / concentration.sum(dim=-1, keepdim=True)
        return {
            "joint": joint,
            "utility": joint[..., 1] + joint[..., 3],
            "attack": joint[..., 2] + joint[..., 3],
            "attack_and_utility": joint[..., 3],
        }

    def loss(
        self,
        outputs: dict[str, Tensor],
        *,
        attention_mask: Tensor,
        next_tool_targets: Tensor,
        argument_signature_targets: Tensor | None = None,
        stop_targets: Tensor | None = None,
        progress_delta_targets: Tensor | None = None,
        joint_outcome_counts: Tensor | None = None,
        joint_sample_weight: Tensor | None = None,
        utility_residual_targets: Tensor | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        loss_weights = {
            "tool": 1.0,
            "argument": 0.25,
            "stop": 0.25,
            "progress": 0.1,
            "joint": 0.5,
            "utility_residual": 0.25,
            **(weights or {}),
        }
        mask = attention_mask.bool()
        components: dict[str, Tensor] = {}
        components["tool"] = F.cross_entropy(
            outputs["next_tool_logits"][mask], next_tool_targets[mask]
        )
        if argument_signature_targets is not None:
            components["argument"] = F.cross_entropy(
                outputs["argument_signature_logits"][mask],
                argument_signature_targets[mask],
            )
        if stop_targets is not None:
            components["stop"] = F.binary_cross_entropy_with_logits(
                outputs["stop_logits"][mask], stop_targets.to(torch.float32)[mask]
            )
        if progress_delta_targets is not None:
            valid_progress = mask & torch.isfinite(progress_delta_targets)
            if torch.any(valid_progress):
                components["progress"] = F.smooth_l1_loss(
                    outputs["progress_delta"][valid_progress],
                    progress_delta_targets[valid_progress],
                )
        if joint_outcome_counts is not None:
            components["joint"] = self.dirichlet_multinomial_nll(
                outputs["joint_concentration"],
                joint_outcome_counts,
                joint_sample_weight,
            )
        if utility_residual_targets is not None:
            valid_residual = torch.isfinite(utility_residual_targets)
            if torch.any(valid_residual):
                components["utility_residual"] = F.smooth_l1_loss(
                    outputs["utility_logit_residual"][valid_residual],
                    utility_residual_targets[valid_residual],
                )
        total = sum(loss_weights[name] * value for name, value in components.items())
        return {"total": total, **components}

    def export_config(self) -> dict[str, Any]:
        return dict(self.config.__dict__)

