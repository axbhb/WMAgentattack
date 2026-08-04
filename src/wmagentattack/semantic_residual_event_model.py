"""Semantic, candidate-constrained victim event world model.

This is the second low-data factorization used by WMagentattack.  It addresses
three concrete failure modes found in the first Event Transformer diagnostic:

* the output vocabulary is constructed from the *training candidate catalog*,
  rather than only from tools that happened to be selected in training;
* tool names are represented compositionally (for example, ``restaurant`` +
  ``generate``), and logits are restricted to the label-blind candidate set;
* repeated utility/security outcomes are represented as a static configuration
  value anchor plus an event-prefix residual.

The module deliberately does not learn AgentDojo tool execution or checker
state transitions.  Those remain exact simulator operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .event_world_model import FactorizedEventWorldModel, JOINT_OUTCOME_ORDER


def tokenize_skill_name(name: str) -> tuple[str, ...]:
    """Return stable components for one normalized skill name."""

    if name.startswith("<") and name.endswith(">"):
        return (name,)
    tokens = tuple(part for part in name.lower().split("_") if part)
    return tokens or (name.lower(),)


def build_skill_token_incidence(
    skill_names: Sequence[str],
) -> tuple[dict[str, int], Tensor]:
    """Build a row-normalized skill-to-name-token incidence matrix."""

    if not skill_names:
        raise ValueError("skill_names cannot be empty")
    components = [tokenize_skill_name(name) for name in skill_names]
    token_names = sorted({token for row in components for token in row})
    token_vocab = {token: index for index, token in enumerate(token_names)}
    incidence = torch.zeros(len(skill_names), len(token_names), dtype=torch.float32)
    for skill_index, row in enumerate(components):
        weight = 1.0 / len(row)
        for token in row:
            incidence[skill_index, token_vocab[token]] += weight
    return token_vocab, incidence


@dataclass(frozen=True)
class SemanticResidualEventConfig:
    num_skills: int
    num_skill_tokens: int
    semantic_cardinalities: tuple[int, ...]
    num_domains: int
    num_argument_signatures: int
    hidden_size: int = 96
    num_layers: int = 2
    num_heads: int = 4
    feedforward_size: int = 192
    dropout: float = 0.1
    max_sequence_length: int = 65
    pad_skill_id: int = 0
    minimum_concentration: float = 0.05
    skill_residual_scale: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            "num_skills",
            "num_skill_tokens",
            "num_domains",
            "num_argument_signatures",
            "hidden_size",
            "num_layers",
            "num_heads",
            "feedforward_size",
            "max_sequence_length",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.semantic_cardinalities or any(
            value <= 0 for value in self.semantic_cardinalities
        ):
            raise ValueError("semantic_cardinalities must be non-empty and positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.minimum_concentration <= 0.0:
            raise ValueError("minimum_concentration must be positive")
        if self.skill_residual_scale < 0.0:
            raise ValueError("skill_residual_scale must be non-negative")


class SemanticResidualEventWorldModel(nn.Module):
    """Causal event model with semantic context and anchored joint value."""

    def __init__(
        self,
        config: SemanticResidualEventConfig,
        skill_token_incidence: Tensor,
    ) -> None:
        super().__init__()
        if skill_token_incidence.shape != (
            config.num_skills,
            config.num_skill_tokens,
        ):
            raise ValueError(
                "skill_token_incidence must have shape "
                f"[{config.num_skills}, {config.num_skill_tokens}]"
            )
        if torch.any(skill_token_incidence < 0):
            raise ValueError("skill_token_incidence cannot contain negative values")
        row_sums = skill_token_incidence.sum(-1)
        if torch.any(row_sums <= 0):
            raise ValueError("every skill needs at least one name token")

        self.config = config
        hidden = config.hidden_size
        normalized = skill_token_incidence / row_sums.unsqueeze(-1)
        self.register_buffer("skill_token_incidence", normalized)
        self.skill_token_embedding = nn.Embedding(config.num_skill_tokens, hidden)
        self.skill_residual = nn.Embedding(config.num_skills, hidden)
        nn.init.normal_(self.skill_residual.weight, mean=0.0, std=0.02)
        self.skill_norm = nn.LayerNorm(hidden)
        self.skill_output_bias = nn.Parameter(torch.zeros(config.num_skills))

        self.semantic_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, hidden)
            for cardinality in config.semantic_cardinalities
        )
        self.domain_embedding = nn.Embedding(config.num_domains, hidden)
        self.clean_prior_projection = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.context_norm = nn.LayerNorm(hidden)
        self.position_embedding = nn.Embedding(config.max_sequence_length, hidden)

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

        self.argument_signature_head = nn.Linear(
            hidden, config.num_argument_signatures
        )
        self.stop_head = nn.Linear(hidden, 1)
        self.static_joint_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        self.dynamic_joint_residual_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        # Start from the static value model and make the trajectory contribution
        # earn its way into the prediction during optimization.
        nn.init.zeros_(self.dynamic_joint_residual_head[-1].weight)
        nn.init.zeros_(self.dynamic_joint_residual_head[-1].bias)

    def _skill_table(self) -> Tensor:
        composed = self.skill_token_incidence @ self.skill_token_embedding.weight
        composed = composed + self.config.skill_residual_scale * self.skill_residual.weight
        return self.skill_norm(composed)

    def _static_context(
        self,
        semantic_context_ids: Tensor,
        domain_ids: Tensor,
        clean_utility_prior: Tensor,
    ) -> Tensor:
        if semantic_context_ids.ndim != 2:
            raise ValueError("semantic_context_ids must have shape [batch, fields]")
        batch, fields = semantic_context_ids.shape
        if fields != len(self.semantic_embeddings):
            raise ValueError("semantic context field count does not match config")
        if domain_ids.shape != (batch,):
            raise ValueError("domain_ids must have shape [batch]")
        if clean_utility_prior.shape not in {(batch,), (batch, 1)}:
            raise ValueError("clean_utility_prior must have shape [batch] or [batch, 1]")
        prior = clean_utility_prior.reshape(batch, 1).to(
            semantic_context_ids.device, torch.float32
        )
        context = self.domain_embedding(domain_ids)
        context = context + self.clean_prior_projection(prior)
        for field_index, embedding in enumerate(self.semantic_embeddings):
            context = context + embedding(semantic_context_ids[:, field_index])
        return self.context_norm(context / math.sqrt(fields + 2.0))

    def forward(
        self,
        skill_ids: Tensor,
        semantic_context_ids: Tensor,
        domain_ids: Tensor,
        clean_utility_prior: Tensor,
        attention_mask: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if skill_ids.ndim != 2:
            raise ValueError("skill_ids must have shape [batch, time]")
        batch, length = skill_ids.shape
        if length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds configured max_sequence_length")
        if attention_mask is None:
            attention_mask = skill_ids.ne(self.config.pad_skill_id)
        attention_mask = attention_mask.bool()
        if attention_mask.shape != skill_ids.shape:
            raise ValueError("attention_mask shape must match skill_ids")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("every sequence needs at least one unmasked token")
        if candidate_mask is not None and candidate_mask.shape != (
            batch,
            length,
            self.config.num_skills,
        ):
            raise ValueError("candidate_mask must have shape [batch, time, skills]")

        static_context = self._static_context(
            semantic_context_ids, domain_ids, clean_utility_prior
        )
        table = self._skill_table()
        positions = torch.arange(length, device=skill_ids.device).unsqueeze(0)
        hidden = F.embedding(skill_ids, table)
        hidden = hidden + self.position_embedding(positions)
        hidden = hidden + static_context.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=skill_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
        )
        encoded = self.final_norm(encoded)
        next_skill_logits = torch.einsum("bth,sh->bts", encoded, table)
        next_skill_logits = next_skill_logits / math.sqrt(self.config.hidden_size)
        next_skill_logits = next_skill_logits + self.skill_output_bias
        if candidate_mask is not None:
            next_skill_logits = next_skill_logits.masked_fill(
                ~candidate_mask.bool(), torch.finfo(next_skill_logits.dtype).min
            )

        last_indices = attention_mask.long().sum(dim=1) - 1
        pooled = encoded[torch.arange(batch, device=skill_ids.device), last_indices]
        static_logits = self.static_joint_head(static_context)
        dynamic_residual = self.dynamic_joint_residual_head(pooled)
        dynamic_logits = static_logits + dynamic_residual
        static_concentration = F.softplus(static_logits) + self.config.minimum_concentration
        dynamic_concentration = (
            F.softplus(dynamic_logits) + self.config.minimum_concentration
        )
        return {
            "next_skill_logits": next_skill_logits,
            "argument_signature_logits": self.argument_signature_head(encoded),
            "stop_logits": self.stop_head(encoded).squeeze(-1),
            "static_joint_concentration": static_concentration,
            "dynamic_joint_concentration": dynamic_concentration,
            "dynamic_joint_logit_residual": dynamic_residual,
        }

    @staticmethod
    def dirichlet_multinomial_nll(
        concentration: Tensor,
        counts: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        return FactorizedEventWorldModel.dirichlet_multinomial_nll(
            concentration, counts, sample_weight
        )

    @staticmethod
    def outcome_probabilities(concentration: Tensor) -> dict[str, Tensor]:
        return FactorizedEventWorldModel.outcome_probabilities(concentration)

    def loss(
        self,
        outputs: dict[str, Tensor],
        *,
        event_loss_mask: Tensor,
        next_skill_targets: Tensor,
        candidate_mask: Tensor | None = None,
        argument_signature_targets: Tensor | None = None,
        stop_targets: Tensor | None = None,
        joint_outcome_counts: Tensor | None = None,
        joint_sample_weight: Tensor | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        loss_weights = {
            "skill": 1.0,
            "argument": 0.1,
            "stop": 0.1,
            "static_joint": 0.25,
            "dynamic_joint": 0.5,
            "dynamic_residual_penalty": 0.01,
            **(weights or {}),
        }
        mask = event_loss_mask.bool()
        if not torch.any(mask):
            raise ValueError("event_loss_mask must contain at least one event")
        if candidate_mask is not None:
            allowed = candidate_mask.gather(
                -1, next_skill_targets.unsqueeze(-1)
            ).squeeze(-1)
            if torch.any(mask & ~allowed):
                raise ValueError("a next-skill target is absent from its candidate set")

        components: dict[str, Tensor] = {}
        components["skill"] = F.cross_entropy(
            outputs["next_skill_logits"][mask], next_skill_targets[mask]
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
        if joint_outcome_counts is not None:
            components["static_joint"] = self.dirichlet_multinomial_nll(
                outputs["static_joint_concentration"],
                joint_outcome_counts,
                joint_sample_weight,
            )
            components["dynamic_joint"] = self.dirichlet_multinomial_nll(
                outputs["dynamic_joint_concentration"],
                joint_outcome_counts,
                joint_sample_weight,
            )
        components["dynamic_residual_penalty"] = outputs[
            "dynamic_joint_logit_residual"
        ].square().mean()
        total = sum(loss_weights[name] * value for name, value in components.items())
        return {"total": total, **components}

    def export_config(self) -> dict[str, Any]:
        payload = dict(self.config.__dict__)
        payload["semantic_cardinalities"] = list(
            self.config.semantic_cardinalities
        )
        payload["joint_outcome_order"] = list(JOINT_OUTCOME_ORDER)
        return payload
