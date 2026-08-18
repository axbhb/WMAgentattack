"""Domain-routed affordance latents with a parameter-matched dense control.

Routing uses only the inference-visible AgentDojo track.  Task identifiers,
outcomes, confirmation membership, and free text outside the legal interface
are never exposed to the router.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .relational_slot_latent import RelationalSlotEncoder
from .structured_residual_dynamics import StructuredResidualDynamics


DOMAIN_NAMES = ("banking", "slack", "travel", "workspace")
_DOMAIN_INDEX = {name: index for index, name in enumerate(DOMAIN_NAMES)}


def domain_index_from_causal(causal: Mapping[str, object]) -> int:
    track = str(causal.get("track", ""))
    domain = track.rsplit(":", 1)[-1].lower()
    if domain not in _DOMAIN_INDEX:
        raise ValueError(f"unsupported inference-visible track: {track!r}")
    return _DOMAIN_INDEX[domain]


def stack_domain_indices(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [domain_index_from_causal(row["causal_model_input"]) for row in rows],
        dtype=np.int64,
    )


def _adapter(hidden_size: int, bottleneck_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, bottleneck_size), nn.GELU(),
        nn.Linear(bottleneck_size, hidden_size),
    )


class _BaseAffordanceResidual(nn.Module):
    def __init__(
        self, *, candidate_size: int, slot_feature_size: int, hidden_size: int,
        slot_layers: int, dropout: float,
    ) -> None:
        super().__init__()
        # Keep construction order identical across the dense and routed arms.
        self.base = StructuredResidualDynamics(
            candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout
        )
        self.slot_encoder = RelationalSlotEncoder(
            feature_size=slot_feature_size, hidden_size=hidden_size,
            layers=slot_layers, dropout=dropout,
        )
        self.successor_projection = nn.Linear(hidden_size, candidate_size)
        nn.init.zeros_(self.successor_projection.weight)
        nn.init.zeros_(self.successor_projection.bias)

    def slot_latent(
        self, features: Tensor, node_types: Tensor, relations: Tensor, mask: Tensor,
    ) -> Tensor:
        return self.slot_encoder(features, node_types, relations, mask)

    def successor_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.successor_projection(hidden) @ candidates.T / math.sqrt(candidates.shape[1])

    def one_step_delta_logits(self, context: Tensor, candidates: Tensor) -> Tensor:
        return self.base.one_step_delta_logits(context, candidates)

    def advance(self, hidden: Tensor, action_inputs: Tensor) -> Tensor:
        return self.base.advance(hidden, action_inputs)

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        return self.base.rollout_logits(hidden, candidates)

    def projected_context(self, hidden: Tensor) -> Tensor:
        return self.base.projected_context(hidden)

    def joint_logits(self, hidden: Tensor) -> Tensor:
        return self.base.joint_logits(hidden)


class DenseCapacityAffordanceResidual(_BaseAffordanceResidual):
    """A dense adapter matched to the total routed-expert parameter budget."""

    def __init__(
        self, *, candidate_size: int, slot_feature_size: int, hidden_size: int,
        slot_layers: int, dense_bottleneck_size: int, dropout: float,
    ) -> None:
        super().__init__(
            candidate_size=candidate_size, slot_feature_size=slot_feature_size,
            hidden_size=hidden_size, slot_layers=slot_layers, dropout=dropout,
        )
        self.adapter = _adapter(hidden_size, dense_bottleneck_size)
        self.adapter_gate = nn.Parameter(torch.zeros(()))

    def initial_hidden(
        self, context: Tensor, features: Tensor, node_types: Tensor,
        relations: Tensor, mask: Tensor, domain_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del domain_indices
        slot = self.slot_latent(features, node_types, relations, mask)
        adapted = self.adapter(slot)
        return context + torch.tanh(self.adapter_gate) * adapted, slot


class DomainExpertAffordanceResidual(_BaseAffordanceResidual):
    """Four deterministic domain experts over a shared affordance encoder."""

    def __init__(
        self, *, candidate_size: int, slot_feature_size: int, hidden_size: int,
        slot_layers: int, expert_bottleneck_size: int, dropout: float,
    ) -> None:
        super().__init__(
            candidate_size=candidate_size, slot_feature_size=slot_feature_size,
            hidden_size=hidden_size, slot_layers=slot_layers, dropout=dropout,
        )
        self.experts = nn.ModuleList(
            _adapter(hidden_size, expert_bottleneck_size) for _ in DOMAIN_NAMES
        )
        self.expert_gates = nn.Parameter(torch.zeros(len(DOMAIN_NAMES)))

    def initial_hidden(
        self, context: Tensor, features: Tensor, node_types: Tensor,
        relations: Tensor, mask: Tensor, domain_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if domain_indices.ndim != 1 or len(domain_indices) != len(context):
            raise ValueError("one causal domain index is required per state")
        if torch.any(domain_indices < 0) or torch.any(domain_indices >= len(DOMAIN_NAMES)):
            raise ValueError("domain index out of range")
        slot = self.slot_latent(features, node_types, relations, mask)
        adapted = torch.zeros_like(slot)
        for domain, expert in enumerate(self.experts):
            selected = domain_indices == domain
            if torch.any(selected):
                adapted[selected] = torch.tanh(self.expert_gates[domain]) * expert(slot[selected])
        return context + adapted, slot


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def routed_parameter_gap_fraction(
    dense: DenseCapacityAffordanceResidual,
    expert: DomainExpertAffordanceResidual,
) -> float:
    left = trainable_parameter_count(dense)
    right = trainable_parameter_count(expert)
    return abs(left - right) / max(left, right)
