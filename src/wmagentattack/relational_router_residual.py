"""Non-lexical relational signatures and state-dependent sparse residuals.

The signature deliberately removes every hashed lexical coordinate.  Routing
uses only inference-visible node-type counts, relation counts, and numeric
affordance/evidence summaries.  Task IDs, tracks, labels, and raw values are
never inputs to the router.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .relational_slot_latent import NODE_TYPES, RELATION_TYPES
from .structured_residual_dynamics import StructuredResidualDynamics


def stack_relation_signature_features(
    slots: Mapping[str, object], *, hash_dimension: int
) -> np.ndarray:
    """Compress graph topology and numeric state while excluding lexical hashes."""
    features = np.asarray(slots["features"], dtype=np.float32)
    node_types = np.asarray(slots["node_types"], dtype=np.int64)
    relations = np.asarray(slots["relations"], dtype=np.int64)
    mask = np.asarray(slots["mask"], dtype=bool)
    if features.ndim != 3 or features.shape[:2] != mask.shape:
        raise ValueError("invalid slot feature shape")
    if features.shape[2] <= hash_dimension:
        raise ValueError("no numeric coordinates remain after lexical removal")
    rows: list[np.ndarray] = []
    for index in range(len(features)):
        valid = mask[index]
        count = max(int(valid.sum()), 1)
        node_histogram = np.bincount(
            node_types[index, valid], minlength=len(NODE_TYPES)
        ).astype(np.float32) / count
        pair_mask = valid[:, None] & valid[None, :]
        pair_count = max(int(pair_mask.sum()), 1)
        relation_histogram = np.bincount(
            relations[index][pair_mask], minlength=len(RELATION_TYPES)
        ).astype(np.float32)[1:] / pair_count
        numeric = features[index, valid, hash_dimension:]
        numeric_mean = numeric.mean(axis=0)
        numeric_max = numeric.max(axis=0)
        rows.append(np.concatenate((node_histogram, relation_histogram, numeric_mean, numeric_max)))
    return np.stack(rows).astype(np.float32)


def _adapter(hidden_size: int, bottleneck_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, bottleneck_size), nn.GELU(),
        nn.Linear(bottleneck_size, hidden_size),
    )


class _BaseRelationalSignatureResidual(nn.Module):
    def __init__(
        self, *, candidate_size: int, route_feature_size: int,
        hidden_size: int, dropout: float,
    ) -> None:
        super().__init__()
        self.base = StructuredResidualDynamics(
            candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout
        )
        self.signature_encoder = nn.Sequential(
            nn.LayerNorm(route_feature_size), nn.Linear(route_feature_size, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.successor_projection = nn.Linear(hidden_size, candidate_size)
        nn.init.zeros_(self.successor_projection.weight)
        nn.init.zeros_(self.successor_projection.bias)

    def _encoded(self, signatures: Tensor) -> Tensor:
        return self.signature_encoder(signatures)

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


class DenseRelationalSignatureResidual(_BaseRelationalSignatureResidual):
    """Parameter-matched dense control using the identical relational signature."""

    def __init__(
        self, *, candidate_size: int, route_feature_size: int, hidden_size: int,
        dense_bottleneck_size: int, dropout: float,
    ) -> None:
        super().__init__(
            candidate_size=candidate_size, route_feature_size=route_feature_size,
            hidden_size=hidden_size, dropout=dropout,
        )
        self.adapter = _adapter(hidden_size, dense_bottleneck_size)
        self.adapter_gate = nn.Parameter(torch.zeros(()))

    def initial_hidden(self, context: Tensor, signatures: Tensor, *ignored: Tensor) -> tuple[Tensor, Tensor]:
        del ignored
        encoded = self._encoded(signatures)
        return context + torch.tanh(self.adapter_gate) * self.adapter(encoded), encoded


class SparseRelationalSignatureResidual(_BaseRelationalSignatureResidual):
    """Top-k state-dependent basis dynamics over a non-lexical relation signature."""

    def __init__(
        self, *, candidate_size: int, route_feature_size: int, hidden_size: int,
        experts: int, active_experts: int, expert_bottleneck_size: int,
        router_hidden_size: int, dropout: float,
    ) -> None:
        super().__init__(
            candidate_size=candidate_size, route_feature_size=route_feature_size,
            hidden_size=hidden_size, dropout=dropout,
        )
        if not 1 <= active_experts < experts:
            raise ValueError("active_experts must be between one and experts")
        self.active_experts = active_experts
        self.experts = nn.ModuleList(
            _adapter(hidden_size, expert_bottleneck_size) for _ in range(experts)
        )
        self.expert_gates = nn.Parameter(torch.zeros(experts))
        self.router = nn.Sequential(
            nn.LayerNorm(route_feature_size),
            nn.Linear(route_feature_size, router_hidden_size), nn.Tanh(),
            nn.Linear(router_hidden_size, experts),
        )

    def routing_weights(self, signatures: Tensor) -> Tensor:
        logits = self.router(signatures)
        top_values, top_indices = torch.topk(logits, self.active_experts, dim=1)
        weights = torch.softmax(top_values, dim=1)
        output = torch.zeros_like(logits)
        return output.scatter(1, top_indices, weights)

    def initial_hidden(self, context: Tensor, signatures: Tensor, *ignored: Tensor) -> tuple[Tensor, Tensor]:
        del ignored
        encoded = self._encoded(signatures)
        weights = self.routing_weights(signatures)
        adapted = torch.zeros_like(encoded)
        for index, expert in enumerate(self.experts):
            adapted = adapted + (
                weights[:, index:index + 1]
                * torch.tanh(self.expert_gates[index])
                * expert(encoded)
            )
        return context + adapted, encoded


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_gap_fraction(left: nn.Module, right: nn.Module) -> float:
    left_count = trainable_parameter_count(left)
    right_count = trainable_parameter_count(right)
    return abs(left_count - right_count) / max(left_count, right_count)
