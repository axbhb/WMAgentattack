"""Canonical relational slots and a small permutation-equivariant latent encoder.

The builder intentionally discards raw goal, observation, and schema strings.
Only controlled semantic terms, local equality relations, interface names, and
numeric summaries are exposed to the learned encoder.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .semantic_state_v3 import build_goal_semantic_frame
from .structured_residual_dynamics import StructuredResidualDynamics


NODE_TYPES = (
    "global",
    "goal_operation",
    "goal_logic",
    "entity",
    "legal_tool",
    "prior_tool",
    "observation_summary",
)
RELATION_TYPES = (
    "none",
    "self",
    "global",
    "goal_group",
    "observation_group",
    "tool_group",
    "shared_entity",
    "prior_tool_match",
)

_TYPE_INDEX = {name: index for index, name in enumerate(NODE_TYPES)}
_RELATION_INDEX = {name: index for index, name in enumerate(RELATION_TYPES)}
_ERROR = re.compile(r"\b(error|failed|failure|invalid|exception)\b", re.I)


def _signed_hash(namespace: str, value: str, dimension: int) -> np.ndarray:
    output = np.zeros(dimension, dtype=np.float32)
    digest = hashlib.blake2b(
        f"{namespace}\0{value}".encode("utf-8"), digest_size=16
    ).digest()
    bucket = int.from_bytes(digest[:8], "little") % dimension
    output[bucket] = 1.0 if digest[8] & 1 else -1.0
    return output


def _tool_family(name: str) -> str:
    lowered = name.lower()
    for family in ("read", "search", "get", "list", "generate", "create", "modify", "update", "delete", "send", "external"):
        if family in lowered:
            return family
    return "other"


@dataclass(frozen=True)
class RelationalSlotState:
    features: np.ndarray
    node_types: np.ndarray
    relations: np.ndarray
    grounding: np.ndarray
    audit: dict[str, Any]


def build_relational_slot_state(
    causal: Mapping[str, Any], *, hash_dimension: int = 16, max_nodes: int = 32
) -> RelationalSlotState:
    required = {
        "source", "track", "trusted_goal", "visible_observation",
        "visible_prior_tool", "legal_tool_names", "tool_schemas",
    }
    missing = sorted(required - set(causal))
    if missing:
        raise ValueError(f"missing causal slot fields: {missing}")
    if hash_dimension <= 0 or max_nodes < 8:
        raise ValueError("invalid slot dimensions")

    goal_text = str(causal["trusted_goal"])
    observation = str(causal["visible_observation"])
    frame = build_goal_semantic_frame(goal_text)
    observation_frame = build_goal_semantic_frame(observation)
    goal_mentions = {(row.kind, row.value) for row in frame.typed_mentions}
    observation_mentions = {
        (row.kind, row.value) for row in observation_frame.typed_mentions
    }
    entity_keys = sorted(goal_mentions | observation_mentions)
    legal_tools = tuple(sorted(map(str, causal["legal_tool_names"])))
    prior_tool = str(causal["visible_prior_tool"])
    feature_size = hash_dimension + 10
    rows: list[np.ndarray] = []
    types: list[int] = []
    groups: list[str] = []
    entity_roles: list[str | None] = []
    labels: list[str] = []

    def add(node_type: str, label: str, numeric: Sequence[float], group: str, role: str | None = None) -> None:
        if len(rows) >= max_nodes:
            return
        nums = np.zeros(10, dtype=np.float32)
        values = np.asarray(tuple(numeric), dtype=np.float32)
        nums[: min(len(values), len(nums))] = values[: len(nums)]
        rows.append(np.concatenate([_signed_hash(node_type, label, hash_dimension), nums]))
        types.append(_TYPE_INDEX[node_type]); groups.append(group); entity_roles.append(role); labels.append(label)

    add(
        "global", f"{causal['source']}::{causal['track']}",
        (
            math.log1p(len(frame.operation_terms)), math.log1p(len(frame.logic_terms)),
            math.log1p(len(entity_keys)), math.log1p(len(legal_tools)),
            float(frame.has_condition), float(frame.has_comparison),
            float(frame.requires_set_coverage), float(frame.requires_uniqueness),
            float(bool(observation)), float(prior_tool != "<START>"),
        ), "global",
    )
    for term in frame.operation_terms:
        add("goal_operation", term, (1.0,), "goal")
    for term in frame.logic_terms:
        add("goal_logic", term, (1.0,), "goal")
    entity_groups = Counter()
    for kind, value in entity_keys:
        in_goal = (kind, value) in goal_mentions
        in_observation = (kind, value) in observation_mentions
        role = "both" if in_goal and in_observation else ("goal" if in_goal else "observation")
        # Local values are used only for equality grouping and are never encoded.
        entity_groups[(kind, role)] += 1
    for (kind, role), group_count in sorted(entity_groups.items()):
        add(
            "entity", kind,
            (
                float(role in {"goal", "both"}),
                float(role in {"observation", "both"}),
                float(role == "both"),
                math.log1p(group_count),
            ),
            "entity", role,
        )
    for name in legal_tools:
        add("legal_tool", f"{name}::{_tool_family(name)}", (1.0,), "tool")
    add("prior_tool", f"{prior_tool}::{_tool_family(prior_tool)}", (float(prior_tool != "<START>"),), "tool")
    add(
        "observation_summary", "visible_observation_summary",
        (
            math.log1p(len(observation.split())), math.log1p(len(observation.splitlines())),
            float(bool(_ERROR.search(observation))), float("{" in observation or "[" in observation),
            math.log1p(len(observation_mentions)),
        ), "observation",
    )

    count = len(rows)
    relations = np.zeros((max_nodes, max_nodes), dtype=np.int64)
    for i in range(count):
        relations[i, i] = _RELATION_INDEX["self"]
        if i:
            relations[0, i] = relations[i, 0] = _RELATION_INDEX["global"]
    for i in range(count):
        for j in range(count):
            if i == j or i == 0 or j == 0:
                continue
            if groups[i] == groups[j] == "goal":
                relations[i, j] = _RELATION_INDEX["goal_group"]
            elif groups[i] == groups[j] == "tool":
                relations[i, j] = _RELATION_INDEX["tool_group"]
            elif groups[i] == groups[j] == "entity" and entity_roles[i] == entity_roles[j] == "both":
                relations[i, j] = _RELATION_INDEX["shared_entity"]
            elif {groups[i], groups[j]} == {"entity", "observation"} and (
                entity_roles[i] in {"observation", "both"} or entity_roles[j] in {"observation", "both"}
            ):
                relations[i, j] = _RELATION_INDEX["observation_group"]
            if {types[i], types[j]} == {_TYPE_INDEX["prior_tool"], _TYPE_INDEX["legal_tool"]}:
                prior_index = i if types[i] == _TYPE_INDEX["prior_tool"] else j
                legal_index = j if prior_index == i else i
                if labels[prior_index].split("::", 1)[0] == labels[legal_index].split("::", 1)[0]:
                    relations[i, j] = _RELATION_INDEX["prior_tool_match"]

    padded = np.zeros((max_nodes, feature_size), dtype=np.float32)
    padded[:count] = np.stack(rows)
    padded_types = np.zeros(max_nodes, dtype=np.int64)
    padded_types[:count] = np.asarray(types, dtype=np.int64)
    grounding = np.asarray(
        [
            float(frame.has_condition), float(frame.has_comparison),
            float(frame.requires_set_coverage), float(frame.requires_uniqueness),
            math.log1p(len(frame.operation_terms)), math.log1p(len(frame.logic_terms)),
            math.log1p(len(goal_mentions)), math.log1p(len(observation_mentions)),
            math.log1p(len(goal_mentions & observation_mentions)), math.log1p(len(legal_tools)),
            float(bool(observation)), float(bool(_ERROR.search(observation))),
        ], dtype=np.float32,
    )
    return RelationalSlotState(
        features=padded,
        node_types=padded_types,
        relations=relations,
        grounding=grounding,
        audit={
            "node_count": count,
            "truncated": count >= max_nodes,
            "raw_values_encoded": False,
            "goal_mentions": len(goal_mentions),
            "observation_mentions": len(observation_mentions),
            "shared_mentions": len(goal_mentions & observation_mentions),
        },
    )


def stack_relational_slot_states(
    rows: Sequence[Mapping[str, Any]], *, hash_dimension: int = 16, max_nodes: int = 32
) -> dict[str, np.ndarray | list[dict[str, Any]]]:
    states = [
        build_relational_slot_state(row["causal_model_input"], hash_dimension=hash_dimension, max_nodes=max_nodes)
        for row in rows
    ]
    return {
        "features": np.stack([row.features for row in states]),
        "node_types": np.stack([row.node_types for row in states]),
        "relations": np.stack([row.relations for row in states]),
        "mask": np.stack([np.arange(max_nodes) < row.audit["node_count"] for row in states]),
        "grounding": np.stack([row.grounding for row in states]),
        "audit": [row.audit for row in states],
    }


class RelationalMessageLayer(nn.Module):
    def __init__(self, hidden_size: int, relation_count: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.relation_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(relation_count)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, relations: Tensor, mask: Tensor) -> Tensor:
        message = torch.zeros_like(hidden)
        for relation_id, projection in enumerate(self.relation_projections[1:], start=1):
            adjacency = (relations == relation_id) & mask[:, :, None] & mask[:, None, :]
            weights = adjacency.to(hidden.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            message = message + torch.bmm(weights, projection(hidden))
        updated = torch.nn.functional.gelu(self.self_projection(hidden) + message)
        return self.norm(hidden + self.dropout(updated)) * mask[:, :, None]


class RelationalSlotEncoder(nn.Module):
    def __init__(
        self, *, feature_size: int, hidden_size: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden_size)
        self.layers = nn.ModuleList(
            RelationalMessageLayer(hidden_size, len(RELATION_TYPES), dropout)
            for _ in range(layers)
        )
        self.pool_query = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU()
        )

    def forward(self, features: Tensor, node_types: Tensor, relations: Tensor, mask: Tensor) -> Tensor:
        hidden = (self.input_projection(features) + self.type_embedding(node_types)) * mask[:, :, None]
        for layer in self.layers:
            hidden = layer(hidden, relations, mask)
        scores = torch.einsum("bnh,h->bn", hidden, self.pool_query)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        pooled = torch.einsum("bn,bnh->bh", torch.softmax(scores, dim=1), hidden)
        return self.output(pooled)


class SlotAugmentedResidualDynamics(nn.Module):
    """v6 dynamics with a zero-gated relational state residual."""

    def __init__(
        self, *, candidate_size: int, slot_feature_size: int, hidden_size: int,
        slot_layers: int, dropout: float,
    ) -> None:
        super().__init__()
        # Instantiate the frozen-control-compatible residual first so its seeded
        # initialization remains identical to v6.
        self.base = StructuredResidualDynamics(
            candidate_size=candidate_size, hidden_size=hidden_size, dropout=dropout
        )
        self.slot_encoder = RelationalSlotEncoder(
            feature_size=slot_feature_size, hidden_size=hidden_size,
            layers=slot_layers, dropout=dropout,
        )
        self.slot_gate = nn.Parameter(torch.zeros(()))

    def initial_hidden(
        self, context: Tensor, features: Tensor, node_types: Tensor,
        relations: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        slot = self.slot_encoder(features, node_types, relations, mask)
        return context + torch.tanh(self.slot_gate) * slot, slot

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


class GroundedPredictiveSlotResidual(SlotAugmentedResidualDynamics):
    """Stage B adds JEPA prediction and training-only semantic grounding."""

    def __init__(
        self, *, candidate_size: int, slot_feature_size: int, hidden_size: int,
        slot_layers: int, grounding_size: int, dropout: float,
    ) -> None:
        super().__init__(
            candidate_size=candidate_size, slot_feature_size=slot_feature_size,
            hidden_size=hidden_size, slot_layers=slot_layers, dropout=dropout,
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Linear(hidden_size, hidden_size),
        )
        self.static_grounding_head = nn.Linear(hidden_size, grounding_size)
        self.transition_grounding_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, grounding_size),
        )

    def predict_slot_latent(self, hidden: Tensor) -> Tensor:
        return self.latent_predictor(hidden)

    def static_grounding(self, slot: Tensor) -> Tensor:
        return self.static_grounding_head(slot)

    def transition_grounding(self, start_slot: Tensor, predicted_slot: Tensor) -> Tensor:
        return self.transition_grounding_head(torch.cat([start_slot, predicted_slot], dim=-1))
