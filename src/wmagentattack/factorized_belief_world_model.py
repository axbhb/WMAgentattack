"""Factorized neuro-symbolic belief world model for observed AgentDojo traces.

The module deliberately keeps deterministic interface constraints outside the
learned dynamics.  It learns only (1) a typed causal-state belief, (2) the
victim's next legal action, and (3) observable execution outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .adjacent_transition import OBSERVED_OUTCOME_TARGETS


TYPED_STATE_NODES = (
    "source_track",
    "trusted_goal",
    "visible_observation",
    "visible_prior_tool",
    "legal_interface",
    "tool_schema",
    "entity_evidence",
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_@.'-]*|[-+]?\d+(?:\.\d+)?")
_ENTITY = re.compile(
    r"(?:[A-Z][A-Za-z0-9_-]{2,}|[-+]?\d+(?:\.\d+)?|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)


def _signed_bucket(namespace: str, token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        f"{namespace}\0{token}".encode("utf-8"), digest_size=16
    ).digest()
    bucket = int.from_bytes(digest[:8], "little") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


def _hashed_tokens(text: str, *, namespace: str, dimension: int) -> np.ndarray:
    output = np.zeros(dimension, dtype=np.float32)
    tokens = [token.lower() for token in _TOKEN.findall(text)]
    for token in tokens:
        bucket, sign = _signed_bucket(namespace, token, dimension)
        output[bucket] += sign
    norm = float(np.linalg.norm(output))
    if norm > 0.0:
        output /= norm
    return output


def _numeric_features(text: str) -> np.ndarray:
    tokens = _TOKEN.findall(text)
    lowered = [token.lower() for token in tokens]
    total = max(1, len(tokens))
    return np.asarray(
        [
            math.log1p(len(tokens)),
            math.log1p(len(set(lowered))),
            sum(any(char.isdigit() for char in token) for token in tokens) / total,
            sum("@" in token for token in tokens) / total,
            sum(token.isupper() and len(token) > 1 for token in tokens) / total,
            float("error" in text.lower() or "failed" in text.lower()),
            float("instruction" in text.lower()),
            float(bool(text.strip())),
        ],
        dtype=np.float32,
    )


def typed_state_node_features(
    causal_model_input: Mapping[str, Any], *, hash_dimension: int
) -> np.ndarray:
    """Encode visible causal fields as typed nodes instead of one text bag."""

    required = {
        "source",
        "track",
        "trusted_goal",
        "visible_observation",
        "visible_prior_tool",
        "legal_tool_names",
        "tool_schemas",
    }
    missing = sorted(required - set(causal_model_input))
    if missing:
        raise ValueError(f"typed causal state is missing fields: {missing}")
    goal = str(causal_model_input["trusted_goal"])
    observation = str(causal_model_input["visible_observation"])
    fields = {
        "source_track": (
            f"{causal_model_input['source']} {causal_model_input['track']}"
        ),
        "trusted_goal": goal,
        "visible_observation": observation,
        "visible_prior_tool": str(causal_model_input["visible_prior_tool"]),
        "legal_interface": " ".join(
            map(str, causal_model_input["legal_tool_names"])
        ),
        "tool_schema": json.dumps(
            causal_model_input["tool_schemas"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "entity_evidence": " ".join(_ENTITY.findall(f"{goal}\n{observation}")),
    }
    rows = []
    for node_type in TYPED_STATE_NODES:
        text = fields[node_type]
        rows.append(
            np.concatenate(
                [
                    _hashed_tokens(
                        text,
                        namespace=f"fns-bwm-v4::{node_type}",
                        dimension=hash_dimension,
                    ),
                    _numeric_features(text),
                ]
            )
        )
    return np.stack(rows).astype(np.float32, copy=False)


def stack_typed_state_nodes(
    rows: Sequence[Mapping[str, Any]], *, hash_dimension: int
) -> np.ndarray:
    if not rows:
        raise ValueError("cannot encode an empty causal-state collection")
    return np.stack(
        [
            typed_state_node_features(
                row["causal_model_input"], hash_dimension=hash_dimension
            )
            for row in rows
        ]
    )


class FactorizedBeliefWorldModel(nn.Module):
    """Typed belief encoder plus action-conditioned recurrent dynamics."""

    def __init__(
        self,
        *,
        structured_state_size: int,
        node_feature_size: int,
        node_count: int,
        candidate_size: int,
        hidden_size: int,
        attention_heads: int,
        attention_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.structured_encoder = nn.Sequential(
            nn.Linear(structured_state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.node_projection = nn.Linear(node_feature_size, hidden_size)
        self.node_type_embedding = nn.Embedding(node_count, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relational_encoder = nn.TransformerEncoder(
            layer, num_layers=attention_layers
        )
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pool_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.initial_belief = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.next_candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.dynamics = nn.GRUCell(hidden_size, hidden_size)
        self.next_action_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.outcome_head = nn.Linear(hidden_size, len(OBSERVED_OUTCOME_TARGETS))

    def encode_state(self, structured_states: Tensor, typed_nodes: Tensor) -> Tensor:
        if typed_nodes.ndim != 3:
            raise ValueError("typed nodes must have [batch, nodes, features] shape")
        structured = self.structured_encoder(structured_states)
        node_ids = torch.arange(typed_nodes.shape[1], device=typed_nodes.device)
        nodes = self.node_projection(typed_nodes)
        nodes = nodes + self.node_type_embedding(node_ids)[None, :, :]
        nodes = self.relational_encoder(nodes)
        query = self.pool_query.expand(nodes.shape[0], -1, -1)
        pooled, _ = self.pool_attention(query, nodes, nodes, need_weights=False)
        return self.initial_belief(torch.cat([structured, pooled[:, 0]], dim=-1))

    def advance(self, belief: Tensor, selected_action: Tensor) -> Tensor:
        return self.dynamics(self.action_encoder(selected_action), belief)

    def score_candidates(self, belief: Tensor, candidates: Tensor) -> Tensor:
        candidate = self.next_candidate_encoder(candidates)
        expanded_belief = belief[:, None, :].expand(-1, candidate.shape[0], -1)
        expanded_candidate = candidate[None, :, :].expand(belief.shape[0], -1, -1)
        interaction = expanded_belief * expanded_candidate
        joint = torch.cat(
            [expanded_belief, expanded_candidate, interaction], dim=-1
        )
        return self.next_action_head(joint).squeeze(-1)

    def one_step(
        self,
        structured_states: Tensor,
        typed_nodes: Tensor,
        selected_actions: Tensor,
        candidates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        belief = self.encode_state(structured_states, typed_nodes)
        following = self.advance(belief, selected_actions)
        return (
            self.score_candidates(following, candidates),
            self.outcome_head(following),
            following,
        )


def masked_action_probabilities(logits: Tensor, legal_mask: Tensor) -> Tensor:
    if logits.shape != legal_mask.shape:
        raise ValueError("legal mask shape differs from action logits")
    if not bool(torch.all(legal_mask.any(dim=1))):
        raise ValueError("every state must expose at least one legal action")
    masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=1)


def assert_factorized_scope(model: FactorizedBeliefWorldModel) -> None:
    forbidden = {
        "actor",
        "critic",
        "planner",
        "reward_head",
        "utility_head",
        "value_head",
        "security_head",
        "completion_head",
    }
    present = sorted(forbidden & set(dict(model.named_modules())))
    if present:
        raise ValueError(f"forbidden heads are enabled: {present}")
