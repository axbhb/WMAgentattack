"""Open-vocabulary semantic effect scorers for the v22 transition gate.

The label encoder consumes only normalized effect-token slots.  It does not
consume task ids, source ids, final outcomes, utility, security, or planner
labels.  This lets one transition latent score labels that had no positive
example in the training fold.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


EFFECT_CATEGORIES = (
    "attribute",
    "conflict",
    "delta_bit",
    "entity",
    "execution",
    "link",
    "matched_count",
    "other",
)
_WORD = re.compile(r"[a-z0-9]+")


def parse_effect_token(token: str) -> dict[str, str]:
    """Parse a canonical effect token into reusable semantic slots."""

    left, separator, value = token.partition("=")
    if not separator:
        raise ValueError(f"effect token has no value separator: {token}")
    slots = {
        "category": left,
        "entity": "",
        "field": "",
        "kind": "",
        "value": value,
    }
    if left == "attribute":
        parts = value.split("::")
        if len(parts) != 3:
            raise ValueError(f"attribute token is not canonical: {token}")
        slots.update(entity=parts[0], field=parts[1], kind=parts[2], value="")
    elif left.startswith("delta_bit_"):
        slots.update(category="delta_bit", field=left, kind="binary")
    elif left == "entity":
        slots.update(entity=value, value="")
    elif left == "conflict":
        slots.update(field=value, kind="conflict", value="")
    elif left == "matched_count":
        slots.update(kind="count")
    elif left not in EFFECT_CATEGORIES:
        slots["category"] = "other"
        slots["field"] = left
    return slots


def _signed_hash(vector: np.ndarray, namespace: str, value: str, weight: float = 1.0) -> None:
    for word in _WORD.findall(value.lower()):
        digest = hashlib.blake2b(f"{namespace}:{word}".encode(), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % len(vector)
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * weight


def effect_token_feature_vector(token: str, hash_dimension: int = 16) -> np.ndarray:
    """Encode category plus entity/field/kind/value slots without a fixed label id."""

    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive")
    slots = parse_effect_token(token)
    category = np.zeros(len(EFFECT_CATEGORIES), dtype=np.float32)
    category[EFFECT_CATEGORIES.index(slots["category"])] = 1.0
    blocks = []
    for name in ("entity", "field", "kind", "value"):
        block = np.zeros(hash_dimension, dtype=np.float32)
        _signed_hash(block, name, slots[name])
        blocks.append(block)
    shared = np.zeros(hash_dimension * 2, dtype=np.float32)
    for name in ("entity", "field", "kind", "value"):
        _signed_hash(shared, "shared", slots[name])
    vector = np.concatenate((category, *blocks, shared))
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def effect_vocabulary_features(
    vocabulary: Sequence[str], hash_dimension: int = 16
) -> np.ndarray:
    return np.stack([
        effect_token_feature_vector(token, hash_dimension) for token in vocabulary
    ]).astype(np.float32)


def normalized_action_feature_vector(
    action: Mapping[str, Any], hash_dimension: int = 32
) -> np.ndarray:
    """A label-aligned action descriptor using tool and argument semantic slots."""

    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive")
    vector = np.zeros(hash_dimension, dtype=np.float32)
    tool_id = str(action.get("tool_id", ""))
    leaf = tool_id.split("::")[-1]
    _signed_hash(vector, "shared", leaf, 2.0)
    for argument in action.get("arguments", []):
        _signed_hash(vector, "shared", str(argument.get("field", "")), 1.5)
        value_class = argument.get("value_class", {})
        if isinstance(value_class, Mapping):
            for key in ("type", "category", "range"):
                _signed_hash(vector, "shared", str(value_class.get(key, "")), 0.5)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


class _CandidateResidualBase(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_semantic_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size + action_semantic_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
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
        self.query = nn.Linear(hidden_size, hidden_size)

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(
        self, hidden: Tensor, action: Tensor, action_semantic: Tensor
    ) -> tuple[Tensor, Tensor]:
        encoded_action = self.action_encoder(torch.cat((action, action_semantic), dim=-1))
        joint = torch.cat((hidden, encoded_action), dim=-1)
        execution_logit = self.execution_head(joint).squeeze(-1)
        return self.next_norm(hidden + self.residual(joint)), execution_logit

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        action_semantic: Tensor,
        label_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden, execution = self.advance_with_execution(
            self.initial_hidden(state), action, action_semantic
        )
        return self.predict_hidden(hidden, label_features), execution


class CompositionalEffectTransition(_CandidateResidualBase):
    """Score arbitrary effect-token descriptions against a transition latent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_semantic_size: int,
        label_feature_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__(state_size, action_size, action_semantic_size, hidden_size)
        self.label_encoder = nn.Sequential(
            nn.Linear(label_feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.label_bias = nn.Linear(label_feature_size, 1)

    def predict_hidden(self, hidden: Tensor, label_features: Tensor) -> Tensor:
        labels = self.label_encoder(label_features)
        logits = self.query(hidden) @ labels.T / math.sqrt(labels.shape[-1])
        return logits + self.label_bias(label_features).T


class IndependentLabelEffectTransition(_CandidateResidualBase):
    """Capacity control with one unrelated embedding per fixed label id."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_semantic_size: int,
        targets: int,
        hidden_size: int,
    ) -> None:
        super().__init__(state_size, action_size, action_semantic_size, hidden_size)
        self.label_embedding = nn.Embedding(targets, hidden_size)
        self.label_bias = nn.Parameter(torch.zeros(targets))

    def predict_hidden(self, hidden: Tensor, label_features: Tensor) -> Tensor:
        del label_features
        labels = self.label_embedding.weight
        return self.query(hidden) @ labels.T / math.sqrt(labels.shape[-1]) + self.label_bias

