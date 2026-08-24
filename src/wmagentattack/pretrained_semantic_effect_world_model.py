"""Frozen-pretrained semantic effect prototypes for the v23 hybrid world model."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wmagentattack.compositional_effect_world_model import parse_effect_token


def _words(value: str) -> str:
    return " ".join(value.replace("::", " ").replace("_", " ").split())


def effect_token_description(token: str) -> str:
    slots = parse_effect_token(token)
    parts = [f"effect type {_words(slots['category'])}"]
    for name in ("entity", "field", "kind", "value"):
        if slots[name]:
            parts.append(f"{name} {_words(slots[name])}")
    return "passage: " + "; ".join(parts) + "."


def normalized_action_description(action: Mapping[str, Any]) -> str:
    tool_id = str(action.get("tool_id", "unknown tool"))
    parts = [f"tool {_words(tool_id)}"]
    for argument in action.get("arguments", []):
        text = f"argument {_words(str(argument.get('field', 'unknown')))}"
        value_class = argument.get("value_class", {})
        if isinstance(value_class, Mapping):
            for key in ("type", "category", "range", "length_bucket"):
                if key in value_class:
                    text += f" {key} {_words(str(value_class[key]))}"
        text += f" observed in state {bool(argument.get('exact_value_observed_in_state', False))}"
        parts.append(text)
    return "query: action consequence; " + "; ".join(parts) + "."


def calibration_label_mask(vocabulary: Sequence[str], modulus: int = 5) -> np.ndarray:
    if modulus < 2:
        raise ValueError("calibration modulus must be at least two")
    return np.asarray([
        int(hashlib.sha256(f"v23-cal:{token}".encode()).hexdigest(), 16) % modulus == 0
        for token in vocabulary
    ])


class PretrainedSemanticEffectTransition(nn.Module):
    """Map a structured transition latent into a frozen E5 prototype space."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        semantic_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if semantic_size != hidden_size:
            raise ValueError("v23 uses one shared semantic/hidden dimension")
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.action_encoder = nn.Linear(action_size + semantic_size, hidden_size)
        with torch.no_grad():
            self.action_encoder.weight.zero_()
            self.action_encoder.bias.zero_()
            self.action_encoder.weight[:, action_size:] = torch.eye(hidden_size)
        self.action_norm = nn.LayerNorm(hidden_size)
        self.residual = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.next_norm = nn.LayerNorm(hidden_size)
        self.execution_head = nn.Linear(hidden_size * 2, 1)
        self.query = nn.Linear(hidden_size, semantic_size)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.global_bias = nn.Parameter(torch.tensor(-2.0))

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(
        self, hidden: Tensor, action: Tensor, action_semantic: Tensor
    ) -> tuple[Tensor, Tensor]:
        encoded_action = self.action_norm(
            self.action_encoder(torch.cat((action, action_semantic), dim=-1))
        )
        joint = torch.cat((hidden, encoded_action), dim=-1)
        execution = self.execution_head(joint).squeeze(-1)
        return self.next_norm(hidden + self.residual(joint)), execution

    def predict_hidden(self, hidden: Tensor, label_features: Tensor) -> Tensor:
        query = F.normalize(self.query(hidden), dim=-1)
        labels = F.normalize(label_features, dim=-1)
        scale = self.log_scale.exp().clamp(1.0, 50.0)
        return scale * (query @ labels.T) + self.global_bias

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


def semantic_hard_negative_loss(
    logits: Tensor,
    targets: Tensor,
    fit_labels: Tensor,
    label_features: Tensor,
    negatives_per_positive: int,
    margin: float,
) -> Tensor:
    """Rank positives above nearest semantic negatives among fitted labels."""
    similarities = F.normalize(label_features, dim=-1) @ F.normalize(label_features, dim=-1).T
    losses = []
    for row in range(logits.shape[0]):
        positives = torch.where((targets[row] > 0.5) & fit_labels)[0]
        negatives = torch.where((targets[row] < 0.5) & fit_labels)[0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        for positive in positives:
            order = torch.argsort(similarities[positive, negatives], descending=True)
            chosen = negatives[order[:negatives_per_positive]]
            losses.append(F.softplus(margin - logits[row, positive] + logits[row, chosen]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def apply_unseen_calibration(
    logits: np.ndarray, unseen_mask: np.ndarray, temperature: float, bias: float
) -> np.ndarray:
    output = np.asarray(logits, dtype=np.float64).copy()
    output[:, unseen_mask] = output[:, unseen_mask] / temperature + bias
    return output


def select_unseen_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    heldout_labels: np.ndarray,
    temperatures: Sequence[float],
    biases: Sequence[float],
) -> dict[str, float]:
    if not heldout_labels.any():
        raise ValueError("calibration has no held-out semantic labels")
    y = targets[:, heldout_labels]
    if not (y == 1).any():
        raise ValueError("calibration labels have no positive support")
    best = None
    for temperature in temperatures:
        for bias in biases:
            z = logits[:, heldout_labels] / float(temperature) + float(bias)
            probability = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
            bce = float(-(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)).mean())
            positive_nll = float(-np.log(probability[y == 1]).mean())
            objective = bce + 0.25 * positive_nll
            candidate = (objective, float(temperature), float(bias), bce, positive_nll)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "objective": best[0],
        "temperature": best[1],
        "bias": best[2],
        "bce": best[3],
        "positive_nll": best[4],
        "positive_occurrences": int((y == 1).sum()),
        "heldout_labels": int(heldout_labels.sum()),
    }
