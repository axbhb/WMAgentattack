"""Relation-factorized semantic transfer utilities for the v24 effect model."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from wmagentattack.compositional_effect_world_model import parse_effect_token
from wmagentattack.pretrained_semantic_effect_world_model import (
    effect_token_description,
    normalized_action_description,
)


LABEL_CHANNELS = ("full", "category", "entity", "field", "kind_value")
ACTION_CHANNELS = ("full", "tool", "arguments")


def _words(value: str) -> str:
    return " ".join(value.replace("::", " ").replace("_", " ").split())


def effect_relation_descriptions(token: str) -> tuple[list[str], np.ndarray]:
    """Return fixed semantic channels without task, source, or outcome metadata."""

    slots = parse_effect_token(token)
    descriptions = [
        effect_token_description(token),
        f"passage: effect category {_words(slots['category'])}.",
        f"passage: affected entity {_words(slots['entity'])}." if slots["entity"] else "",
        f"passage: affected field {_words(slots['field'])}." if slots["field"] else "",
        (
            "passage: effect relation "
            + "; ".join(
                part for part in (
                    f"kind {_words(slots['kind'])}" if slots["kind"] else "",
                    f"value {_words(slots['value'])}" if slots["value"] else "",
                ) if part
            )
            + "."
        ) if slots["kind"] or slots["value"] else "",
    ]
    return descriptions, np.asarray([bool(value) for value in descriptions], dtype=bool)


def action_relation_descriptions(action: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    tool_id = str(action.get("tool_id", "unknown tool"))
    arguments = []
    for argument in action.get("arguments", []):
        item = f"field {_words(str(argument.get('field', 'unknown')))}"
        value_class = argument.get("value_class", {})
        if isinstance(value_class, Mapping):
            for key in ("type", "category", "range", "length_bucket"):
                if key in value_class:
                    item += f"; {key} {_words(str(value_class[key]))}"
        arguments.append(item)
    descriptions = [
        normalized_action_description(action),
        f"query: tool operation {_words(tool_id)}.",
        "query: action arguments; " + "; ".join(arguments) + "." if arguments else "",
    ]
    return descriptions, np.asarray([bool(value) for value in descriptions], dtype=bool)


def aggregate_channels(
    features: np.ndarray, mask: np.ndarray, weights: Sequence[float]
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    channel_weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("channel features and mask are not aligned")
    if features.shape[1] != len(channel_weights):
        raise ValueError("channel weight count is wrong")
    effective = mask * channel_weights[None, :]
    effective /= np.maximum(effective.sum(1, keepdims=True), 1e-12)
    output = (features * effective[:, :, None]).sum(1)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    return output.astype(np.float32)


def relation_kernel(
    label_channels: np.ndarray,
    label_mask: np.ndarray,
    channel_weights: Sequence[float],
    temperature: float,
) -> np.ndarray:
    """Build a fixed positive relation kernel across candidate labels."""

    if temperature <= 0:
        raise ValueError("relation-kernel temperature must be positive")
    channels = np.asarray(label_channels, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    weights = np.asarray(channel_weights, dtype=np.float64)
    if channels.ndim != 3 or mask.shape != channels.shape[:2]:
        raise ValueError("label channels and mask are not aligned")
    numerator = np.zeros((len(channels), len(channels)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for channel, weight in enumerate(weights):
        present = mask[:, channel][:, None] & mask[:, channel][None, :]
        similarity = channels[:, channel] @ channels[:, channel].T
        numerator += present * weight * similarity
        denominator += present * weight
    score = numerator / np.maximum(denominator, 1e-12)
    score = np.clip(score, -1.0, 1.0)
    kernel = np.exp((score - 1.0) / float(temperature))
    kernel[denominator == 0] = 0.0
    np.fill_diagonal(kernel, 1.0)
    return kernel.astype(np.float32)


def calibration_label_mask(vocabulary: Sequence[str], modulus: int = 5) -> np.ndarray:
    if modulus < 2:
        raise ValueError("calibration modulus must be at least two")
    return np.asarray([
        int(hashlib.sha256(f"v24-cal:{token}".encode()).hexdigest(), 16) % modulus == 0
        for token in vocabulary
    ])


def support_fused_probabilities(
    logits: np.ndarray,
    source_mask: np.ndarray,
    candidate_mask: np.ndarray,
    kernel: np.ndarray,
    support_weight: float,
    top_k: int,
) -> np.ndarray:
    """Diffuse fitted-label probabilities into candidates through fixed relations."""

    raw = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -30.0, 30.0)))
    output = raw.copy()
    sources = np.where(np.asarray(source_mask, dtype=bool))[0]
    candidates = np.where(np.asarray(candidate_mask, dtype=bool))[0]
    if not 0.0 <= support_weight <= 1.0:
        raise ValueError("support weight must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    for candidate in candidates:
        similarities = np.asarray(kernel[sources, candidate], dtype=np.float64)
        order = np.argsort(similarities)[::-1]
        chosen = order[: min(top_k, len(order))]
        selected_sources = sources[chosen]
        selected_weights = similarities[chosen]
        positive = selected_weights > 0
        selected_sources = selected_sources[positive]
        selected_weights = selected_weights[positive]
        if len(selected_sources) == 0 or support_weight == 0:
            continue
        selected_weights /= selected_weights.sum()
        support = raw[:, selected_sources] @ selected_weights
        output[:, candidate] = (
            (1.0 - support_weight) * raw[:, candidate] + support_weight * support
        )
    return np.clip(output, 1e-7, 1.0 - 1e-7)


def similarity_distribution_loss(
    logits: Tensor,
    targets: Tensor,
    fit_labels: Tensor,
    relation: Tensor,
    temperature: float,
) -> Tensor:
    """Match row predictions to semantic distributions induced by fitted positives."""

    if temperature <= 0:
        raise ValueError("distribution temperature must be positive")
    losses = []
    for row in range(logits.shape[0]):
        positives = torch.where((targets[row] > 0.5) & fit_labels)[0]
        if len(positives) == 0:
            continue
        desired = relation[positives].sum(0).clamp_min(0)
        desired = desired / desired.sum().clamp_min(1e-12)
        log_probability = F.log_softmax(logits[row] / float(temperature), dim=-1)
        losses.append(-(desired * log_probability).sum())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _selection_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = probabilities >= threshold
    truth = targets == 1
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta2 = 4.0
    f2 = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
    positive_nll = float(-np.log(probabilities[truth]).mean()) if truth.any() else float("inf")
    bce = float(-(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)).mean())
    return {
        "f2": float(f2),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "mean_predicted_set_size": float(predicted.sum(1).mean()),
        "mean_true_set_size": float(truth.sum(1).mean()),
        "positive_nll": positive_nll,
        "bce": bce,
    }


def select_support_set_rule(
    logits: np.ndarray,
    targets: np.ndarray,
    fitted_labels: np.ndarray,
    heldout_labels: np.ndarray,
    kernel: np.ndarray,
    support_weights: Sequence[float],
    thresholds: Sequence[float],
    top_k: int,
    maximum_false_positive_rate: float,
    maximum_set_size_multiplier: float,
    set_size_offset: float,
    maximum_nll_increase: float = 0.1,
) -> dict[str, float]:
    if not np.asarray(heldout_labels, dtype=bool).any():
        raise ValueError("support selection has no held-out labels")
    y = np.asarray(targets[:, heldout_labels], dtype=np.float64)
    if not (y == 1).any():
        raise ValueError("support selection has no held-out positives")
    raw = support_fused_probabilities(
        logits, fitted_labels, heldout_labels, kernel, 0.0, top_k
    )[:, heldout_labels]
    raw_nll = float(-np.log(raw[y == 1]).mean())
    candidates = []
    for weight in support_weights:
        probability = support_fused_probabilities(
            logits, fitted_labels, heldout_labels, kernel, float(weight), top_k
        )[:, heldout_labels]
        for threshold in thresholds:
            metrics = _selection_metrics(probability, y, float(threshold))
            set_limit = (
                maximum_set_size_multiplier * metrics["mean_true_set_size"] + set_size_offset
            )
            feasible = (
                metrics["false_positive_rate"] <= maximum_false_positive_rate
                and metrics["mean_predicted_set_size"] <= set_limit
                and metrics["positive_nll"] <= raw_nll + maximum_nll_increase
            )
            if feasible:
                candidates.append((
                    (-metrics["f2"], metrics["bce"], float(weight), -float(threshold)),
                    metrics,
                    set_limit,
                ))
    if not candidates:
        fallback_threshold = float(max(thresholds))
        fallback_probability = support_fused_probabilities(
            logits, fitted_labels, heldout_labels, kernel, 0.0, top_k
        )[:, heldout_labels]
        fallback_metrics = _selection_metrics(
            fallback_probability, y, fallback_threshold
        )
        return {
            "support_weight": 0.0,
            "threshold": fallback_threshold,
            "raw_positive_nll": raw_nll,
            "selection_set_limit": float(
                maximum_set_size_multiplier
                * fallback_metrics["mean_true_set_size"]
                + set_size_offset
            ),
            "feasible": False,
            **fallback_metrics,
        }
    best = min(candidates, key=lambda value: value[0])
    return {
        "support_weight": best[0][2],
        "threshold": -best[0][3],
        "raw_positive_nll": raw_nll,
        "selection_set_limit": float(best[2]),
        "feasible": True,
        **best[1],
    }
