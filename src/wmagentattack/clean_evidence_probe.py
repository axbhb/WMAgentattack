"""Frozen clean-only evidence-ledger architecture probe.

The feature builders in this module accept only trusted goals and causally
observed prefixes.  Episode utility and expert-slot coverage are supplied to
the training functions as targets and never enter a feature vector.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


FROZEN_VARIANTS = (
    "static_length",
    "state_only",
    "semantic_markov",
    "semantic_markov_state",
    "semantic_markov_state_evidence",
    "semantic_markov_state_shuffled_evidence",
    "semantic_markov_state_output_length",
    "event_transformer_state_evidence",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*")
_PURE_OUTPUT_LENGTH_KEYS = ("item_count", "character_count", "token_count")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@lru_cache(maxsize=131072)
def _hashed_text_cached(text: str, dimension: int, namespace: str) -> tuple[float, ...]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    terms = list(tokens)
    terms.extend(f"{left}__{right}" for left, right in zip(tokens, tokens[1:]))
    vector = np.zeros(dimension, dtype=np.float32)
    for term in terms:
        digest = hashlib.sha256(f"{namespace}\0{term}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "little") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return tuple(float(value) for value in vector)


def hashed_text(value: Any, dimension: int, namespace: str) -> np.ndarray:
    return np.asarray(
        _hashed_text_cached(_canonical_text(value), dimension, namespace),
        dtype=np.float32,
    )


def pure_output_lengths(prefix: Mapping[str, Any]) -> np.ndarray:
    lengths = prefix["features"]["evidence_length"]
    return np.asarray(
        [math.log1p(float(lengths.get(key, 0.0))) for key in _PURE_OUTPUT_LENGTH_KEYS],
        dtype=np.float32,
    )


def _prefix_length(prefix: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [math.log1p(float(prefix["features"]["prefix_length"]))],
        dtype=np.float32,
    )


def _state_numeric(prefix: Mapping[str, Any]) -> np.ndarray:
    summary = prefix["features"]["state_summary"]
    return np.asarray(
        [
            float(bool(summary.get("last_state_changed", False))),
            math.log1p(float(summary.get("cumulative_state_changes", 0.0))),
            math.log1p(float(summary.get("cumulative_errors", 0.0))),
        ],
        dtype=np.float32,
    )


def vector_features(
    prefix: Mapping[str, Any],
    *,
    variant: str,
    hash_dimension: int,
    evidence_override: str | None = None,
) -> np.ndarray:
    """Build one fixed-dimensional feature vector for a non-transformer arm."""

    if variant not in FROZEN_VARIANTS or variant == "event_transformer_state_evidence":
        raise ValueError(f"not a vector variant: {variant}")
    features = prefix["features"]
    goal = hashed_text(features["trusted_goal"], hash_dimension, "goal")
    event = hashed_text(features["last_event"], hash_dimension, "event")
    state = np.concatenate(
        (
            hashed_text(features["canonical_state"], hash_dimension, "state"),
            hashed_text(features["state_summary"], hash_dimension, "state-summary"),
            _state_numeric(prefix),
        )
    )
    evidence_text = (
        str(evidence_override)
        if evidence_override is not None
        else str(features["evidence_text"])
    )
    evidence = hashed_text(evidence_text, hash_dimension, "evidence")
    length = _prefix_length(prefix)
    output_length = pure_output_lengths(prefix)

    if variant == "static_length":
        parts = (goal, length, output_length)
    elif variant == "state_only":
        parts = (goal, state, length)
    elif variant == "semantic_markov":
        parts = (goal, event, length)
    elif variant == "semantic_markov_state":
        parts = (goal, event, state, length)
    elif variant == "semantic_markov_state_output_length":
        parts = (goal, event, state, length, output_length)
    else:
        parts = (
            goal,
            event,
            state,
            evidence,
            goal * evidence,
            length,
            output_length,
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def transformer_step_features(
    prefix: Mapping[str, Any], *, hash_dimension: int
) -> np.ndarray:
    """Encode one causal prefix state for the full-history transformer arm."""

    features = prefix["features"]
    goal = hashed_text(features["trusted_goal"], hash_dimension, "goal")
    event = hashed_text(features["last_event"], hash_dimension, "event")
    state = hashed_text(features["canonical_state"], hash_dimension, "state")
    state_summary = hashed_text(
        features["state_summary"], hash_dimension, "state-summary"
    )
    evidence = hashed_text(features["evidence_text"], hash_dimension, "evidence")
    new_evidence = hashed_text(
        features.get("new_evidence_text", ""), hash_dimension, "new-evidence"
    )
    return np.concatenate(
        (
            goal,
            event,
            state,
            state_summary,
            evidence,
            new_evidence,
            goal * evidence,
            _state_numeric(prefix),
            _prefix_length(prefix),
            pure_output_lengths(prefix),
        )
    ).astype(np.float32, copy=False)


def build_within_task_cyclic_donors(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode["task_id"])].append(str(episode["episode_id"]))
    donors: dict[str, str] = {}
    for task_id, episode_ids in sorted(grouped.items()):
        ordered = sorted(episode_ids)
        if len(ordered) < 2:
            raise ValueError(f"within-task shuffle has no donor for {task_id}")
        for index, episode_id in enumerate(ordered):
            donors[episode_id] = ordered[(index + 1) % len(ordered)]
    return donors


def task_balanced_weights(task_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(str(task_id) for task_id in task_ids)
    if not counts:
        raise ValueError("cannot weight an empty sample")
    scale = len(task_ids) / len(counts)
    return np.asarray(
        [scale / counts[str(task_id)] for task_id in task_ids], dtype=np.float32
    )


class VectorEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        del mask
        return self.network(inputs)


class EventTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        layers: int,
        heads: int,
        max_length: int,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by transformer heads")
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            raise ValueError("transformer inputs require a valid-token mask")
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        encoded = self.input_projection(inputs) + self.position(positions)
        encoded = self.transformer(encoded, src_key_padding_mask=~mask.bool())
        last = mask.long().sum(dim=1).sub(1).clamp_min(0)
        pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
        return self.normalization(pooled)


class CleanEvidenceProbe(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.progress_head = nn.Linear(hidden_size, 1)
        self.utility_head = nn.Linear(hidden_size, 1)

    def encode(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.encoder(inputs, mask)

    def progress_logits(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.progress_head(self.encode(inputs, mask)).squeeze(-1)

    def utility_logits_from_encoded(self, encoded: Tensor) -> Tensor:
        return self.utility_head(encoded).squeeze(-1)


def _batches(
    size: int, batch_size: int, *, seed: int, shuffle: bool
) -> Iterable[np.ndarray]:
    indices = np.arange(size)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def _as_device(values: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def fit_progress_then_utility(
    model: CleanEvidenceProbe,
    *,
    inputs: np.ndarray,
    masks: np.ndarray | None,
    progress_targets: np.ndarray,
    task_ids: Sequence[str],
    final_indices: np.ndarray,
    utility_targets: np.ndarray,
    progress_epochs: int,
    utility_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, float]:
    """Fit progress first, then a utility head with a frozen encoder."""

    set_deterministic_seed(seed)
    target_device = torch.device(device)
    model.to(target_device)
    progress_weights = task_balanced_weights(task_ids)
    optimizer = torch.optim.AdamW(
        [*model.encoder.parameters(), *model.progress_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    final_progress_loss = math.nan
    model.train()
    for epoch in range(progress_epochs):
        for batch in _batches(
            len(inputs), batch_size, seed=seed * 1009 + epoch, shuffle=True
        ):
            x = _as_device(inputs[batch], target_device)
            mask = (
                torch.as_tensor(masks[batch], dtype=torch.bool, device=target_device)
                if masks is not None
                else None
            )
            target = _as_device(progress_targets[batch], target_device)
            weight = _as_device(progress_weights[batch], target_device)
            prediction = torch.sigmoid(model.progress_logits(x, mask))
            loss = ((prediction - target).square() * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_progress_loss = float(loss.detach().cpu())

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.progress_head.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.progress_head.eval()

    final_task_ids = [str(task_ids[index]) for index in final_indices]
    utility_weights = task_balanced_weights(final_task_ids)
    utility_targets = np.asarray(utility_targets, dtype=np.float32)
    prevalence = float(np.clip(utility_targets.mean(), 1e-4, 1.0 - 1e-4))
    with torch.no_grad():
        model.utility_head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))
    utility_optimizer = torch.optim.AdamW(
        model.utility_head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    final_utility_loss = math.nan
    model.utility_head.train()
    for epoch in range(utility_epochs):
        for local_batch in _batches(
            len(final_indices),
            batch_size,
            seed=seed * 2027 + epoch,
            shuffle=True,
        ):
            global_batch = final_indices[local_batch]
            x = _as_device(inputs[global_batch], target_device)
            mask = (
                torch.as_tensor(
                    masks[global_batch], dtype=torch.bool, device=target_device
                )
                if masks is not None
                else None
            )
            with torch.no_grad():
                encoded = model.encode(x, mask)
            target = _as_device(utility_targets[local_batch], target_device)
            weight = _as_device(utility_weights[local_batch], target_device)
            logits = model.utility_logits_from_encoded(encoded)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (losses * weight).sum() / weight.sum()
            utility_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            utility_optimizer.step()
            final_utility_loss = float(loss.detach().cpu())

    return {
        "final_progress_training_loss": final_progress_loss,
        "final_utility_training_loss": final_utility_loss,
        "utility_training_prevalence": prevalence,
    }


@torch.no_grad()
def predict_probe(
    model: CleanEvidenceProbe,
    *,
    inputs: np.ndarray,
    masks: np.ndarray | None,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    target_device = torch.device(device)
    progress_rows = []
    utility_rows = []
    for batch in _batches(len(inputs), batch_size, seed=0, shuffle=False):
        x = _as_device(inputs[batch], target_device)
        mask = (
            torch.as_tensor(masks[batch], dtype=torch.bool, device=target_device)
            if masks is not None
            else None
        )
        encoded = model.encode(x, mask)
        progress_rows.append(torch.sigmoid(model.progress_head(encoded)).squeeze(-1).cpu())
        utility_rows.append(torch.sigmoid(model.utility_head(encoded)).squeeze(-1).cpu())
    return (
        torch.cat(progress_rows).numpy(),
        torch.cat(utility_rows).numpy(),
    )


def task_macro_errors(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    task_rows = {}
    for task_id, task_samples in sorted(by_task.items()):
        progress_errors = [
            abs(float(row["progress_prediction"]) - float(row["progress_target"]))
            for row in task_samples
        ]
        utility_samples = [row for row in task_samples if row["is_final_prefix"]]
        brier = [
            (float(row["utility_probability"]) - float(row["utility_target"])) ** 2
            for row in utility_samples
        ]
        log_losses = []
        for row in utility_samples:
            probability = min(max(float(row["utility_probability"]), 1e-7), 1 - 1e-7)
            target = float(row["utility_target"])
            log_losses.append(
                -(target * math.log(probability) + (1 - target) * math.log(1 - probability))
            )
        task_rows[task_id] = {
            "progress_mae": float(np.mean(progress_errors)),
            "utility_brier": float(np.mean(brier)),
            "utility_log_loss": float(np.mean(log_losses)),
            "prefixes": len(task_samples),
            "episodes": len(utility_samples),
        }
    return {
        "task_macro_progress_mae": float(
            np.mean([row["progress_mae"] for row in task_rows.values()])
        ),
        "task_macro_utility_brier": float(
            np.mean([row["utility_brier"] for row in task_rows.values()])
        ),
        "task_macro_utility_log_loss": float(
            np.mean([row["utility_log_loss"] for row in task_rows.values()])
        ),
        "task_metrics": task_rows,
    }
