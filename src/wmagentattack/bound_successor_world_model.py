"""Bound successor-record dynamics for the v28 clean transition probe.

The model scores whole typed evidence records and current-goal pointers.  It
never predicts independent entity/attribute atoms or a standalone matched
count.  Canonical effect probabilities are rendered from the structured
predictions so record bindings and count semantics remain explicit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .clean_evidence_probe import hashed_text


def record_signature(record: Mapping[str, Any]) -> str:
    payload = {
        "entity_type": str(record["entity_type"]),
        "link_status": str(record["link_status"]),
        "attributes": sorted(
            (
                {"name": str(attribute["name"]), "kind": str(attribute["kind"])}
                for attribute in record.get("attributes", ())
            ),
            key=lambda value: (value["name"], value["kind"]),
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_record_signature(signature: str) -> dict[str, Any]:
    return json.loads(signature)


def record_vocabulary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            record_signature(record)
            for row in rows
            for record in row["model_target"]["structured_successor_delta"]["added_evidence_records"]
        }
    )


def conflict_signature(conflict: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "attribute_name": str(conflict["attribute_name"]),
            "reason": str(conflict["reason"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def conflict_vocabulary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            conflict_signature(conflict)
            for row in rows
            for conflict in row["model_target"]["structured_successor_delta"]["added_conflicts"]
        }
    )


def candidate_features(values: Sequence[str], dimension: int, namespace: str) -> np.ndarray:
    if not values:
        return np.zeros((0, dimension), dtype=np.float32)
    return np.stack([hashed_text(value, dimension, namespace) for value in values]).astype(np.float32)


def goal_term_features(row: Mapping[str, Any], dimension: int) -> np.ndarray:
    terms = row["model_input"]["current_semantic_state"].get("goal", {}).get("fact_terms", ())
    return candidate_features([str(term) for term in terms], dimension, "v28-goal-term")


def clipped_poisson_binomial(probabilities: Tensor, maximum: int = 3) -> Tensor:
    """Distribution for counts 0..maximum-1 and a final >=maximum bin."""

    if probabilities.ndim != 1:
        raise ValueError("pointer probabilities must be one-dimensional")
    distribution = probabilities.new_zeros(maximum + 1)
    distribution[0] = 1.0
    for probability in probabilities:
        following = probabilities.new_zeros(maximum + 1)
        following[0] = distribution[0] * (1.0 - probability)
        for count in range(1, maximum):
            following[count] = (
                distribution[count] * (1.0 - probability)
                + distribution[count - 1] * probability
            )
        following[maximum] = distribution[maximum] + distribution[maximum - 1] * probability
        distribution = following
    return distribution


class BoundSuccessorRecordTransition(nn.Module):
    """Zero-start recurrent state dynamics with candidate-conditioned set heads."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        record_feature_size: int,
        pointer_feature_size: int,
        conflict_feature_size: int,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
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
        self.delta_head = nn.Linear(hidden_size, 5)
        self.record_query = nn.Linear(hidden_size, hidden_size)
        self.record_key = nn.Sequential(nn.Linear(record_feature_size, hidden_size), nn.GELU())
        self.record_bias = nn.Linear(record_feature_size, 1)
        self.pointer_query = nn.Linear(hidden_size, hidden_size)
        self.pointer_key = nn.Sequential(nn.Linear(pointer_feature_size, hidden_size), nn.GELU())
        self.pointer_bias = nn.Linear(pointer_feature_size, 1)
        self.conflict_query = nn.Linear(hidden_size, hidden_size)
        self.conflict_key = nn.Sequential(nn.Linear(conflict_feature_size, hidden_size), nn.GELU())
        self.conflict_bias = nn.Linear(conflict_feature_size, 1)
        self.scale = math.sqrt(hidden_size)

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(self, hidden: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        encoded_action = self.action_encoder(action)
        joint = torch.cat((hidden, encoded_action), dim=-1)
        execution = self.execution_head(joint).squeeze(-1)
        following = self.next_norm(hidden + self.residual(joint))
        return following, execution

    def _score(
        self,
        hidden: Tensor,
        candidates: Tensor,
        query: nn.Linear,
        key: nn.Module,
        bias: nn.Linear,
    ) -> Tensor:
        if candidates.shape[0] == 0:
            return hidden.new_zeros((hidden.shape[0], 0))
        q = query(hidden)
        k = key(candidates)
        return q @ k.T / self.scale + bias(candidates).squeeze(-1)[None, :]

    def predict_hidden(
        self,
        hidden: Tensor,
        record_features: Tensor,
        conflict_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        records = self._score(
            hidden, record_features, self.record_query, self.record_key, self.record_bias
        )
        conflicts = self._score(
            hidden, conflict_features, self.conflict_query, self.conflict_key, self.conflict_bias
        )
        return records, self.delta_head(hidden), conflicts

    def pointer_logits(self, hidden: Tensor, term_features: Tensor) -> Tensor:
        if hidden.shape[0] != 1:
            raise ValueError("pointer scoring consumes one variable-length row at a time")
        return self._score(
            hidden, term_features, self.pointer_query, self.pointer_key, self.pointer_bias
        )[0]

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        record_features: Tensor,
        conflict_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden, execution = self.advance_with_execution(self.initial_hidden(state), action)
        records, delta, conflicts = self.predict_hidden(hidden, record_features, conflict_features)
        return records, delta, conflicts, execution


def _noisy_or(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1e-7
    return float(1.0 - np.prod(1.0 - np.clip(values, 1e-7, 1.0 - 1e-7)))


def render_effect_probabilities(
    record_probabilities: np.ndarray,
    pointer_probabilities: Sequence[np.ndarray],
    delta_probabilities: np.ndarray,
    execution_error_probabilities: np.ndarray,
    conflict_probabilities: np.ndarray,
    record_candidates: Sequence[str],
    conflict_candidates: Sequence[str],
    effect_vocabulary: Sequence[str],
) -> np.ndarray:
    records = [parse_record_signature(value) for value in record_candidates]
    conflicts = [json.loads(value) for value in conflict_candidates]
    output = np.full(
        (len(record_probabilities), len(effect_vocabulary)), 1e-7, dtype=np.float64
    )
    for row_index in range(len(record_probabilities)):
        count = clipped_poisson_binomial(
            torch.tensor(pointer_probabilities[row_index], dtype=torch.float64), maximum=3
        ).numpy()
        for token_index, token in enumerate(effect_vocabulary):
            if token.startswith("delta_bit_"):
                left, raw = token.split("=", 1)
                index = int(left.removeprefix("delta_bit_"))
                probability = float(delta_probabilities[row_index, index])
                output[row_index, token_index] = probability if int(raw) else 1.0 - probability
            elif token == "execution=error":
                output[row_index, token_index] = execution_error_probabilities[row_index]
            elif token == "execution=success":
                output[row_index, token_index] = 1.0 - execution_error_probabilities[row_index]
            elif token.startswith("matched_count="):
                value = min(int(token.split("=", 1)[1]), 3)
                output[row_index, token_index] = count[value]
            elif token.startswith("entity="):
                value = token.split("=", 1)[1]
                selected = np.asarray([
                    record_probabilities[row_index, index]
                    for index, record in enumerate(records)
                    if record["entity_type"] == value
                ])
                output[row_index, token_index] = _noisy_or(selected)
            elif token.startswith("link="):
                value = token.split("=", 1)[1]
                selected = np.asarray([
                    record_probabilities[row_index, index]
                    for index, record in enumerate(records)
                    if record["link_status"] == value
                ])
                output[row_index, token_index] = _noisy_or(selected)
            elif token.startswith("attribute="):
                entity, name, kind = token.split("=", 1)[1].split("::", 2)
                selected = []
                for index, record in enumerate(records):
                    attributes = {(value["name"], value["kind"]) for value in record["attributes"]}
                    if record["entity_type"] == entity and (name, kind) in attributes:
                        selected.append(record_probabilities[row_index, index])
                output[row_index, token_index] = _noisy_or(np.asarray(selected))
            elif token.startswith("conflict="):
                value = token.split("=", 1)[1]
                selected = np.asarray([
                    conflict_probabilities[row_index, index]
                    for index, conflict in enumerate(conflicts)
                    if conflict["attribute_name"] == value
                ])
                output[row_index, token_index] = _noisy_or(selected)
    return np.clip(output, 1e-7, 1.0 - 1e-7)
