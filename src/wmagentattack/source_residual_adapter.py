"""Shared action backbone with small source-specific residual adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


FROZEN_SOURCES = ("agentdojo", "injecagent", "tool_sandbox")


class _ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        return self.up(self.activation(self.down(value)))


class SourceResidualActionModel(nn.Module):
    """Use a shared encoder while isolating source-local residual structure."""

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        hidden_size: int,
        bottleneck_size: int,
        source_count: int,
        residual_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        self.residual_scale = float(residual_scale)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.state_adapters = nn.ModuleList(
            [_ResidualAdapter(hidden_size, bottleneck_size) for _ in range(source_count)]
        )
        self.candidate_adapters = nn.ModuleList(
            [_ResidualAdapter(hidden_size, bottleneck_size) for _ in range(source_count)]
        )
        self.action_head = nn.Linear(hidden_size, 1)

    @staticmethod
    def _apply_adapters(
        encoded: Tensor, source_indices: Tensor, adapters: nn.ModuleList, scale: float
    ) -> Tensor:
        if encoded.shape[0] != source_indices.shape[0]:
            raise ValueError("source index count differs from encoded rows")
        output = encoded.clone()
        for source_index, adapter in enumerate(adapters):
            mask = source_indices == source_index
            if bool(mask.any()):
                output[mask] = encoded[mask] + scale * adapter(encoded[mask])
        return output

    def forward(
        self,
        states: Tensor,
        candidates: Tensor,
        row_source_indices: Tensor,
        candidate_source_indices: Tensor,
    ) -> Tensor:
        state = self._apply_adapters(
            self.state_encoder(states),
            row_source_indices,
            self.state_adapters,
            self.residual_scale,
        )
        candidate = self._apply_adapters(
            self.candidate_encoder(candidates),
            candidate_source_indices,
            self.candidate_adapters,
            self.residual_scale,
        )
        joint = torch.tanh(state[:, None, :] + candidate[None, :, :])
        return self.action_head(joint).squeeze(-1)

    def action_probabilities(
        self,
        states: Tensor,
        candidates: Tensor,
        row_source_indices: Tensor,
        candidate_source_indices: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        logits = self(
            states, candidates, row_source_indices, candidate_source_indices
        )
        if logits.shape != legal_mask.shape:
            raise ValueError("legal mask shape differs from logits")
        if not bool(torch.all(legal_mask.any(dim=1))):
            raise ValueError("every state must expose at least one legal action")
        masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=1)


def source_indices(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    source_index = {name: index for index, name in enumerate(FROZEN_SOURCES)}
    row_values = np.asarray(
        [source_index[str(row["source"])] for row in rows], dtype=np.int64
    )
    candidate_values = np.asarray(
        [source_index[str(catalog[candidate]["source"])] for candidate in candidates],
        dtype=np.int64,
    )
    return row_values, candidate_values


class SourceSpecificHeadActionModel(nn.Module):
    """Shared representation with only the final action scorer source-local."""

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        hidden_size: int,
        source_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.source_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(source_count)]
        )

    def forward(
        self, states: Tensor, candidates: Tensor, row_source_indices: Tensor
    ) -> Tensor:
        state = self.state_encoder(states)
        candidate = self.candidate_encoder(candidates)
        joint = torch.tanh(state[:, None, :] + candidate[None, :, :])
        logits = torch.empty(
            joint.shape[:2], dtype=joint.dtype, device=joint.device
        )
        for source_index, head in enumerate(self.source_heads):
            mask = row_source_indices == source_index
            if bool(mask.any()):
                logits[mask] = head(joint[mask]).squeeze(-1)
        return logits

    def action_probabilities(
        self,
        states: Tensor,
        candidates: Tensor,
        row_source_indices: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        logits = self(states, candidates, row_source_indices)
        if logits.shape != legal_mask.shape:
            raise ValueError("legal mask shape differs from logits")
        masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=1)
