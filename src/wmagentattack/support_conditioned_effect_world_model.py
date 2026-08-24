"""Support-conditioned compositional effect decoder for the v26 world model.

The module predicts reusable effect atoms and renders canonical effect tokens
from those atoms.  Seen canonical labels remain the responsibility of the
frozen v21 head; this module is used only for labels without positive support
in a task-disjoint training fold.  Scalar matched-count effects use a separate
cumulative-link ordinal head instead of textual or atom similarity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compositional_effect_world_model import parse_effect_token


ATOM_SLOTS = ("category", "entity", "field", "kind", "value")


def effect_slot_atoms(token: str) -> tuple[str, ...]:
    slots = parse_effect_token(token)
    return tuple(sorted({f"{slot}::{slots[slot]}" for slot in ATOM_SLOTS if slots[slot]}))


def atom_vocabulary(effect_vocabulary: Sequence[str], extra_atoms: Sequence[str] = ()) -> list[str]:
    return sorted({atom for token in effect_vocabulary for atom in effect_slot_atoms(token)} | set(extra_atoms))


def atom_target_matrix(
    effect_rows: Sequence[Sequence[str]], vocabulary: Sequence[str]
) -> np.ndarray:
    lookup = {atom: index for index, atom in enumerate(vocabulary)}
    target = np.zeros((len(effect_rows), len(vocabulary)), dtype=np.float32)
    for row_index, tokens in enumerate(effect_rows):
        for token in tokens:
            for atom in effect_slot_atoms(token):
                if atom in lookup:
                    target[row_index, lookup[atom]] = 1.0
    return target


def support_atom_target_matrix(
    support_rows: Sequence[Mapping[str, object]], vocabulary: Sequence[str]
) -> np.ndarray:
    """Read only the explicit model_target atom field, never audit_only."""

    lookup = {atom: index for index, atom in enumerate(vocabulary)}
    target = np.zeros((len(support_rows), len(vocabulary)), dtype=np.float32)
    for row_index, row in enumerate(support_rows):
        model_target = row["model_target"]
        if not isinstance(model_target, Mapping):
            raise ValueError("support model_target must be a mapping")
        atoms = model_target["effect_slot_atoms"]
        if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)):
            raise ValueError("support effect_slot_atoms must be a sequence")
        for atom in atoms:
            if str(atom) not in lookup:
                raise ValueError(f"support atom absent from frozen vocabulary: {atom}")
            target[row_index, lookup[str(atom)]] = 1.0
    return target


def matched_count_targets(effect_rows: Sequence[Sequence[str]]) -> np.ndarray:
    values = []
    for tokens in effect_rows:
        matches = [int(token.split("=", 1)[1]) for token in tokens if token.startswith("matched_count=")]
        if len(matches) != 1:
            raise ValueError("every hard transition requires exactly one matched_count token")
        value = matches[0]
        if value < 0 or value > 3:
            raise ValueError(f"v26 ordinal support is frozen to counts 0..3, got {value}")
        values.append(value)
    return np.asarray(values, dtype=np.int64)


class SupportConditionedEffectTransition(nn.Module):
    """Zero-start residual dynamics with atom and cumulative ordinal heads."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int, atoms: int) -> None:
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
        self.atom_head = nn.Linear(hidden_size, atoms)
        self.ordinal_score = nn.Linear(hidden_size, 1)
        self.ordinal_base = nn.Parameter(torch.tensor(-0.5))
        self.ordinal_raw_gaps = nn.Parameter(torch.tensor([0.0, 0.0]))

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(self, hidden: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        encoded_action = self.action_encoder(action)
        joint = torch.cat((hidden, encoded_action), dim=-1)
        execution = self.execution_head(joint).squeeze(-1)
        following = self.next_norm(hidden + self.residual(joint))
        return following, execution

    def ordinal_thresholds(self) -> Tensor:
        gaps = F.softplus(self.ordinal_raw_gaps) + 1e-4
        return torch.cat((self.ordinal_base[None], self.ordinal_base[None] + torch.cumsum(gaps, dim=0)))

    def ordinal_probabilities(self, hidden: Tensor) -> Tensor:
        score = self.ordinal_score(hidden)
        greater = torch.sigmoid(score - self.ordinal_thresholds()[None, :])
        probabilities = torch.stack(
            (
                1.0 - greater[:, 0],
                greater[:, 0] - greater[:, 1],
                greater[:, 1] - greater[:, 2],
                greater[:, 2],
            ),
            dim=-1,
        )
        return probabilities.clamp_min(1e-7)

    def predict_hidden(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.atom_head(hidden), self.ordinal_probabilities(hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden, execution = self.advance_with_execution(self.initial_hidden(state), action)
        atoms, counts = self.predict_hidden(hidden)
        return atoms, counts, execution


def compose_effect_probabilities(
    atom_probabilities: np.ndarray,
    count_probabilities: np.ndarray,
    effect_vocabulary: Sequence[str],
    atom_vocabulary_values: Sequence[str],
) -> np.ndarray:
    """Render exact effect probabilities with a geometric fuzzy-AND."""

    atom_lookup = {atom: index for index, atom in enumerate(atom_vocabulary_values)}
    output = np.zeros((len(atom_probabilities), len(effect_vocabulary)), dtype=np.float64)
    clipped = np.clip(atom_probabilities, 1e-7, 1.0)
    for token_index, token in enumerate(effect_vocabulary):
        if token.startswith("matched_count="):
            count = int(token.split("=", 1)[1])
            if 0 <= count < count_probabilities.shape[1]:
                output[:, token_index] = count_probabilities[:, count]
            continue
        required = [atom_lookup[atom] for atom in effect_slot_atoms(token) if atom in atom_lookup]
        if required:
            output[:, token_index] = np.exp(np.log(clipped[:, required]).mean(axis=1))
    return np.clip(output, 1e-7, 1.0 - 1e-7)


def ordinal_nll(probabilities: Tensor, targets: Tensor) -> Tensor:
    return -torch.log(probabilities[torch.arange(len(targets)), targets]).mean()
