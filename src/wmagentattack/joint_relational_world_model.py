"""Joint record--goal relational successor dynamics for v30."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class JointRelationalSuccessorTransition(nn.Module):
    """Zero-start dynamics with action-conditioned records and record--goal edges."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        record_feature_size: int,
        goal_feature_size: int,
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
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.next_norm = nn.LayerNorm(hidden_size)
        self.execution_head = nn.Linear(hidden_size * 2, 1)
        self.delta_head = nn.Linear(hidden_size, 5)

        self.record_query = nn.Linear(hidden_size, hidden_size)
        self.record_key = nn.Sequential(nn.Linear(record_feature_size, hidden_size), nn.GELU())
        self.record_bias = nn.Linear(record_feature_size, 1)

        self.relation_hidden = nn.Linear(hidden_size, hidden_size)
        self.relation_record = nn.Linear(record_feature_size, hidden_size)
        self.relation_goal = nn.Linear(goal_feature_size, hidden_size)
        self.relation_bias = nn.Linear(goal_feature_size, 1)

        self.conflict_query = nn.Linear(hidden_size, hidden_size)
        self.conflict_key = nn.Sequential(nn.Linear(conflict_feature_size, hidden_size), nn.GELU())
        self.conflict_bias = nn.Linear(conflict_feature_size, 1)
        self.scale = math.sqrt(hidden_size)

    def initial_hidden(self, state: Tensor) -> Tensor:
        return self.state_encoder(state)

    def advance_with_execution(self, hidden: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        encoded_action = self.action_encoder(action)
        joint = torch.cat((hidden, encoded_action), dim=-1)
        following = self.next_norm(hidden + self.residual(joint))
        return following, self.execution_head(joint).squeeze(-1)

    def _score(
        self, hidden: Tensor, candidates: Tensor, query: nn.Linear, key: nn.Module, bias: nn.Linear
    ) -> Tensor:
        if candidates.shape[0] == 0:
            return hidden.new_zeros((hidden.shape[0], 0))
        return query(hidden) @ key(candidates).T / self.scale + bias(candidates).squeeze(-1)[None, :]

    def record_logits(self, hidden: Tensor, record_features: Tensor) -> Tensor:
        return self._score(hidden, record_features, self.record_query, self.record_key, self.record_bias)

    def conflict_logits(self, hidden: Tensor, conflict_features: Tensor) -> Tensor:
        return self._score(
            hidden, conflict_features, self.conflict_query, self.conflict_key, self.conflict_bias
        )

    def relation_logits(
        self, hidden: Tensor, record_features: Tensor, goal_features: Tensor
    ) -> Tensor:
        """Return [records, goal terms] logits for one successor hidden state."""
        if hidden.shape[0] != 1:
            raise ValueError("relation scoring consumes one variable-length row at a time")
        if record_features.shape[0] == 0 or goal_features.shape[0] == 0:
            return hidden.new_zeros((record_features.shape[0], goal_features.shape[0]))
        context = torch.tanh(
            self.relation_hidden(hidden)[0][None, :] + self.relation_record(record_features)
        )
        goals = self.relation_goal(goal_features)
        return context @ goals.T / self.scale + self.relation_bias(goal_features).squeeze(-1)[None, :]

    def predict_hidden(
        self, hidden: Tensor, record_features: Tensor, conflict_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.record_logits(hidden, record_features), self.delta_head(hidden), self.conflict_logits(hidden, conflict_features)

    def forward(
        self, state: Tensor, action: Tensor, record_features: Tensor, conflict_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hidden, execution = self.advance_with_execution(self.initial_hidden(state), action)
        records, delta, conflicts = self.predict_hidden(hidden, record_features, conflict_features)
        return hidden, records, delta, conflicts, execution


def global_pointer_probabilities(record_logits: Tensor, relation_logits: Tensor) -> Tensor:
    """Noisy-OR over joint record-presence and record--goal edge probabilities."""
    if relation_logits.shape[0] != record_logits.shape[0]:
        raise ValueError("record/relation candidate mismatch")
    if relation_logits.shape[1] == 0:
        return record_logits.new_zeros((0,))
    joint = torch.sigmoid(record_logits)[:, None] * torch.sigmoid(relation_logits)
    return 1.0 - torch.prod(1.0 - joint.clamp(1e-7, 1.0 - 1e-7), dim=0)
