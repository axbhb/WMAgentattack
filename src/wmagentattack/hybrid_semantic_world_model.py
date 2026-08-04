"""Hybrid semantic world model for causal AgentDojo trajectory modelling.

The deterministic component validates and applies an actually observed
AgentDojo transition.  The learned component predicts the victim's next tool
choice/argument-key set and candidate-conditional evidence deltas.  It does
not reconstruct semantic state, predict reward/utility, or contain a planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .clean_evidence_probe import hashed_text
from .semantic_state_v3 import (
    StructuredSemanticStateV3,
    find_semantic_state_v3_leakage,
)


EVIDENCE_DELTA_TARGETS = (
    "record_added",
    "goal_term_newly_matched",
    "conflict_added",
    "execution_error",
    "ambiguous_or_unlinked_added",
)


def _as_state(
    value: StructuredSemanticStateV3 | Mapping[str, Any],
) -> StructuredSemanticStateV3:
    if isinstance(value, StructuredSemanticStateV3):
        return value
    leakage = find_semantic_state_v3_leakage(value)
    if leakage:
        raise ValueError(f"semantic-state leakage: {list(leakage)}")
    return StructuredSemanticStateV3.model_validate(value)


def _log_count(value: int | float) -> float:
    return math.log1p(float(value))


def semantic_state_v3_feature_vector(
    value: StructuredSemanticStateV3 | Mapping[str, Any],
    *,
    hash_dimension: int,
) -> np.ndarray:
    """Encode only the frozen causal v3 fields with field-specific hashes."""

    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive")
    state = _as_state(value)
    goal = state.goal.model_dump(mode="json")
    execution = state.execution.model_dump(mode="json")
    records = [row.model_dump(mode="json") for row in state.evidence_records]
    goal_linked_records = [
        {
            "entity_type": row.entity_type,
            "entity_key": row.entity_key,
            "attributes": [attribute.name for attribute in row.attributes],
            "matched_goal_terms": row.matched_goal_terms,
        }
        for row in state.evidence_records
    ]
    parts = [
        hashed_text(goal, hash_dimension, "semantic-v3-goal"),
        hashed_text(state.legal_actions, hash_dimension, "semantic-v3-legal"),
        hashed_text(
            state.execution.last_action,
            hash_dimension,
            "semantic-v3-last-action",
        ),
        hashed_text(execution, hash_dimension, "semantic-v3-execution"),
        hashed_text(records, hash_dimension, "semantic-v3-evidence-records"),
        hashed_text(
            goal_linked_records,
            hash_dimension,
            "semantic-v3-goal-linked-evidence",
        ),
        hashed_text(
            [row.model_dump(mode="json") for row in state.conflicts],
            hash_dimension,
            "semantic-v3-conflicts",
        ),
    ]
    summary = state.goal_evidence
    numeric = np.asarray(
        [
            _log_count(state.step_index),
            _log_count(len(state.legal_actions)),
            _log_count(len(state.execution.history)),
            _log_count(state.execution.cumulative_errors),
            _log_count(state.execution.consecutive_errors),
            _log_count(state.execution.repeated_last_tool_count),
            _log_count(len(state.evidence_records)),
            _log_count(summary.unique_entity_records),
            _log_count(summary.ambiguous_entity_records),
            _log_count(summary.unlinked_entity_records),
            _log_count(summary.conflict_count),
            _log_count(len(summary.matched_fact_terms)),
            _log_count(len(summary.unmatched_fact_terms)),
            float(state.goal.has_condition),
            float(state.goal.has_comparison),
            float(state.goal.requires_set_coverage),
            float(state.goal.requires_uniqueness),
            float(state.execution.last_status == "error"),
        ],
        dtype=np.float32,
    )
    return np.concatenate((*parts, numeric)).astype(np.float32, copy=False)


def semantic_state_v3_feature_size(hash_dimension: int) -> int:
    return 7 * hash_dimension + 18


def tool_candidate_vector(
    descriptor: Mapping[str, Any] | str,
    *,
    hash_dimension: int,
) -> np.ndarray:
    return hashed_text(descriptor, hash_dimension, "semantic-v3-tool-candidate")


def _leaf_action(action_id: str) -> str:
    return action_id if action_id == "STOP" else action_id.rsplit("::", 1)[-1]


@dataclass(frozen=True)
class ExactTransitionAudit:
    from_step: int
    to_step: int
    executed_action: str
    records_added: int
    conflicts_added: int
    matched_goal_terms_added: int


class ExactObservedSemanticTransition:
    """Validate a replayed observed transition without learning state updates.

    This component deliberately requires the sandbox-produced next state.  It
    therefore makes no claim that tool outputs can be imagined without running
    AgentDojo.  Its role is to keep deterministic bookkeeping exact while the
    two stochastic heads model victim and evidence dynamics.
    """

    def advance(
        self,
        current_value: StructuredSemanticStateV3 | Mapping[str, Any],
        next_value: StructuredSemanticStateV3 | Mapping[str, Any],
        *,
        executed_action_id: str,
    ) -> tuple[StructuredSemanticStateV3, ExactTransitionAudit]:
        current = _as_state(current_value)
        following = _as_state(next_value)
        if following.step_index != current.step_index + 1:
            raise ValueError("exact transition must advance exactly one prefix")
        if following.goal != current.goal:
            raise ValueError("trusted goal changed across an exact transition")
        if following.policy_track != current.policy_track:
            raise ValueError("policy track changed across an exact transition")
        if following.legal_actions != current.legal_actions:
            raise ValueError("legal action interface changed across a panel episode")

        old_history = current.execution.history
        new_history = following.execution.history
        if new_history[: len(old_history)] != old_history:
            raise ValueError("observed execution history was rewritten")
        appended = new_history[len(old_history) :]
        if len(appended) != 1:
            raise ValueError("exact transition must append one execution receipt")
        event = appended[0]
        expected_leaf = _leaf_action(str(executed_action_id))
        if event.call_index != current.step_index:
            raise ValueError("appended receipt has the wrong call index")
        if event.tool_name != expected_leaf:
            raise ValueError("appended receipt does not match the executed action")
        if str(following.execution.last_action.get("function")) != expected_leaf:
            raise ValueError("next state last action does not match the executed action")
        if following.execution.last_status != event.execution_status:
            raise ValueError("next execution status disagrees with its receipt")

        old_records = current.evidence_records
        new_records = following.evidence_records
        if new_records[: len(old_records)] != old_records:
            raise ValueError("entity evidence history was rewritten")
        old_conflicts = current.conflicts
        new_conflicts = following.conflicts
        if new_conflicts[: len(old_conflicts)] != old_conflicts:
            raise ValueError("evidence conflict history was rewritten")
        old_matched = set(current.goal_evidence.matched_fact_terms)
        new_matched = set(following.goal_evidence.matched_fact_terms)
        if not old_matched <= new_matched:
            raise ValueError("matched goal evidence regressed")
        return following, ExactTransitionAudit(
            from_step=current.step_index,
            to_step=following.step_index,
            executed_action=str(executed_action_id),
            records_added=len(new_records) - len(old_records),
            conflicts_added=len(new_conflicts) - len(old_conflicts),
            matched_goal_terms_added=len(new_matched - old_matched),
        )


def evidence_delta_target(
    current_value: StructuredSemanticStateV3 | Mapping[str, Any],
    next_value: StructuredSemanticStateV3 | Mapping[str, Any],
) -> np.ndarray:
    """Create candidate-conditional next-evidence labels from observed states."""

    current = _as_state(current_value)
    following = _as_state(next_value)
    if following.step_index != current.step_index + 1:
        raise ValueError("evidence delta requires adjacent prefixes")
    current_matched = set(current.goal_evidence.matched_fact_terms)
    following_matched = set(following.goal_evidence.matched_fact_terms)
    current_unresolved = (
        current.goal_evidence.ambiguous_entity_records
        + current.goal_evidence.unlinked_entity_records
    )
    following_unresolved = (
        following.goal_evidence.ambiguous_entity_records
        + following.goal_evidence.unlinked_entity_records
    )
    return np.asarray(
        [
            len(following.evidence_records) > len(current.evidence_records),
            bool(following_matched - current_matched),
            len(following.conflicts) > len(current.conflicts),
            following.execution.last_status == "error",
            following_unresolved > current_unresolved,
        ],
        dtype=np.float32,
    )


class HybridSemanticWorldModel(nn.Module):
    """Learned stochastic heads composed with an exact observed transition."""

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        argument_keys: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
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
        self.victim_action_head = nn.Linear(hidden_size, 1)
        self.argument_key_head = nn.Linear(hidden_size, argument_keys)
        self.evidence_delta_head = nn.Linear(
            hidden_size, len(EVIDENCE_DELTA_TARGETS)
        )
        self.exact_transition = ExactObservedSemanticTransition()

    def forward(
        self,
        states: Tensor,
        candidates: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = self.state_encoder(states)
        candidate = self.candidate_encoder(candidates)
        joint = torch.tanh(state[:, None, :] + candidate[None, :, :])
        action_logits = self.victim_action_head(joint).squeeze(-1)
        argument_logits = self.argument_key_head(state)
        evidence_delta_logits = self.evidence_delta_head(joint)
        return action_logits, argument_logits, evidence_delta_logits

    def action_probabilities(
        self,
        states: Tensor,
        candidates: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        action_logits, _, _ = self(states, candidates)
        if legal_mask.shape != action_logits.shape:
            raise ValueError("legal mask shape differs from action logits")
        if not bool(torch.all(legal_mask.any(dim=1))):
            raise ValueError("every state must expose at least one legal action")
        masked = action_logits.masked_fill(
            ~legal_mask, torch.finfo(action_logits.dtype).min
        )
        return torch.softmax(masked, dim=1)


def assert_no_planning_or_value_heads(model: HybridSemanticWorldModel) -> None:
    forbidden = {
        "actor",
        "critic",
        "planner",
        "planning_head",
        "reward_head",
        "utility_head",
        "value_head",
        "completion_head",
    }
    present = sorted(forbidden & set(dict(model.named_modules())))
    if present:
        raise ValueError(f"forbidden planning/value heads are enabled: {present}")


def stack_state_vectors(
    states: Sequence[StructuredSemanticStateV3 | Mapping[str, Any]],
    *,
    hash_dimension: int,
) -> np.ndarray:
    if not states:
        raise ValueError("cannot stack an empty state sequence")
    return np.stack(
        [
            semantic_state_v3_feature_vector(
                state, hash_dimension=hash_dimension
            )
            for state in states
        ]
    )
