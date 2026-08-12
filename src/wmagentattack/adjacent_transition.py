"""Observed adjacent-step dynamics for the frozen AgentDojo-v2 traces.

The action at step t is an intervention input.  Only information visible at
step t is used as state input; the action/outcome at t+1 remains a target.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .multisource_suitability import stable_hash


ADJACENT_TRANSITION_SCHEMA_VERSION = (
    "wmagentattack.agentdojo_observed_adjacent_transition.v1"
)
OBSERVED_OUTCOME_TARGETS = (
    "execution_error",
    "output_nonempty",
    "trajectory_continues",
)

_FORBIDDEN_CAUSAL_KEYS = {
    "attack_action",
    "attack_success",
    "decision",
    "execution",
    "future_action",
    "next_action",
    "policy_violation",
    "reward",
    "security",
    "target",
    "task_success",
    "utility",
}


def build_adjacent_transition_dataset(
    *,
    unified: Mapping[str, Any],
    raw_steps: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one action-conditioned outcome row per observed AgentDojo step."""

    unified_rows = {
        str(row["row_id"]): row
        for row in unified["rows"]
        if row["source"] == "agentdojo"
    }
    if len(unified_rows) != int(protocol["source"]["expected_step_rows"]):
        raise ValueError("AgentDojo unified row count differs from protocol")

    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for step in raw_steps:
        by_trajectory[str(step["trajectory_id"])].append(step)

    events: list[dict[str, Any]] = []
    contiguous = True
    same_trajectory = True
    next_target_legal = True
    current_action_legal = True
    interfaces_equal = 0
    next_observation_equals_output = 0
    next_observation_changed = 0
    variant_counts: Counter[str] = Counter()
    error_tasks: set[str] = set()
    adjacent_tasks: set[str] = set()

    for trajectory_id, unsorted in sorted(by_trajectory.items()):
        steps = sorted(unsorted, key=lambda row: int(row["step_id"]))
        contiguous &= [int(row["step_id"]) for row in steps] == list(
            range(len(steps))
        )
        for index, step in enumerate(steps):
            row_id = f"agentdojo::{trajectory_id}::step{index}"
            current = unified_rows.get(row_id)
            if current is None:
                raise ValueError(f"missing unified AgentDojo step: {row_id}")
            following_step = steps[index + 1] if index + 1 < len(steps) else None
            following = None
            if following_step is not None:
                following_id = f"agentdojo::{trajectory_id}::step{index + 1}"
                following = unified_rows.get(following_id)
                if following is None:
                    raise ValueError(f"missing following unified step: {following_id}")
                same_trajectory &= str(following_step["trajectory_id"]) == trajectory_id
                next_target_legal &= (
                    following["target_candidate_id"]
                    in following["legal_candidate_ids"]
                )
                interfaces_equal += int(
                    current["legal_candidate_ids"]
                    == following["legal_candidate_ids"]
                )
                output = str(step.get("skill_output", "")).strip()
                next_observation = str(
                    following_step.get("current_observation", "")
                ).strip()
                next_observation_equals_output += int(next_observation == output)
                next_observation_changed += int(
                    next_observation
                    != str(step.get("current_observation", "")).strip()
                )
                adjacent_tasks.add(str(current["task_name"]))

            current_action = str(current["target_candidate_id"])
            current_action_legal &= current_action in current["legal_candidate_ids"]
            error = bool(step.get("tool_error"))
            if error:
                error_tasks.add(str(current["task_name"]))
            variant = str(current.get("variant", "unknown"))
            variant_counts[variant] += 1
            events.append(
                {
                    "event_id": f"{row_id}::observed_transition",
                    "task_key": current["task_key"],
                    "task_name": current["task_name"],
                    "task_cohort": current["task_cohort"],
                    "group_id": str(step["multiseed_group_id"]),
                    "trajectory_id": trajectory_id,
                    "step_id": index,
                    "variant": variant,
                    "causal_model_input": current["causal_model_input"],
                    "causal_input_fingerprint": current[
                        "causal_input_fingerprint"
                    ],
                    "current_action_candidate_id": current_action,
                    "current_legal_candidate_ids": current[
                        "legal_candidate_ids"
                    ],
                    "next_target_candidate_id": (
                        following["target_candidate_id"]
                        if following is not None
                        else None
                    ),
                    "next_legal_candidate_ids": (
                        following["legal_candidate_ids"]
                        if following is not None
                        else current["legal_candidate_ids"]
                    ),
                    "observed_outcome": {
                        "execution_error": error,
                        "output_nonempty": bool(step.get("skill_output")),
                        "trajectory_continues": following is not None,
                    },
                }
            )

    forbidden = sorted(
        {
            key
            for event in events
            for key in event["causal_model_input"]
            if str(key).lower() in _FORBIDDEN_CAUSAL_KEYS
        }
    )
    trajectory_lengths = Counter(len(rows) for rows in by_trajectory.values())
    outcome_positives = {
        name: sum(bool(row["observed_outcome"][name]) for row in events)
        for name in OBSERVED_OUTCOME_TARGETS
    }
    checks = {
        "expected_step_rows": len(events)
        == int(protocol["source"]["expected_step_rows"]),
        "expected_trajectories": len(by_trajectory)
        == int(protocol["source"]["expected_trajectories"]),
        "expected_adjacent_transitions": outcome_positives[
            "trajectory_continues"
        ]
        == int(protocol["source"]["expected_adjacent_transitions"]),
        "expected_multistep_trajectories": sum(
            count for length, count in trajectory_lengths.items() if length > 1
        )
        == int(protocol["source"]["expected_multistep_trajectories"]),
        "all_step_indices_contiguous": contiguous,
        "all_adjacent_rows_same_trajectory": same_trajectory,
        "current_actions_legal": current_action_legal,
        "next_targets_legal": next_target_legal,
        "all_expected_tasks_have_adjacent_transitions": len(adjacent_tasks)
        == int(protocol["source"]["expected_tasks_with_adjacent_transitions"]),
        "minimum_execution_errors": outcome_positives["execution_error"]
        >= int(protocol["preflight_gate"]["minimum_execution_errors"]),
        "minimum_tasks_with_execution_errors": len(error_tasks)
        >= int(protocol["preflight_gate"]["minimum_tasks_with_execution_errors"]),
        "forbidden_causal_keys_absent": not forbidden,
    }
    catalog = {
        key: value
        for key, value in unified["candidate_catalog"].items()
        if value["source"] == "agentdojo"
    }
    dataset = {
        "schema_version": ADJACENT_TRANSITION_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "events": events,
        "candidate_catalog": dict(sorted(catalog.items())),
        "agentdojo_cohorts": unified["agentdojo_cohorts"],
        "folds": unified["folds"],
    }
    audit = {
        "schema_version": ADJACENT_TRANSITION_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "event_rows": len(events),
        "trajectories": len(by_trajectory),
        "trajectory_length_distribution": {
            str(key): value for key, value in sorted(trajectory_lengths.items())
        },
        "multistep_trajectories": sum(
            count for length, count in trajectory_lengths.items() if length > 1
        ),
        "adjacent_transitions": outcome_positives["trajectory_continues"],
        "outcome_positive_rows": outcome_positives,
        "tasks_with_adjacent_transitions": len(adjacent_tasks),
        "tasks_with_execution_errors": len(error_tasks),
        "variant_rows": dict(sorted(variant_counts.items())),
        "adjacent_interfaces_equal": interfaces_equal,
        "next_observation_equals_current_output": next_observation_equals_output,
        "next_observation_changed": next_observation_changed,
        "forbidden_causal_keys": forbidden,
        "candidate_count": len(catalog),
        "dataset_content_sha256": stable_hash(dataset),
        "counterevidence": {
            "clean_rows_are_a_small_minority": variant_counts.get("clean", 0)
            < len(events) / 10,
            "observable_outcomes_are_not_hidden_simulator_state": True,
            "state_changed_or_task_success_is_not_claimed": True,
            "nonidentical_adjacent_legal_interfaces": (
                outcome_positives["trajectory_continues"] - interfaces_equal
            ),
        },
    }
    return dataset, audit


class ObservedAdjacentTransitionModel(nn.Module):
    """Predict the next victim action and observable outcome after action t."""

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
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
        self.next_action_head = nn.Linear(hidden_size, 1)
        self.outcome_head = nn.Linear(hidden_size, len(OBSERVED_OUTCOME_TARGETS))

    def forward(
        self,
        states: Tensor,
        selected_actions: Tensor,
        next_candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state = self.state_encoder(states)
        action = self.action_encoder(selected_actions)
        context = torch.tanh(state + action)
        candidate = self.next_candidate_encoder(next_candidates)
        joint = torch.tanh(context[:, None, :] + candidate[None, :, :])
        next_action_logits = self.next_action_head(joint).squeeze(-1)
        outcome_logits = self.outcome_head(context)
        return next_action_logits, outcome_logits

    def next_action_probabilities(
        self,
        states: Tensor,
        selected_actions: Tensor,
        next_candidates: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        logits, _ = self(states, selected_actions, next_candidates)
        if logits.shape != legal_mask.shape:
            raise ValueError("legal mask shape differs from next-action logits")
        if not bool(torch.all(legal_mask.any(dim=1))):
            raise ValueError("every row must expose a legal next-action interface")
        masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=1)


def evaluate_adjacent_transition_gate(
    *,
    action_nll_seed_gains: Sequence[float],
    action_accuracy_seed_gains: Sequence[float],
    action_task_gains: Sequence[float],
    outcome_bce_seed_gains: Sequence[float],
    execution_error_bce_gain: float,
    all_predictions_legal: bool,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    minimum_seeds = int(gates["minimum_threshold_positive_seeds"])
    return {
        "tail_action_mean_nll_gain": float(np.mean(action_nll_seed_gains))
        >= float(gates["minimum_tail_action_nll_gain"]),
        "tail_action_mean_accuracy_gain": float(
            np.mean(action_accuracy_seed_gains)
        )
        >= float(gates["minimum_tail_action_accuracy_gain"]),
        "tail_action_nll_seed_replication": sum(
            value >= float(gates["minimum_tail_action_nll_gain"])
            for value in action_nll_seed_gains
        )
        >= minimum_seeds,
        "tail_action_accuracy_seed_replication": sum(
            value >= float(gates["minimum_tail_action_accuracy_gain"])
            for value in action_accuracy_seed_gains
        )
        >= minimum_seeds,
        "tail_action_positive_task_fraction": (
            sum(value > 0.0 for value in action_task_gains)
            / max(1, len(action_task_gains))
        )
        >= float(gates["minimum_positive_task_fraction"]),
        "observable_outcome_mean_bce_gain": float(
            np.mean(outcome_bce_seed_gains)
        )
        >= float(gates["minimum_outcome_bce_gain_over_train_prior"]),
        "observable_outcome_seed_replication": sum(
            value >= float(gates["minimum_outcome_bce_gain_over_train_prior"])
            for value in outcome_bce_seed_gains
        )
        >= minimum_seeds,
        "execution_error_bce_gain": execution_error_bce_gain
        >= float(gates["minimum_execution_error_bce_gain"]),
        "all_predictions_legal": all_predictions_legal,
    }
