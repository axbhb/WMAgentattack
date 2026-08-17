"""Trajectory-level four-cell outcome supervision for Structured Markov."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .adjacent_transition import OBSERVED_OUTCOME_TARGETS
from .multisource_suitability import stable_hash


JOINT_OUTCOME_CLASSES = (
    "attack0_utility0",
    "attack0_utility1",
    "attack1_utility0",
    "attack1_utility1",
)


def build_joint_outcome_dataset(
    *,
    adjacent: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    dirichlet_prior: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_trajectory = {str(row["trajectory_id"]): row for row in metadata}
    if len(by_trajectory) != len(metadata):
        raise ValueError("duplicate trajectory ids in metadata")
    event_trajectories = {str(row["trajectory_id"]) for row in adjacent["events"]}
    if event_trajectories != set(by_trajectory):
        raise ValueError("adjacent events and metadata trajectories differ")

    attack_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metadata:
        if str(row["source_kind"]) == "attack":
            attack_groups[str(row["multiseed_group_id"])].append(row)
    targets: dict[str, dict[str, Any]] = {}
    aggregate = Counter()
    cell_coverage = Counter()
    for group_id, rows in sorted(attack_groups.items()):
        counts = Counter(
            f"attack{int(bool(row['security']))}_utility{int(bool(row['utility']))}"
            for row in rows
        )
        if set(counts) - set(JOINT_OUTCOME_CLASSES):
            raise ValueError("unknown joint outcome cell")
        ordered_counts = {name: int(counts[name]) for name in JOINT_OUTCOME_CLASSES}
        alpha = {
            name: float(ordered_counts[name] + dirichlet_prior)
            for name in JOINT_OUTCOME_CLASSES
        }
        total = sum(alpha.values())
        probabilities = {name: alpha[name] / total for name in JOINT_OUTCOME_CLASSES}
        targets[group_id] = {
            "counts": ordered_counts,
            "dirichlet_alpha": alpha,
            "probability_target": probabilities,
            "trials": len(rows),
        }
        aggregate.update(ordered_counts)
        cell_coverage[sum(value > 0 for value in ordered_counts.values())] += 1

    events = []
    attack_event_rows = 0
    for source in adjacent["events"]:
        row = dict(source)
        meta = by_trajectory[str(row["trajectory_id"])]
        group_id = str(meta["multiseed_group_id"])
        trainable = str(meta["source_kind"]) == "attack"
        if trainable:
            target = targets[group_id]
            row["joint_outcome_target"] = target["probability_target"]
            row["joint_outcome_counts"] = target["counts"]
            attack_event_rows += 1
        else:
            row["joint_outcome_target"] = None
            row["joint_outcome_counts"] = None
        row["joint_outcome_group_id"] = group_id
        row["joint_outcome_trainable"] = trainable
        events.append(row)
    dataset = {
        **{key: value for key, value in adjacent.items() if key != "events"},
        "schema_version": "wmagentattack.structured_joint_outcome_auxiliary.v1",
        "events": events,
        "joint_outcome_classes": list(JOINT_OUTCOME_CLASSES),
        "joint_outcome_group_targets": targets,
    }
    checks = {
        "trajectory_alignment": event_trajectories == set(by_trajectory),
        "expected_attack_groups": len(attack_groups) == 400,
        "five_seed_rectangular_groups": all(len(rows) == 5 for rows in attack_groups.values()),
        "all_four_cells_observed_globally": all(aggregate[name] > 0 for name in JOINT_OUTCOME_CLASSES),
        "targets_sum_to_one": all(
            abs(sum(value["probability_target"].values()) - 1.0) < 1e-9
            for value in targets.values()
        ),
        "labels_outside_causal_input": all(
            "joint_outcome_target" not in event["causal_model_input"] for event in events
        ),
        "all_attack_events_labeled": all(
            (event["joint_outcome_target"] is not None)
            == bool(event["joint_outcome_trainable"])
            for event in events
        ),
    }
    audit = {
        "passed": all(checks.values()),
        "checks": checks,
        "event_rows": len(events),
        "trajectories": len(event_trajectories),
        "attack_groups": len(attack_groups),
        "attack_trajectories": sum(len(rows) for rows in attack_groups.values()),
        "attack_event_rows": attack_event_rows,
        "joint_outcome_counts": dict(aggregate),
        "groups_by_observed_cell_count": {str(key): value for key, value in sorted(cell_coverage.items())},
        "dirichlet_prior_per_cell": dirichlet_prior,
        "dataset_content_sha256": stable_hash(dataset),
    }
    return dataset, audit


class StructuredJointOutcomeModel(nn.Module):
    """Original structured MLP with one additional four-cell auxiliary head."""

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
            nn.Linear(state_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.next_candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.next_action_head = nn.Linear(hidden_size, 1)
        self.outcome_head = nn.Linear(hidden_size, len(OBSERVED_OUTCOME_TARGETS))
        self.joint_outcome_head = nn.Linear(hidden_size, len(JOINT_OUTCOME_CLASSES))

    def forward(
        self, states: Tensor, selected_actions: Tensor, next_candidates: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        context = torch.tanh(
            self.state_encoder(states) + self.action_encoder(selected_actions)
        )
        candidate = self.next_candidate_encoder(next_candidates)
        joint = torch.tanh(context[:, None, :] + candidate[None, :, :])
        return (
            self.next_action_head(joint).squeeze(-1),
            self.outcome_head(context),
            self.joint_outcome_head(context),
        )


def normalized_joint_event_weights(
    events: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> np.ndarray:
    """Equalize task, configuration, seed trajectory, and trajectory length."""

    selected = [events[int(index)] for index in indices]
    tasks = sorted({str(row["task_name"]) for row in selected})
    groups_by_task: dict[str, set[str]] = defaultdict(set)
    trajectories_by_group: dict[str, set[str]] = defaultdict(set)
    events_by_trajectory = Counter(str(row["trajectory_id"]) for row in selected)
    for row in selected:
        task = str(row["task_name"])
        group = str(row["joint_outcome_group_id"])
        groups_by_task[task].add(group)
        trajectories_by_group[group].add(str(row["trajectory_id"]))
    weights = []
    for row in selected:
        task = str(row["task_name"])
        group = str(row["joint_outcome_group_id"])
        trajectory = str(row["trajectory_id"])
        weights.append(
            1.0 / len(tasks)
            / len(groups_by_task[task])
            / len(trajectories_by_group[group])
            / events_by_trajectory[trajectory]
        )
    output = np.asarray(weights, dtype=np.float32)
    output *= len(output) / float(output.sum())
    return output
