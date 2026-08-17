from __future__ import annotations

import numpy as np
import torch

from wmagentattack.joint_outcome_auxiliary import (
    JOINT_OUTCOME_CLASSES,
    StructuredJointOutcomeModel,
    build_joint_outcome_dataset,
    normalized_joint_event_weights,
)


def test_joint_builder_keeps_labels_outside_causal_input() -> None:
    metadata = []
    events = []
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
    for seed, (attack, utility) in enumerate(outcomes):
        trajectory = f"t{seed}"
        metadata.append({"trajectory_id": trajectory, "source_kind": "attack", "multiseed_group_id": "g", "security": attack, "utility": utility})
        events.append({"trajectory_id": trajectory, "causal_model_input": {"trusted_goal": "x"}})
    adjacent = {"events": events}
    dataset, audit = build_joint_outcome_dataset(
        adjacent=adjacent, metadata=metadata, dirichlet_prior=0.5
    )
    assert not audit["passed"]  # toy fixture intentionally has fewer than 400 groups
    target = dataset["events"][0]["joint_outcome_target"]
    assert set(target) == set(JOINT_OUTCOME_CLASSES)
    assert abs(sum(target.values()) - 1.0) < 1e-9
    assert "joint_outcome_target" not in dataset["events"][0]["causal_model_input"]


def test_hierarchical_weights_sum_equally_by_trajectory() -> None:
    events = []
    for trajectory, length in (("a", 2), ("b", 4)):
        for _ in range(length):
            events.append({"task_name": "task", "joint_outcome_group_id": "group", "trajectory_id": trajectory})
    weights = normalized_joint_event_weights(events, list(range(len(events))))
    assert np.isclose(weights.sum(), len(events))
    assert np.isclose(weights[:2].sum(), weights[2:].sum())


def test_structured_joint_model_shapes() -> None:
    model = StructuredJointOutcomeModel(
        state_size=10, candidate_size=8, hidden_size=16, dropout=0.0
    )
    action, outcome, joint = model(
        torch.randn(3, 10), torch.randn(3, 8), torch.randn(5, 8)
    )
    assert action.shape == (3, 5)
    assert outcome.shape == (3, 3)
    assert joint.shape == (3, 4)
