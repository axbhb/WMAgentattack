from __future__ import annotations

import numpy as np
import torch

from wmagentattack.comparison_reward_policy import (
    ComparisonRewardPolicy,
    build_preference_pairs,
    constrained_reward,
    four_cell_counts_from_label_group,
    posterior_preference_probability,
    preference_metrics,
    soft_preference_loss,
)


def test_four_cell_counts_and_reward_penalize_attack_only() -> None:
    counts = four_cell_counts_from_label_group(
        {
            "joint_success_probability_trials": 5,
            "attack_probability_successes": 3,
            "utility_probability_successes": 4,
            "joint_success_probability_successes": 2,
        }
    )
    assert counts == [0, 2, 1, 2]
    assert constrained_reward([0, 0, 1, 0]) < constrained_reward([0, 1, 0, 0])
    assert constrained_reward([0, 1, 0, 0]) < constrained_reward([0, 0, 0, 1])


def test_posterior_preference_is_deterministic_and_orders_clear_counts() -> None:
    first = posterior_preference_probability(
        [0, 0, 0, 5], [0, 5, 0, 0], left_id="a", right_id="b", draws=2048, seed=17
    )
    second = posterior_preference_probability(
        [0, 0, 0, 5], [0, 5, 0, 0], left_id="a", right_id="b", draws=2048, seed=17
    )
    assert first == second
    assert first > 0.9


def test_pair_support_and_metrics_are_task_level() -> None:
    rows = []
    for task in ("t", "u"):
        for index, counts in enumerate(([0, 0, 0, 5], [0, 5, 0, 0], [5, 0, 0, 0])):
            target = (np.asarray(counts, dtype=float) + 0.5) / 7.0
            rows.append(
                {
                    "row_id": f"{task}-{index}",
                    "task_name": task,
                    "attack_family": f"family-{index}",
                    "counts": counts,
                    "target_p11": float(target[3]),
                    "target_utility": float(target[1] + target[3]),
                    "target_reward": constrained_reward(target),
                    "score": float(2 - index),
                }
            )
    pairs, audit = build_preference_pairs(
        rows, draws=1024, posterior_seed=23, minimum_confidence_gap=0.1
    )
    assert audit["tasks_with_confident_pairs"] == 2
    metrics = preference_metrics(rows, score_key="score", pairs=pairs)
    assert metrics["task_count"] == 2
    assert metrics["top1_target_p11"] > metrics["random_expected_p11"]
    assert metrics["posterior_pairwise_accuracy"] > 0.5


def test_soft_preference_loss_and_zero_reward_start() -> None:
    model = ComparisonRewardPolicy(input_size=5, hidden_size=8, dropout=0.0)
    scores, outcomes = model(torch.randn(3, 5))
    assert torch.allclose(scores, torch.zeros_like(scores))
    assert outcomes.shape == (3, 4)
    pairs = torch.tensor([[0.0, 1.0, 0.9, 0.4], [1.0, 2.0, 0.8, 0.3]])
    loss = soft_preference_loss(torch.tensor([1.0, 0.0, -1.0]), pairs)
    reversed_loss = soft_preference_loss(torch.tensor([-1.0, 0.0, 1.0]), pairs)
    assert loss < reversed_loss
