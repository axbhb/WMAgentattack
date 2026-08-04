import numpy as np
import pytest
from types import SimpleNamespace

from wmagentattack.full_dreamer_v3 import (
    FullDreamerV3Config,
    _build_ranking_pairs,
    _append_ranking_pairs,
    _continuous_group_pair_indices,
    _grouped_probability_metrics,
    _grouped_ranking_order,
    _inject_ranking_pairs,
    _multiseed_group_batches,
)


def test_full_dreamer_defaults_include_behavior_learning():
    config = FullDreamerV3Config()
    assert config.imagination_horizon > 0
    assert config.actor_learning_rate > 0
    assert config.critic_learning_rate > 0
    assert config.target_critic_tau > 0
    assert config.behavior_cloning_scale > 0
    assert config.stochastic_size * config.discrete_size > 0
    assert config.observation_feature_mode == "hash"
    assert config.observation_feature_path is None


def test_precomputed_observation_config_is_explicit_and_validated():
    config = FullDreamerV3Config(
        obs_dim=800,
        observation_feature_mode="precomputed",
        observation_feature_path="features.npz",
    )
    assert config.obs_dim == 800
    with pytest.raises(ValueError):
        FullDreamerV3Config(observation_feature_mode="semantic")
    with pytest.raises(ValueError):
        FullDreamerV3Config(observation_feature_mode="precomputed")
    with pytest.raises(ValueError):
        FullDreamerV3Config(observation_feature_path="features.npz")


def test_hybrid_utility_config_is_explicit_and_validated():
    config = FullDreamerV3Config(
        binary_utility_loss_scale=1.0,
        soft_utility_loss_scale=0.0,
        utility_ranking_loss_scale=1.0,
        utility_ranking_detach_latent=True,
        ranking_pairs_per_batch=1,
        utility_reward_binary_mix=1.0,
        risk_reward_binary_mix=0.0,
        binary_risk_loss_scale=0.0,
        soft_risk_loss_scale=1.0,
        validation_risk_mode="continuous",
        validation_utility_mode="binary",
        validation_aggregation="multiseed_group",
        grouped_ranking_batches=True,
        grouped_risk_calibration_batches=True,
        risk_final_step_only=True,
        group_risk_calibration_loss_scale=0.25,
        group_risk_calibration_detach_latent=True,
    )
    assert config.utility_ranking_margin == 0.2
    assert config.utility_ranking_detach_latent
    assert config.ranking_pairs_per_batch == 1
    assert config.grouped_ranking_batches
    assert config.soft_risk_loss_scale == 1.0
    assert config.validation_aggregation == "multiseed_group"
    assert config.group_risk_calibration_loss_scale == 0.25
    with pytest.raises(ValueError):
        FullDreamerV3Config(utility_reward_binary_mix=1.1)
    with pytest.raises(ValueError):
        FullDreamerV3Config(risk_reward_binary_mix=-0.1)
    with pytest.raises(ValueError):
        FullDreamerV3Config(group_risk_calibration_loss_scale=0.1)


def test_grouped_continuous_utility_config_is_opt_in_and_validated():
    config = FullDreamerV3Config(
        group_utility_calibration_loss_scale=1.0,
        group_utility_calibration_detach_latent=True,
        group_utility_ranking_loss_scale=0.5,
        group_utility_ranking_detach_latent=True,
        group_utility_min_target_gap=0.1,
        group_utility_pairs_per_task=8,
        grouped_utility_batches=True,
        group_utility_head_only_updates=True,
        validation_aggregation="multiseed_group",
        validation_group_step="first",
    )
    assert config.grouped_utility_batches
    assert config.group_utility_head_only_updates
    assert config.validation_group_step == "first"
    assert config.group_utility_ranking_loss_scale == 0.5
    with pytest.raises(ValueError):
        FullDreamerV3Config(group_utility_calibration_loss_scale=0.1)
    with pytest.raises(ValueError):
        FullDreamerV3Config(
            grouped_utility_batches=True,
            group_utility_ranking_loss_scale=0.1,
            group_utility_pairs_per_task=0,
        )
    with pytest.raises(ValueError):
        FullDreamerV3Config(group_utility_min_target_gap=1.1)
    with pytest.raises(ValueError):
        FullDreamerV3Config(
            grouped_utility_batches=True,
            group_utility_calibration_loss_scale=1.0,
            group_utility_head_only_updates=True,
        )
    with pytest.raises(ValueError):
        FullDreamerV3Config(validation_group_step="middle")


def test_configuration_value_head_is_head_only_and_checkpoint_validated():
    config = FullDreamerV3Config(
        configuration_value_head_enabled=True,
        group_value_calibration_loss_scale=1.0,
        group_value_ranking_loss_scale=0.5,
        group_value_min_target_gap=0.1,
        group_value_pairs_per_task=8,
        group_value_head_only_updates=True,
        checkpoint_objective="grouped_configuration_value_brier",
    )
    assert config.configuration_value_head_enabled
    assert config.group_value_head_only_updates
    with pytest.raises(ValueError):
        FullDreamerV3Config(group_value_calibration_loss_scale=1.0)
    with pytest.raises(ValueError):
        FullDreamerV3Config(
            configuration_value_head_enabled=True,
            group_value_calibration_loss_scale=1.0,
        )
    with pytest.raises(ValueError):
        FullDreamerV3Config(
            configuration_value_head_enabled=True,
            group_value_ranking_loss_scale=0.5,
            group_value_pairs_per_task=0,
            group_value_head_only_updates=True,
        )
    with pytest.raises(ValueError):
        FullDreamerV3Config(
            checkpoint_objective="grouped_configuration_value_brier"
        )


def test_grouped_ranking_order_interleaves_binary_outcomes():
    sequences = [
        {"group_key": "workspace|task", "final_binary_utility": value}
        for value in (1.0, 1.0, 0.0, 0.0)
    ]
    order = _grouped_ranking_order(sequences, np.random.default_rng(7))
    labels = [sequences[index]["final_binary_utility"] for index in order]
    assert len(order) == len(sequences)
    assert all(left != right for left, right in zip(labels, labels[1:]))


def test_pair_replay_injects_opposite_same_task_outcomes():
    sequences = [
        {"group_key": "workspace|task", "final_binary_utility": 1.0},
        {"group_key": "workspace|task", "final_binary_utility": 0.0},
        {"group_key": "slack|other", "final_binary_utility": 1.0},
        {"group_key": "slack|other", "final_binary_utility": 1.0},
    ]
    pairs = _build_ranking_pairs(sequences)
    assert pairs == [(0, 1)]
    injected = _inject_ranking_pairs(
        np.asarray([2, 3, 2, 3]),
        pairs,
        pairs_per_batch=1,
        rng=np.random.default_rng(7),
    )
    assert tuple(injected[-2:]) == (0, 1)


def test_multiseed_batches_keep_complete_groups_and_append_pairs_outside_loss():
    sequences = []
    for group in ("g1", "g2"):
        for _ in range(5):
            sequences.append(
                {
                    "risk_group_key": group,
                    "risk_group_expected_size": 5,
                    "final_soft_risk": 0.25,
                }
            )
    sequences.append(
        {
            "risk_group_key": None,
            "risk_group_expected_size": 0,
            "final_soft_risk": 0.0,
        }
    )
    batches = _multiseed_group_batches(
        sequences, batch_size=6, rng=np.random.default_rng(7)
    )
    flattened = [int(index) for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(sequences)))
    for batch in batches:
        for group_start in (0, 5):
            count = sum(group_start <= int(index) < group_start + 5 for index in batch)
            assert count in {0, 5}

    extended, calibration_members = _append_ranking_pairs(
        batches[0], [(0, 5)], pairs_per_batch=1, rng=np.random.default_rng(7)
    )
    assert len(extended) == len(batches[0]) + 2
    assert calibration_members.tolist() == [True] * len(batches[0]) + [False, False]


def test_task_aware_utility_batches_keep_groups_complete_and_tasks_separate():
    sequences = []
    group_ranges = {}
    for task, group_count in (("workspace|task-a", 3), ("slack|task-b", 2)):
        for group_index in range(group_count):
            group = f"{task}::g{group_index}"
            start = len(sequences)
            target = 0.1 + 0.2 * group_index
            for _ in range(5):
                sequences.append(
                    {
                        "group_key": task,
                        "utility_group_key": group,
                        "utility_group_expected_size": 5,
                        "final_soft_utility": target,
                    }
                )
            group_ranges[group] = set(range(start, start + 5))
    sequences.append(
        {
            "group_key": "clean|task",
            "utility_group_key": None,
            "utility_group_expected_size": 0,
            "final_soft_utility": 0.0,
        }
    )
    batches = _multiseed_group_batches(
        sequences,
        batch_size=10,
        rng=np.random.default_rng(7),
        group_key_field="utility_group_key",
        expected_size_field="utility_group_expected_size",
        target_field="final_soft_utility",
        group_label="utility",
        task_aware=True,
    )
    flattened = [int(index) for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(sequences)))
    for batch in batches:
        members = set(int(index) for index in batch)
        represented_tasks = {
            sequences[index]["group_key"]
            for index in members
            if sequences[index]["utility_group_key"] is not None
        }
        assert len(represented_tasks) <= 1
        for group_members in group_ranges.values():
            assert len(members & group_members) in {0, 5}


def test_continuous_group_pairs_are_same_task_gap_filtered_and_capped():
    pairs = _continuous_group_pair_indices(
        task_ids=[0, 0, 0, 1, 1],
        targets=[0.2, 0.7, 0.8, 0.1, 0.9],
        min_target_gap=0.15,
        max_pairs_per_task=2,
    )
    assert pairs == [(2, 0, pytest.approx(0.6)), (1, 0, pytest.approx(0.5)), (4, 3, pytest.approx(0.8))]
    assert all(
        [0, 0, 0, 1, 1][high] == [0, 0, 0, 1, 1][low]
        for high, low, _ in pairs
    )


def test_grouped_probability_metrics_use_one_final_prediction_per_trajectory():
    steps = [
        SimpleNamespace(
            trajectory_id="a",
            step_id=0,
            multiseed_group_id="attack::g1",
            attack_probability_target=0.25,
            utility_probability_target=0.75,
            preservation_probability_target=0.8,
            preservation_trainable=True,
        ),
        SimpleNamespace(
            trajectory_id="a",
            step_id=1,
            multiseed_group_id="attack::g1",
            attack_probability_target=0.25,
            utility_probability_target=0.75,
            preservation_probability_target=0.8,
            preservation_trainable=True,
        ),
        SimpleNamespace(
            trajectory_id="b",
            step_id=0,
            multiseed_group_id="attack::g1",
            attack_probability_target=0.25,
            utility_probability_target=0.75,
            preservation_probability_target=0.8,
            preservation_trainable=True,
        ),
        SimpleNamespace(
            trajectory_id="clean",
            step_id=0,
            multiseed_group_id="clean::task",
            attack_probability_target=None,
            utility_probability_target=0.9,
            preservation_probability_target=None,
            preservation_trainable=False,
        ),
    ]
    predictions = {
        # The first value must be ignored because trajectory a ends at step 1.
        "risk_score": np.asarray([0.99, 0.2, 0.3, 0.0]),
        "utility_score": np.asarray([0.01, 0.7, 0.8, 1.0]),
        "preservation_score": np.asarray([0.01, 0.7, 0.9, 1.0]),
        "configuration_value_score": np.asarray([0.2, 0.9, 1.1, 1.0]),
    }
    metrics = _grouped_probability_metrics(steps, predictions)
    assert metrics["grouped_configuration_count"] == 1
    assert metrics["grouped_trajectory_count"] == 2
    assert metrics["grouped_risk_probability_brier_score"] == pytest.approx(0.0)
    assert metrics["grouped_utility_probability_brier_score"] == pytest.approx(0.0)
    assert metrics["grouped_preservation_probability_brier_score"] == pytest.approx(
        0.0
    )
    assert metrics[
        "grouped_configuration_value_normalized_brier_score"
    ] == pytest.approx(0.0)

    first_metrics = _grouped_probability_metrics(
        steps, predictions, decision_step="first"
    )
    assert first_metrics["grouped_utility_probability_brier_score"] == pytest.approx(
        (0.405 - 0.75) ** 2
    )
    assert first_metrics["grouped_risk_probability_brier_score"] == pytest.approx(
        (0.645 - 0.25) ** 2
    )
    assert first_metrics[
        "grouped_configuration_value_normalized_brier_score"
    ] == pytest.approx(((0.65 - 1.0) / 2.0) ** 2)
