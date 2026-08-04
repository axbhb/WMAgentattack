import pytest

from wmagentattack.multiseed_labels import (
    build_multiseed_labels,
    jeffreys_posterior,
    joint_outcome_posterior,
)


ATTACK_SEEDS = [7, 17, 29, 43, 61]


def _metadata(*, missing_seed=None, duplicate_seed=False):
    rows = []
    for index, seed in enumerate([101, 102, 103]):
        rows.append(
            {
                "trajectory_id": f"clean-{seed}",
                "source_kind": "clean",
                "run_seed": seed,
                "suite": "workspace",
                "user_task_id": "user_task_10",
                "task_split": "train",
                "utility": index < 2,
                "security": False,
            }
        )
    seeds = [seed for seed in ATTACK_SEEDS if seed != missing_seed]
    if duplicate_seed:
        seeds[-1] = seeds[0]
    for index, seed in enumerate(seeds):
        rows.append(
            {
                "trajectory_id": f"attack-{index}",
                "source_kind": "attack",
                "run_seed": seed,
                "row_id": "row-1",
                "suite": "workspace",
                "user_task_id": "user_task_10",
                "injection_task_id": "injection_task_3",
                "task_split": "train",
                "attack_name": "static_control",
                "attack_family": "static_control",
                "attack_role": "control",
                "utility": index < 3,
                "security": index in (0, 2),
            }
        )
    return rows


def test_jeffreys_posterior_is_continuous_and_finite():
    posterior = jeffreys_posterior(3, 5)
    assert posterior.alpha == 3.5
    assert posterior.beta == 2.5
    assert posterior.mean == pytest.approx(3.5 / 6.0)
    assert 0.0 < posterior.variance < 1.0 / 12.0
    assert 0.0 < posterior.confidence < 1.0


def test_multiseed_groups_share_configuration_probability_labels():
    annotations, groups, audit = build_multiseed_labels(
        _metadata(), expected_attack_seeds=ATTACK_SEEDS
    )
    attack = next(row for row in groups if row["source_kind"] == "attack")
    assert attack["utility_probability_successes"] == 3
    assert attack["attack_probability_successes"] == 2
    assert attack["joint_success_probability_successes"] == 2
    assert attack["utility_probability_target"] == pytest.approx(3.5 / 6.0)
    assert attack["attack_probability_target"] == pytest.approx(2.5 / 6.0)
    assert attack["preservation_trainable"] is True
    assert attack["joint_outcome_counts"] == {
        "attack0_utility0": 2,
        "attack0_utility1": 1,
        "attack1_utility0": 0,
        "attack1_utility1": 2,
    }
    assert attack["joint_outcome_trials"] == 5
    assert sum(attack["joint_outcome_probability_target"].values()) == pytest.approx(1.0)
    assert attack["attack_utility_logit_residual_target"] < 0.0
    values = {
        annotations[f"attack-{index}"]["utility_probability_target"]
        for index in range(5)
    }
    assert len(values) == 1
    assert next(iter(values)) == pytest.approx(3.5 / 6.0)
    assert audit["attack_groups"] == 1
    assert audit["attack_trajectories"] == 5
    assert audit["multi_cell_joint_groups"] == 1


def test_joint_outcome_posterior_uses_attack_then_utility_bit_order():
    counts, alpha, probabilities = joint_outcome_posterior(_metadata()[3:])
    assert counts["attack1_utility1"] == 2
    assert counts["attack1_utility0"] == 0
    assert alpha["attack1_utility0"] == 0.5
    assert probabilities["attack0_utility0"] == pytest.approx(2.5 / 7.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"missing_seed": 61}, "seed mismatch"),
        ({"duplicate_seed": True}, "duplicate seeds"),
    ],
)
def test_multiseed_design_rejects_incomplete_or_duplicate_seeds(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_multiseed_labels(
            _metadata(**kwargs), expected_attack_seeds=ATTACK_SEEDS
        )
