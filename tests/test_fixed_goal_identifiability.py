import pytest
from wmagentattack.fixed_goal_identifiability import align_fixed_goals, fixed_goal_audit

POSTERIOR = {"draws": 512, "seed": 31, "dirichlet_prior": 0.5, "confidence_gap": 0.15}
GATE = {"minimum_informative_tasks": 1, "minimum_informative_goals": 1,
        "minimum_task_macro_oracle_random_p11_gap": 0.05, "maximum_feature_collision_fraction": 0.25,
        "minimum_suites_with_two_informative_tasks": 0}


def make_rows(same_within_goal=True, collide=False):
    rows = []
    for goal in range(2):
        for variant in range(2):
            success = 5 * (goal if same_within_goal else variant)
            rows.append({"row_id": f"r{goal}{variant}", "task": "banking|task",
                         "goal": str(goal), "counts": [0, 5-success, 0, success],
                         "empirical_p11": success / 5, "features": {"variant-slot": 0 if collide else variant}})
    return rows


def test_cross_goal_difficulty_does_not_count_as_strategy_signal():
    result = fixed_goal_audit(make_rows(), POSTERIOR, GATE)
    assert result["metrics"]["cross_goal_reward_pairs"] == 4
    assert result["metrics"]["within_goal_p11_pairs"] == 0
    assert result["metrics"]["informative_goals"] == 0
    assert result["metrics"]["task_macro_oracle_random_p11_gap"] == 0
    assert result["decision"].startswith("NO_GO")


def test_real_within_goal_signal_passes_without_feature_collision():
    result = fixed_goal_audit(make_rows(False), POSTERIOR, GATE)
    assert result["metrics"]["informative_goals"] == 2
    assert result["metrics"]["task_macro_oracle_random_p11_gap"] == 0.5
    assert result["decision"].startswith("GO")


def test_identical_allowed_features_fail_even_with_outcome_variation():
    result = fixed_goal_audit(make_rows(False, True), POSTERIOR, GATE)
    assert result["metrics"]["feature_collision_fraction"] == 1
    assert not result["checks"]["feature_distinguishability"]


def test_utility_only_signal_is_not_joint_success_signal():
    rows = make_rows(False)
    for row in rows:
        success = row["counts"][3]
        row["counts"] = [5-success, success, 0, 0]
        row["empirical_p11"] = 0
    result = fixed_goal_audit(rows, POSTERIOR, GATE)
    assert result["metrics"]["informative_goals"] == 0


def test_alignment_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        align_fixed_goals([{"row_id": "x"}, {"row_id": "x"}], [], {"candidates": 2})


def test_alignment_rejects_incorrect_goal_id():
    row = {"row_id": "x", "suite": "s", "user_task_id": "t", "injection_task_id": "i", "base_pair_id": "wrong"}
    with pytest.raises(ValueError, match="goal identity"):
        align_fixed_goals([row], [{"row_id": "x", "source_kind": "attack"}], {"candidates": 1})
