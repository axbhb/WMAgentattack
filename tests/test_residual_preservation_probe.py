import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "91_probe_v2_residual_preservation.py"
)
SPEC = importlib.util.spec_from_file_location("residual_preservation_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(clean, utility):
    _, clean_variance = MODULE._jeffreys_posterior(
        round((clean * 4) - 0.5), 3
    )
    _, utility_variance = MODULE._jeffreys_posterior(
        round((utility * 6) - 0.5), 5
    )
    return {
        "clean_probability": clean,
        "clean_variance": clean_variance,
        "target_bup": utility,
        "utility_variance": utility_variance,
    }


def test_jeffreys_posterior_matches_three_and_five_trial_labels():
    assert MODULE._jeffreys_posterior(2, 3)[0] == pytest.approx(0.625)
    assert MODULE._jeffreys_posterior(4, 5)[0] == pytest.approx(0.75)


def test_logit_residual_model_orders_attack_damage_conditional_on_clean_prior():
    rows = [
        _row(0.875, 0.75),
        _row(0.875, 0.25),
        _row(0.625, 0.5833333333333334),
        _row(0.625, 0.08333333333333333),
    ]
    matrix = np.asarray([[1.0], [0.0], [0.8], [-0.2]], dtype=np.float64)
    model = MODULE._fit_residual_model(
        matrix, rows, target_kind="logit_residual", alpha=0.1
    )
    prediction, standard_deviation, _ = MODULE._predict_residual_model(
        model, matrix, rows
    )
    assert prediction[0] > prediction[1]
    assert prediction[2] > prediction[3]
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))
    assert np.all(standard_deviation >= 0.0)


def test_uplift_prediction_restores_clean_probability_offset():
    train_rows = [_row(0.875, 0.75), _row(0.375, 0.25)]
    train_matrix = np.zeros((2, 1), dtype=np.float64)
    model = MODULE._fit_residual_model(
        train_matrix,
        train_rows,
        target_kind="uplift_probability",
        alpha=1.0,
    )
    evaluation_rows = [_row(0.875, 0.75), _row(0.375, 0.25)]
    prediction, _, _ = MODULE._predict_residual_model(
        model, np.zeros((2, 1), dtype=np.float64), evaluation_rows
    )
    assert prediction[0] > prediction[1]


def test_selection_rejects_candidate_below_control_bup():
    def candidate(kind, joint, bup, order):
        return {
            "target_kind": kind,
            "alpha": 1.0,
            "utility_weight": 1.0,
            "uncertainty_penalty": 0.0,
            "fixed_order": order,
            "validation": {
                "top1_target_ASR_plus_BUP": joint,
                "top1_target_BUP": bup,
                "mean_task_spearman": 0.0,
                "normalized_brier": 0.1,
            },
        }

    grid = [
        candidate("direct_probability", 1.5, 0.4, 0),
        candidate("direct_probability", 1.2, 0.6, 1),
        candidate("uplift_probability", 1.3, 0.6, 2),
        candidate("logit_residual", 1.4, 0.6, 3),
    ]
    overall, by_target = MODULE._select_candidates(grid, control_bup=0.6)
    assert overall["target_kind"] == "logit_residual"
    assert not grid[0]["BUP_constraint_pass"]
    assert by_target["direct_probability"]["fixed_order"] == 1


def test_selection_treats_machine_precision_joint_values_as_ties():
    base = {
        "alpha": 1.0,
        "utility_weight": 1.0,
        "uncertainty_penalty": 0.0,
    }
    grid = []
    for order, (kind, joint, correlation) in enumerate(
        [
            ("direct_probability", 1.0, -0.1),
            ("uplift_probability", 0.9, 0.0),
            ("logit_residual", 0.9999999999999999, 0.1),
        ]
    ):
        grid.append(
            {
                **base,
                "target_kind": kind,
                "fixed_order": order,
                "validation": {
                    "top1_target_ASR_plus_BUP": joint,
                    "top1_target_BUP": 0.6,
                    "mean_task_spearman": correlation,
                    "normalized_brier": 0.1,
                },
            }
        )
    overall, _ = MODULE._select_candidates(grid, control_bup=0.6)
    assert overall["target_kind"] == "logit_residual"
