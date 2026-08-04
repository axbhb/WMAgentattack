import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grouped_train_hybrid",
    ROOT / "scripts" / "49_evaluate_grouped_train_hybrid.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(task, injection):
    seed_predictions = {}
    for seed, offset in zip(MODULE.SEEDS, (0.0, 0.1, 0.2), strict=True):
        seed_predictions[str(seed)] = {}
        for mode, base in zip(MODULE.MODES, (0.2, 0.6), strict=True):
            seed_predictions[str(seed)][mode] = {
                field: base + offset for field in MODULE.WORLD_FIELDS
            }
    return {
        "suite": "banking",
        "user_task_id": task,
        "injection_task_id": injection,
        "seed_predictions": seed_predictions,
    }


def test_world_matrix_contains_mode_statistics_and_deltas():
    matrix = MODULE._world_matrix([_row("task", "a")])
    assert matrix.shape == (1, len(MODULE.WORLD_FIELDS) * 9)
    assert np.isclose(matrix[0, 0], 0.3)
    assert np.isclose(matrix[0, -1], 0.4)


def test_pairwise_ranker_uses_only_within_task_pairs():
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    labels = np.asarray([0.0, 1.0, 1.0, 0.0])
    groups = np.asarray(["a", "a", "b", "b"])
    prediction, pair_count = MODULE._pairwise_rank_predict(
        matrix, labels, groups, matrix, c_value=0.1
    )
    assert prediction.shape == (4,)
    assert pair_count == 2


def test_decoupled_metrics_change_rank_without_changing_brier():
    rows = [
        {
            "suite": "banking",
            "user_task_id": "task",
            "injection_task_id": str(index),
        }
        for index in range(3)
    ]
    rates = np.asarray([0.0, 0.5, 1.0])
    probability = np.asarray([0.2, 0.5, 0.8])
    good = MODULE._method(rates, rates, probability, probability)
    bad = MODULE._method(rates[::-1], rates[::-1], probability, probability)
    good_metrics = MODULE._method_metrics(rows, rates, rates, good)
    bad_metrics = MODULE._method_metrics(rows, rates, rates, bad)
    assert good_metrics["mean_pair_soft_brier"] == bad_metrics["mean_pair_soft_brier"]
    assert (
        good_metrics["primary_mean_within_task_pairwise_accuracy"]
        > bad_metrics["primary_mean_within_task_pairwise_accuracy"]
    )


def test_fixed_candidate_budget_is_seventeen():
    assert 5 + 3 * len(MODULE.C_VALUES) + len(MODULE.BLEND_ALPHAS) == 17
