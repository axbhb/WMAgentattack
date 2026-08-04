import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregation_diagnostic",
    ROOT / "scripts" / "48_diagnose_grouped_confirmation_aggregation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_probability_aggregators_are_deterministic():
    matrix = np.asarray([[0.1, 0.8], [0.2, 0.4], [0.9, 0.2]])
    assert np.allclose(
        MODULE._aggregate_probability(matrix, "mean_probability"),
        matrix.mean(axis=0),
    )
    assert np.allclose(
        MODULE._aggregate_probability(matrix, "median_probability"),
        [0.2, 0.4],
    )
    assert np.allclose(
        MODULE._aggregate_probability(matrix, "lower_confidence_bound_1std"),
        np.clip(matrix.mean(axis=0) - matrix.std(axis=0), 1e-5, 1 - 1e-5),
    )


def test_borda_aggregation_is_task_local():
    rows = [
        {"suite": "banking", "user_task_id": "a"},
        {"suite": "banking", "user_task_id": "a"},
        {"suite": "slack", "user_task_id": "b"},
        {"suite": "slack", "user_task_id": "b"},
    ]
    matrix = np.asarray(
        [[0.1, 0.9, 0.8, 0.2], [0.2, 0.8, 0.9, 0.1], [0.3, 0.7, 0.7, 0.3]]
    )
    scores = MODULE._aggregate_borda(rows, matrix)
    assert np.allclose(scores, [0.0, 1.0, 1.0, 0.0])

