import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grouped_transfer",
    ROOT / "scripts" / "45_evaluate_grouped_task_transfer.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dual_view_uses_injection_risk_and_clean_utility():
    rows = [
        {
            "suite": "banking",
            "user_task_id": "task",
            "injection_task_id": "injection",
        }
    ]
    key = ("banking", "task", "injection")
    mappings = {
        "clean_prefix_rollout": {
            7: {key: {"risk_score": 0.1, "utility_score": 0.8}},
            13: {key: {"risk_score": 0.2, "utility_score": 0.6}},
            21: {key: {"risk_score": 0.3, "utility_score": 0.7}},
        },
        "injection_conditioned_rollout": {
            7: {key: {"risk_score": 0.7, "utility_score": 0.2}},
            13: {key: {"risk_score": 0.8, "utility_score": 0.3}},
            21: {key: {"risk_score": 0.9, "utility_score": 0.4}},
        },
    }
    predictions = MODULE._predictions(rows, mappings, MODULE.SEEDS)
    assert np.allclose(predictions["dual_view"][0], [0.8])
    assert np.allclose(predictions["dual_view"][1], [0.7])
