import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "headwise_hybrid",
    ROOT / "scripts" / "50_evaluate_train_selected_headwise_hybrid.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_probability_candidates_have_fixed_budget():
    base = {
        "clean_raw": np.asarray([0.1]),
        "text": np.asarray([0.2]),
        "world": np.asarray([0.3]),
        "combined": np.asarray([0.4]),
    }
    candidates = MODULE._probability_candidates(base)
    assert len(candidates) == 10
    assert np.allclose(candidates["text_combined_alpha_0p5"], [0.3])


def test_select_head_uses_mean_task_before_pooled_accuracy():
    metrics = {
        "a": {
            "mean_task_pairwise_accuracy": 0.7,
            "pooled_pairwise_accuracy": 0.6,
        },
        "b": {
            "mean_task_pairwise_accuracy": 0.6,
            "pooled_pairwise_accuracy": 0.9,
        },
    }
    assert MODULE._select_head(metrics, ["a", "b"]) == "a"
