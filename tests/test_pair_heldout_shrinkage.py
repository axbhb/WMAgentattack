import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pair_heldout_shrinkage",
    ROOT / "scripts" / "44_evaluate_pair_heldout_shrinkage.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_blend_and_replication_rule():
    clean = np.asarray([0.2, 0.8])
    injection = np.asarray([0.6, 0.4])
    assert np.allclose(MODULE._blend(clean, injection, 0.5), [0.4, 0.6])
    status = MODULE._replication_status(
        {
            "pairwise_accuracy_difference": 0.1,
            "pairwise_accuracy_difference_95ci": [-0.02, 0.2],
            "brier_difference": 0.005,
        }
    )
    assert status["directional_replication"] is True
    assert status["strong_replication"] is False
