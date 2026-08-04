import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fresh_headwise",
    ROOT / "scripts" / "51_diagnose_headwise_on_fresh_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_headwise_uses_rank_and_probability_from_declared_sources():
    attack = MODULE.BASE._method(
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([0.1]),
        np.asarray([0.2]),
    )
    utility = MODULE.BASE._method(
        np.asarray([3.0]),
        np.asarray([4.0]),
        np.asarray([0.3]),
        np.asarray([0.4]),
    )
    probability = MODULE.BASE._method(
        np.asarray([5.0]),
        np.asarray([6.0]),
        np.asarray([0.7]),
        np.asarray([0.8]),
    )
    result = MODULE._headwise(attack, utility, probability)
    assert np.allclose(result["attack_rank"], [1.0])
    assert np.allclose(result["utility_rank"], [4.0])
    assert np.allclose(result["attack_probability"], [0.7])
    assert np.allclose(result["utility_probability"], [0.8])
