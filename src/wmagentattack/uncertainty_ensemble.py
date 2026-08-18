"""Uniform categorical world-model ensembles and epistemic decomposition."""

from __future__ import annotations

import numpy as np


def categorical_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim < 1 or np.any(probabilities < 0):
        raise ValueError("invalid categorical probabilities")
    totals = probabilities.sum(axis=-1)
    if not np.allclose(totals, 1.0, atol=1e-5):
        raise ValueError("categorical probabilities must sum to one")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(probabilities * np.log(clipped)).sum(axis=-1)


def uniform_categorical_ensemble(
    member_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mixture, total entropy, expected entropy, and epistemic MI."""
    members = np.asarray(member_probabilities, dtype=np.float64)
    if members.ndim < 2 or len(members) < 2:
        raise ValueError("at least two member distributions are required")
    mixture = members.mean(axis=0)
    predictive = categorical_entropy(mixture)
    expected = categorical_entropy(members).mean(axis=0)
    epistemic = np.maximum(predictive - expected, 0.0)
    return mixture, predictive, expected, epistemic
