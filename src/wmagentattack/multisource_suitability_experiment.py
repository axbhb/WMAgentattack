"""Frozen evaluation helpers for the multi-source suitability study.

The functions in this module contain only preregistered aggregation and gate
logic.  They deliberately do not select hyperparameters or inspect calibration
results to alter the confirmation decision.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


SOURCE_SCOPES = ("tool_sandbox", "injecagent", "tau3")
TRAINING_SCOPES = (*SOURCE_SCOPES, "combined")
NEURAL_VARIANTS = (
    "semantic_markov",
    "structured_markov_v3",
    "full_history_diagnostic",
)
BASELINE_VARIANTS = ("frequency_prior", "tfidf_candidate_logistic")


def rows_for_scope(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> list[Mapping[str, Any]]:
    if scope not in TRAINING_SCOPES:
        raise ValueError(f"unsupported scope: {scope}")
    if scope == "combined":
        return list(rows)
    return [row for row in rows if str(row["source"]) == scope]


def task_balanced_weights(task_ids: Sequence[str]) -> np.ndarray:
    """Give each task equal aggregate mass while preserving row-level loss."""

    counts = Counter(map(str, task_ids))
    if not counts:
        raise ValueError("cannot weight an empty row set")
    scale = len(task_ids) / len(counts)
    return np.asarray(
        [scale / counts[str(task_id)] for task_id in task_ids], dtype=np.float32
    )


def task_metric_map(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[str(row["task_key"])].append(float(value))
    if not grouped:
        raise ValueError(f"no task values for metric: {metric}")
    return {
        task: float(np.mean(values)) for task, values in sorted(grouped.items())
    }


def task_macro(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(np.mean(list(task_metric_map(rows, metric).values())))


def paired_task_gain(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    """Return baseline-minus-candidate task gains for a lower-is-better metric."""

    baseline = task_metric_map(baseline_rows, metric)
    candidate = task_metric_map(candidate_rows, metric)
    if set(baseline) != set(candidate):
        raise ValueError("paired task surfaces differ")
    return {
        task: float(baseline[task] - candidate[task]) for task in sorted(baseline)
    }


def error_probe_supported(
    rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    exact = [row for row in rows if row["exact_outcome"]["available"]]
    by_split: dict[str, dict[str, int]] = {}
    for split in ("training", "calibration", "confirmation"):
        selected = [row for row in exact if row["split"] == split]
        errors = sum(bool(row["exact_outcome"]["execution_error"]) for row in selected)
        by_split[split] = {
            "rows": len(selected),
            "errors": int(errors),
            "successes": int(len(selected) - errors),
        }
    errors = sum(bool(row["exact_outcome"]["execution_error"]) for row in exact)
    successes = len(exact) - errors
    minimum_each = int(gate["minimum_each_error_class_per_training_and_confirmation"])
    supported = (
        len(exact) >= int(gate["minimum_exact_rows_for_error_probe"])
        and errors >= int(gate["minimum_exact_errors_for_error_probe"])
        and successes >= int(gate["minimum_exact_successes_for_error_probe"])
        and all(
            by_split[split]["errors"] >= minimum_each
            and by_split[split]["successes"] >= minimum_each
            for split in ("training", "confirmation")
        )
    )
    return supported, {
        "exact_rows": len(exact),
        "errors": int(errors),
        "successes": int(successes),
        "by_split": by_split,
    }


def exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value > 0.0 for value in values)
    losses = sum(value < 0.0 for value in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        lower = min(wins, losses)
        tail = sum(math.comb(n, index) for index in range(lower + 1))
        p_value = min(1.0, 2.0 * tail / (2**n))
    return {"wins": wins, "losses": losses, "ties": ties, "p_value": p_value}


def paired_bootstrap(
    values: Sequence[float], *, draws: int, seed: int
) -> dict[str, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty paired surface")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
    }


def evaluate_action_gate(
    *,
    nll_seed_gains: Sequence[float],
    accuracy_seed_gains: Sequence[float],
    paired_nll_task_gains: Sequence[float],
    structured_nll_gap_to_tfidf: float,
    legal_prediction_rate: float,
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    minimum_seed_count = int(gate["minimum_threshold_positive_seeds"])
    minimum_task_fraction = float(gate["minimum_positive_task_fraction"])
    minimum_nll = float(gate["minimum_structured_nll_gain_over_frequency"])
    minimum_accuracy = float(
        gate["minimum_structured_accuracy_gain_over_frequency"]
    )
    return {
        "structured_nll_mean_gain": float(np.mean(nll_seed_gains)) >= minimum_nll,
        "structured_nll_seed_replication": sum(
            gain >= minimum_nll for gain in nll_seed_gains
        )
        >= minimum_seed_count,
        "structured_accuracy_mean_gain": float(np.mean(accuracy_seed_gains))
        >= minimum_accuracy,
        "structured_accuracy_seed_replication": sum(
            gain >= minimum_accuracy for gain in accuracy_seed_gains
        )
        >= minimum_seed_count,
        "structured_positive_task_fraction": (
            sum(gain > 0.0 for gain in paired_nll_task_gains)
            / len(paired_nll_task_gains)
        )
        >= minimum_task_fraction,
        "structured_within_tfidf_nll_gap": structured_nll_gap_to_tfidf
        <= float(gate["maximum_structured_nll_gap_to_tfidf"]),
        "all_predictions_legal": legal_prediction_rate == 1.0,
    }


def evaluate_error_gate(
    *,
    bce_seed_gains: Sequence[float],
    paired_bce_task_gains: Sequence[float],
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    minimum_gain = float(gate["minimum_bce_gain_over_frequency"])
    return {
        "structured_error_bce_mean_gain": float(np.mean(bce_seed_gains))
        >= minimum_gain,
        "structured_error_bce_seed_replication": sum(
            gain >= minimum_gain for gain in bce_seed_gains
        )
        >= int(gate["minimum_threshold_positive_seeds"]),
        "structured_error_positive_task_fraction": (
            sum(gain > 0.0 for gain in paired_bce_task_gains)
            / len(paired_bce_task_gains)
        )
        >= float(gate["minimum_positive_task_fraction"]),
    }
