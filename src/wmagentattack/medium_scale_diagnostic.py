"""Frozen aggregation and gate logic for the v32 medium-scale diagnostic."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def task_surface(
    rows: Sequence[Mapping[str, Any]], control: str, horizons: set[int]
) -> dict[tuple[int, int, str], dict[str, float]]:
    grouped: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("control") != control or int(row.get("horizon", -1)) not in horizons:
            continue
        key = (int(row["training_seed"]), int(row["horizon"]), str(row["task_name"]))
        grouped[key]["nll"].append(float(row["action_nll"]))
        grouped[key]["accuracy"].append(float(row["action_correct"]))
        grouped[key]["legal"].append(float(row["legal_prediction"]))
    return {
        key: {name: float(np.mean(values)) for name, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def paired_gain(
    baseline: Mapping[tuple[int, int, str], Mapping[str, float]],
    candidate: Mapping[tuple[int, int, str], Mapping[str, float]],
    metric: str,
    *,
    higher_is_better: bool,
) -> tuple[list[float], dict[int, list[float]], dict[str, list[float]], float]:
    keys = sorted(set(baseline) & set(candidate))
    denominator = max(len(set(baseline) | set(candidate)), 1)
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_task: dict[str, list[float]] = defaultdict(list)
    values = []
    for key in keys:
        if higher_is_better:
            value = float(candidate[key][metric]) - float(baseline[key][metric])
        else:
            value = float(baseline[key][metric]) - float(candidate[key][metric])
        values.append(value)
        by_seed[key[0]].append(value)
        by_task[key[2]].append(value)
    return values, by_seed, by_task, len(keys) / denominator


def effect_average(
    rows: Sequence[Mapping[str, Any]], arm: str, key: str
) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row["split_suite"] == "task_disjoint"
        and row["arm"] == arm
        and row.get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def noninferior_units(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    unit: str,
    nll_margin: float,
    rollout_margin: float,
    baseline_arm: str,
    candidate_arm: str,
) -> tuple[int, int]:
    field = "fold_marker" if unit == "fold" else "seed"
    baseline: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    candidate: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rows, arm, target in (
        (baseline_rows, baseline_arm, baseline),
        (candidate_rows, candidate_arm, candidate),
    ):
        for row in rows:
            if row["split_suite"] != "task_disjoint" or row["arm"] != arm:
                continue
            target[int(row[field])]["nll"].append(float(row["hard_positive_task_macro_nll"]))
            target[int(row[field])]["rollout"].append(float(row["v19_rollout_hard_bce"]))
    common = sorted(set(baseline) & set(candidate))
    passed = 0
    for key in common:
        passed += int(
            np.mean(candidate[key]["nll"]) <= np.mean(baseline[key]["nll"]) + nll_margin
            and np.mean(candidate[key]["rollout"])
            <= np.mean(baseline[key]["rollout"]) + rollout_margin
        )
    return passed, len(common)


def evaluate_medium_scale_gate(
    *,
    action_baseline_rows: Sequence[Mapping[str, Any]],
    action_candidate_rows: Sequence[Mapping[str, Any]],
    effect_baseline_rows: Sequence[Mapping[str, Any]],
    effect_candidate_rows: Sequence[Mapping[str, Any]],
    training_metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_h1 = task_surface(action_baseline_rows, "free_latent_residual", {1})
    candidate_h1 = task_surface(action_candidate_rows, "free_latent_residual", {1})
    baseline_multi = task_surface(action_baseline_rows, "free_latent_residual", {2, 3, 4, 5})
    candidate_multi = task_surface(action_candidate_rows, "free_latent_residual", {2, 3, 4, 5})
    h1_nll, _, _, h1_coverage = paired_gain(
        baseline_h1, candidate_h1, "nll", higher_is_better=False
    )
    h1_accuracy, _, _, _ = paired_gain(
        baseline_h1, candidate_h1, "accuracy", higher_is_better=True
    )
    multi_nll, by_seed, by_task, multi_coverage = paired_gain(
        baseline_multi, candidate_multi, "nll", higher_is_better=False
    )
    legal = [
        float(row["legal_prediction"])
        for row in action_candidate_rows
        if row.get("control") == "free_latent_residual"
        and int(row.get("horizon", 99)) <= 5
    ]

    baseline_arm = str(thresholds["effect_baseline_arm"])
    candidate_arm = str(thresholds["effect_candidate_arm"])
    effect_keys = (
        "hard_task_macro_bce",
        "hard_positive_task_macro_nll",
        "hard_positive_task_macro_recall",
        "execution_brier",
        "pair_assignment_accuracy",
        "v19_rollout_hard_bce",
        "unseen_positive_recall",
    )
    effect_baseline = {
        key: effect_average(effect_baseline_rows, baseline_arm, key) for key in effect_keys
    }
    effect_candidate = {
        key: effect_average(effect_candidate_rows, candidate_arm, key) for key in effect_keys
    }
    fold_passed, fold_total = noninferior_units(
        effect_baseline_rows,
        effect_candidate_rows,
        unit="fold",
        nll_margin=float(thresholds["maximum_effect_positive_nll_degradation"]),
        rollout_margin=float(thresholds["maximum_effect_rollout_bce_degradation"]),
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
    )
    seed_passed, seed_total = noninferior_units(
        effect_baseline_rows,
        effect_candidate_rows,
        unit="seed",
        nll_margin=float(thresholds["maximum_effect_positive_nll_degradation"]),
        rollout_margin=float(thresholds["maximum_effect_rollout_bce_degradation"]),
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
    )

    counts = training_metrics["parameter_counts"]
    combined_parameters = int(
        counts["action_teacher"] + counts["action_residual"] + counts["effect_transition"]
    )
    metrics = {
        "h1_nll_gain_over_small_v22": float(np.mean(h1_nll)),
        "h1_accuracy_gain_over_small_v22": float(np.mean(h1_accuracy)),
        "h2_h5_nll_gain_over_small_v22": float(np.mean(multi_nll)),
        "h2_h5_positive_task_fraction": float(
            np.mean([np.mean(values) > 0 for values in by_task.values()])
        ),
        "h2_h5_positive_seeds": int(sum(np.mean(values) > 0 for values in by_seed.values())),
        "h1_key_coverage": h1_coverage,
        "h2_h5_key_coverage": multi_coverage,
        "effect_baseline": effect_baseline,
        "effect_candidate": effect_candidate,
        "effect_positive_nll_gain": float(
            effect_baseline["hard_positive_task_macro_nll"]
            - effect_candidate["hard_positive_task_macro_nll"]
        ),
        "effect_recall_gain": float(
            effect_candidate["hard_positive_task_macro_recall"]
            - effect_baseline["hard_positive_task_macro_recall"]
        ),
        "effect_rollout_bce_gain": float(
            effect_baseline["v19_rollout_hard_bce"]
            - effect_candidate["v19_rollout_hard_bce"]
        ),
        "effect_noninferior_folds": fold_passed,
        "effect_fold_units": fold_total,
        "effect_noninferior_seeds": seed_passed,
        "effect_seed_units": seed_total,
        "combined_parameter_count": combined_parameters,
    }
    clauses = {
        "complete_fixed_budget": int(training_metrics["completed_model_fits"])
        == int(thresholds["required_model_fits"]),
        "zero_runtime_failures": int(training_metrics["runtime_failures"]) == 0,
        "cuda_execution": str(training_metrics["device"]).startswith("cuda"),
        "medium_parameter_floor": combined_parameters
        >= int(thresholds["minimum_combined_parameters"]),
        "medium_parameter_ceiling": combined_parameters
        <= int(thresholds["maximum_combined_parameters"]),
        "paired_key_coverage": min(h1_coverage, multi_coverage)
        >= float(thresholds["minimum_paired_key_coverage"]),
        "h1_nll_noninferiority": metrics["h1_nll_gain_over_small_v22"]
        >= -float(thresholds["maximum_h1_nll_degradation"]),
        "h1_accuracy_noninferiority": metrics["h1_accuracy_gain_over_small_v22"]
        >= -float(thresholds["maximum_h1_accuracy_degradation"]),
        "h2_h5_nll_gain": metrics["h2_h5_nll_gain_over_small_v22"]
        >= float(thresholds["minimum_h2_h5_nll_gain"]),
        "h2_h5_positive_task_fraction": metrics["h2_h5_positive_task_fraction"]
        >= float(thresholds["minimum_h2_h5_positive_task_fraction"]),
        "h2_h5_seed_replication": metrics["h2_h5_positive_seeds"]
        >= int(thresholds["minimum_h2_h5_positive_seeds"]),
        "all_predictions_legal": bool(legal) and min(legal) == 1.0,
        "effect_positive_nll_noninferiority": metrics["effect_positive_nll_gain"]
        >= -float(thresholds["maximum_effect_positive_nll_degradation"]),
        "effect_recall_noninferiority": metrics["effect_recall_gain"]
        >= -float(thresholds["maximum_effect_recall_degradation"]),
        "effect_rollout_noninferiority": metrics["effect_rollout_bce_gain"]
        >= -float(thresholds["maximum_effect_rollout_bce_degradation"]),
        "effect_fold_replication": fold_passed
        >= int(thresholds["minimum_effect_noninferior_folds"]),
        "effect_seed_replication": seed_passed
        >= int(thresholds["minimum_effect_noninferior_seeds"]),
    }
    clauses = {key: bool(value) for key, value in clauses.items()}
    decision = (
        "GO_MEDIUM_SCALE_CAPACITY_V32"
        if all(clauses.values())
        else "NO_GO_MEDIUM_SCALE_CAPACITY_V32"
    )
    return {
        "schema_version": "wmagentattack.medium_scale_gate.v32",
        "decision": decision,
        "clauses": clauses,
        "passed": sum(clauses.values()),
        "total": len(clauses),
        "metrics": metrics,
        "authorization": {
            "retain_medium_checkpoint": decision == "GO_MEDIUM_SCALE_CAPACITY_V32",
            "formal_large_scale_training": False,
            "attack_generation": False,
            "planner_or_dreamer": False,
        },
        "counterevidence": {
            "closed_vocabulary_unseen_recall_is_diagnostic_only": True,
            "capacity_cannot_repair_missing_entity_occurrence_binding": True,
            "v31_representation_no_go_remains_in_force": True,
        },
    }
