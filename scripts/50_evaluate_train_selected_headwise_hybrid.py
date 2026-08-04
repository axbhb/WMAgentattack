"""Select attack and utility ordering heads on grouped-train OOF predictions.

This is a second, explicitly post-0713 development round.  Architecture and
hyperparameter choices are made from grouped training tasks only.  The already
inspected grouped validation split is used as external development evidence,
not as a fresh confirmatory test.  A future replay is allowed only if the frozen
gate in this script passes before any new outcomes are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]


def _load_base_module():
    path = ROOT / "scripts" / "49_evaluate_grouped_train_hybrid.py"
    spec = importlib.util.spec_from_file_location("grouped_hybrid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import grouped hybrid helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
CALIBRATION_ALPHAS = (0.25, 0.5, 0.75)


def _groups(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(["::".join(BASE._task_key(row)) for row in rows])


def _pairwise_oof(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    folds: int = 5,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    predictions = np.full(len(labels), np.nan)
    for train, valid in splitter.split(matrix, labels, groups):
        predictions[valid], _ = BASE._pairwise_rank_predict(
            matrix[train],
            labels[train],
            groups[train],
            matrix[valid],
            c_value=c_value,
        )
    if np.isnan(predictions).any():
        raise AssertionError("Pairwise OOF predictions are incomplete")
    return predictions


def _pointwise_oof(
    matrices: dict[str, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    groups: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    attack, utility, utility_weight = labels
    output = {}
    for feature_set, matrix in matrices.items():
        output[feature_set] = {
            "attack": BASE._grouped_oof_probability(
                matrix,
                attack,
                np.ones(len(attack)),
                groups,
            ),
            "utility": BASE._grouped_oof_probability(
                matrix,
                utility,
                utility_weight,
                groups,
            ),
        }
    return output


def _raw_heads(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    clean_attack, _ = BASE._raw_scores(
        rows, "clean_prefix_rollout", "risk_score"
    )
    clean_utility, _ = BASE._raw_scores(
        rows, "clean_prefix_rollout", "utility_score"
    )
    injection_attack, injection_attack_std = BASE._raw_scores(
        rows, "injection_conditioned_rollout", "risk_score"
    )
    injection_utility, _ = BASE._raw_scores(
        rows, "injection_conditioned_rollout", "utility_score"
    )
    return {
        "clean_attack": clean_attack,
        "clean_utility": clean_utility,
        "injection_attack": injection_attack,
        "injection_attack_lcb": injection_attack - injection_attack_std,
        "injection_utility": injection_utility,
        "injection_attack_borda": BASE._borda_attack(rows),
    }


def _head_candidates_oof(
    rows: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    pointwise: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    attack, utility, _ = labels
    groups = _groups(rows)
    raw = _raw_heads(rows)
    pairwise: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for feature_set in ("text", "world", "combined"):
        for c_value in BASE.C_VALUES:
            pairwise[(feature_set, c_value)] = {
                "attack": _pairwise_oof(
                    matrices[feature_set],
                    attack,
                    groups,
                    c_value=c_value,
                ),
                "utility": _pairwise_oof(
                    matrices[feature_set],
                    utility,
                    groups,
                    c_value=c_value,
                ),
            }

    attack_candidates = {
        "clean_raw": raw["clean_attack"],
        "dual_raw": raw["injection_attack"],
        "dual_lcb_1std": raw["injection_attack_lcb"],
        "dual_borda": raw["injection_attack_borda"],
        "text_pointwise": pointwise["text"]["attack"],
        "world_pointwise": pointwise["world"]["attack"],
        "combined_pointwise": pointwise["combined"]["attack"],
    }
    utility_candidates = {
        "clean_raw": raw["clean_utility"],
        "injection_raw": raw["injection_utility"],
        "text_pointwise": pointwise["text"]["utility"],
        "world_pointwise": pointwise["world"]["utility"],
        "combined_pointwise": pointwise["combined"]["utility"],
    }
    for feature_set in ("text", "world", "combined"):
        for c_value in BASE.C_VALUES:
            name = f"{feature_set}_pairwise_c{BASE._c_name(c_value)}"
            attack_candidates[name] = pairwise[(feature_set, c_value)]["attack"]
            utility_candidates[name] = pairwise[(feature_set, c_value)]["utility"]
    text_rank = BASE._within_task_rank(rows, pointwise["text"]["attack"])
    for alpha in BASE.BLEND_ALPHAS:
        name = f"text_borda_alpha_{str(alpha).replace('.', 'p')}"
        attack_candidates[name] = (
            (1.0 - alpha) * text_rank
            + alpha * raw["injection_attack_borda"]
        )
    return attack_candidates, utility_candidates


def _head_candidates_full(
    train_rows: list[dict[str, Any]],
    predict_rows: list[dict[str, Any]],
    train_matrices: dict[str, np.ndarray],
    predict_matrices: dict[str, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    pointwise: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    attack, utility, _ = labels
    groups = _groups(train_rows)
    raw = _raw_heads(predict_rows)
    attack_candidates = {
        "clean_raw": raw["clean_attack"],
        "dual_raw": raw["injection_attack"],
        "dual_lcb_1std": raw["injection_attack_lcb"],
        "dual_borda": raw["injection_attack_borda"],
        "text_pointwise": pointwise["text"]["attack"],
        "world_pointwise": pointwise["world"]["attack"],
        "combined_pointwise": pointwise["combined"]["attack"],
    }
    utility_candidates = {
        "clean_raw": raw["clean_utility"],
        "injection_raw": raw["injection_utility"],
        "text_pointwise": pointwise["text"]["utility"],
        "world_pointwise": pointwise["world"]["utility"],
        "combined_pointwise": pointwise["combined"]["utility"],
    }
    for feature_set in ("text", "world", "combined"):
        for c_value in BASE.C_VALUES:
            name = f"{feature_set}_pairwise_c{BASE._c_name(c_value)}"
            attack_candidates[name], _ = BASE._pairwise_rank_predict(
                train_matrices[feature_set],
                attack,
                groups,
                predict_matrices[feature_set],
                c_value=c_value,
            )
            utility_candidates[name], _ = BASE._pairwise_rank_predict(
                train_matrices[feature_set],
                utility,
                groups,
                predict_matrices[feature_set],
                c_value=c_value,
            )
    text_rank = BASE._within_task_rank(
        predict_rows, pointwise["text"]["attack"]
    )
    for alpha in BASE.BLEND_ALPHAS:
        name = f"text_borda_alpha_{str(alpha).replace('.', 'p')}"
        attack_candidates[name] = (
            (1.0 - alpha) * text_rank
            + alpha * raw["injection_attack_borda"]
        )
    return attack_candidates, utility_candidates


def _head_metrics(
    rows: list[dict[str, Any]], labels: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    metrics = BASE.METRICS._within_task_metrics(rows, labels, predictions)
    task_accuracies = [
        value["pairwise_accuracy"]
        for value in metrics["per_task"].values()
        if value["pairwise_accuracy"] is not None
    ]
    return {
        "mean_task_pairwise_accuracy": (
            float(np.mean(task_accuracies)) if task_accuracies else None
        ),
        "pooled_pairwise_accuracy": metrics["pairwise_accuracy"],
        "informative_task_count": len(task_accuracies),
        "comparable_pair_count": metrics["comparable_pair_count"],
    }


def _select_head(
    metrics: dict[str, dict[str, Any]], allowed: list[str]
) -> str:
    viable = [
        name
        for name in allowed
        if metrics[name]["mean_task_pairwise_accuracy"] is not None
    ]
    if not viable:
        raise ValueError("No head candidate has informative within-task pairs")
    return max(
        viable,
        key=lambda name: (
            metrics[name]["mean_task_pairwise_accuracy"],
            metrics[name]["pooled_pairwise_accuracy"],
            name,
        ),
    )


def _probability_candidates(
    base: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output = dict(base)
    for reference in ("text", "clean_raw"):
        for alpha in CALIBRATION_ALPHAS:
            name = f"{reference}_combined_alpha_{str(alpha).replace('.', 'p')}"
            output[name] = (
                (1.0 - alpha) * base[reference] + alpha * base["combined"]
            )
    return output


def _full_pointwise(
    train_matrices: dict[str, np.ndarray],
    predict_matrices: dict[str, np.ndarray],
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    attack, utility, utility_weight = labels
    output = {}
    for feature_set in ("text", "world", "combined"):
        output[feature_set] = {
            "attack": BASE._fit_soft_probability(
                train_matrices[feature_set],
                attack,
                predict_matrices[feature_set],
                weights=np.ones(len(attack)),
            ),
            "utility": BASE._fit_soft_probability(
                train_matrices[feature_set],
                utility,
                predict_matrices[feature_set],
                weights=utility_weight,
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--eval-archive", type=Path, required=True)
    parser.add_argument("--train-steps", type=Path, required=True)
    parser.add_argument("--validation-steps", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows = BASE._load_split(args.train_archive, "train")
    validation_rows = BASE._load_split(args.eval_archive, "val")
    test_rows = BASE._load_split(args.eval_archive, "test")
    train_tasks = {BASE._task_key(row) for row in train_rows}
    validation_tasks = {BASE._task_key(row) for row in validation_rows}
    test_tasks = {BASE._task_key(row) for row in test_rows}
    if train_tasks & validation_tasks or train_tasks & test_tasks or validation_tasks & test_tasks:
        raise ValueError("Grouped train/validation/test task sets must be disjoint")

    train_labels = BASE._labels(
        train_rows, BASE._soft_targets(args.train_steps)
    )
    validation_labels = BASE._labels(
        validation_rows, BASE._soft_targets(args.validation_steps)
    )
    matrices = {}
    for split, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        text = BASE._text_matrix(rows)
        world = BASE._world_matrix(rows)
        matrices[split] = {
            "text": text,
            "world": world,
            "combined": np.column_stack([text, world]),
        }

    groups = _groups(train_rows)
    pointwise_oof = _pointwise_oof(
        matrices["train"], train_labels, groups
    )
    clean_attack, _ = BASE._raw_scores(
        train_rows, "clean_prefix_rollout", "risk_score"
    )
    clean_utility, _ = BASE._raw_scores(
        train_rows, "clean_prefix_rollout", "utility_score"
    )
    probability_oof: dict[str, dict[str, np.ndarray]] = {}
    for head, clean in (("attack", clean_attack), ("utility", clean_utility)):
        probability_oof[head] = _probability_candidates(
            {
                "clean_raw": clean,
                "text": pointwise_oof["text"][head],
                "world": pointwise_oof["world"][head],
                "combined": pointwise_oof["combined"][head],
            }
        )
    attack_y, utility_y, utility_weight = train_labels
    probability_brier = {
        "attack": {
            name: float(np.mean((prediction - attack_y) ** 2))
            for name, prediction in probability_oof["attack"].items()
        },
        "utility": {
            name: float(
                np.average(
                    (prediction - utility_y) ** 2, weights=utility_weight
                )
            )
            for name, prediction in probability_oof["utility"].items()
        },
    }
    selected_calibrators = {
        head: min(values, key=lambda name: (values[name], name))
        for head, values in probability_brier.items()
    }

    attack_oof, utility_oof = _head_candidates_oof(
        train_rows,
        matrices["train"],
        train_labels,
        pointwise_oof,
    )
    attack_metrics = {
        name: _head_metrics(train_rows, attack_y, prediction)
        for name, prediction in attack_oof.items()
    }
    utility_metrics = {
        name: _head_metrics(train_rows, utility_y, prediction)
        for name, prediction in utility_oof.items()
    }
    world_attack_candidates = [
        "dual_raw",
        "dual_lcb_1std",
        "dual_borda",
        "world_pointwise",
        "combined_pointwise",
        *[
            f"world_pairwise_c{BASE._c_name(c_value)}"
            for c_value in BASE.C_VALUES
        ],
        *[
            f"combined_pairwise_c{BASE._c_name(c_value)}"
            for c_value in BASE.C_VALUES
        ],
        *[
            f"text_borda_alpha_{str(alpha).replace('.', 'p')}"
            for alpha in BASE.BLEND_ALPHAS
        ],
    ]
    text_candidates = [
        "text_pointwise",
        *[
            f"text_pairwise_c{BASE._c_name(c_value)}"
            for c_value in BASE.C_VALUES
        ],
    ]
    utility_candidates = sorted(utility_oof)
    selected_attack = _select_head(attack_metrics, world_attack_candidates)
    selected_utility = _select_head(utility_metrics, utility_candidates)
    selected_text_attack = _select_head(attack_metrics, text_candidates)
    selected_text_utility = _select_head(utility_metrics, text_candidates)

    full_pointwise = {}
    full_heads = {}
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        full_pointwise[split] = _full_pointwise(
            matrices["train"], matrices[split], train_labels
        )
        attack_heads, utility_heads = _head_candidates_full(
            train_rows,
            rows,
            matrices["train"],
            matrices[split],
            train_labels,
            full_pointwise[split],
        )
        full_heads[split] = {"attack": attack_heads, "utility": utility_heads}

    def full_probability_candidates(split: str, head: str) -> dict[str, np.ndarray]:
        rows = validation_rows if split == "validation" else test_rows
        clean, _ = BASE._raw_scores(
            rows,
            "clean_prefix_rollout",
            "risk_score" if head == "attack" else "utility_score",
        )
        return _probability_candidates(
            {
                "clean_raw": clean,
                "text": full_pointwise[split]["text"][head],
                "world": full_pointwise[split]["world"][head],
                "combined": full_pointwise[split]["combined"][head],
            }
        )

    methods = {}
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        attack_probabilities = full_probability_candidates(split, "attack")
        utility_probabilities = full_probability_candidates(split, "utility")
        clean_attack = full_heads[split]["attack"]["clean_raw"]
        clean_utility = full_heads[split]["utility"]["clean_raw"]
        methods[split] = {
            "train_selected_headwise_world": BASE._method(
                full_heads[split]["attack"][selected_attack],
                full_heads[split]["utility"][selected_utility],
                attack_probabilities[selected_calibrators["attack"]],
                utility_probabilities[selected_calibrators["utility"]],
            ),
            "train_selected_headwise_text": BASE._method(
                full_heads[split]["attack"][selected_text_attack],
                full_heads[split]["utility"][selected_text_utility],
                full_pointwise[split]["text"]["attack"],
                full_pointwise[split]["text"]["utility"],
            ),
            "text_pointwise_fixed": BASE._method(
                full_pointwise[split]["text"]["attack"],
                full_pointwise[split]["text"]["utility"],
                full_pointwise[split]["text"]["attack"],
                full_pointwise[split]["text"]["utility"],
            ),
            "clean_raw": BASE._method(
                clean_attack, clean_utility, clean_attack, clean_utility
            ),
        }

    validation_attack, validation_utility, _ = validation_labels
    results = {
        name: BASE._method_metrics(
            validation_rows,
            validation_attack,
            validation_utility,
            method,
        )
        for name, method in methods["validation"].items()
    }
    text_counterbaseline = max(
        ("train_selected_headwise_text", "text_pointwise_fixed"),
        key=lambda name: (
            results[name]["primary_mean_within_task_pairwise_accuracy"],
            -results[name]["mean_pair_soft_brier"],
            name,
        ),
    )
    selected_name = "train_selected_headwise_world"
    comparisons = {
        f"{selected_name}__minus__clean_raw": BASE._bootstrap_difference(
            validation_rows,
            methods["validation"][selected_name],
            methods["validation"]["clean_raw"],
            validation_attack,
            validation_utility,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        f"{selected_name}__minus__{text_counterbaseline}": BASE._bootstrap_difference(
            validation_rows,
            methods["validation"][selected_name],
            methods["validation"][text_counterbaseline],
            validation_attack,
            validation_utility,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        ),
    }
    clean_comparison = comparisons[f"{selected_name}__minus__clean_raw"]
    text_comparison = comparisons[
        f"{selected_name}__minus__{text_counterbaseline}"
    ]
    selected_brier = results[selected_name]["mean_pair_soft_brier"]
    reference_brier = min(
        results["clean_raw"]["mean_pair_soft_brier"],
        results[text_counterbaseline]["mean_pair_soft_brier"],
    )
    gate_checks = {
        "pairwise_gain_vs_clean_at_least_0p03": clean_comparison[
            "pairwise_accuracy_difference"
        ]
        >= 0.03,
        "pairwise_gain_vs_strongest_text_at_least_0p03": text_comparison[
            "pairwise_accuracy_difference"
        ]
        >= 0.03,
        "pairwise_ci_lower_vs_clean_at_least_minus_0p02": clean_comparison[
            "pairwise_accuracy_difference_95ci"
        ][0]
        >= -0.02,
        "pairwise_ci_lower_vs_text_at_least_minus_0p02": text_comparison[
            "pairwise_accuracy_difference_95ci"
        ][0]
        >= -0.02,
        "brier_within_0p01_of_best_reference": selected_brier
        <= reference_brier + 0.01,
        "at_least_6_informative_tasks": min(
            clean_comparison["informative_pairwise_task_count"],
            text_comparison["informative_pairwise_task_count"],
        )
        >= 6,
    }
    gate_status = "GO" if all(gate_checks.values()) else "NO-GO"

    protocol_core = {
        "scope": "train_oof_selected_headwise_world_text_hybrid",
        "development_status": "post_0713_hypothesis_round",
        "fresh_confirmation_claim": False,
        "train_task_count": len(train_tasks),
        "validation_task_count": len(validation_tasks),
        "test_task_count": len(test_tasks),
        "task_overlap": 0,
        "world_attack_candidate_budget": len(world_attack_candidates),
        "utility_candidate_budget": len(utility_candidates),
        "text_head_candidate_budget": len(text_candidates),
        "probability_candidate_budget_per_head": len(probability_oof["attack"]),
        "selection_data": "grouped-train five-fold user-task OOF only",
        "selected_attack_head": selected_attack,
        "selected_utility_head": selected_utility,
        "selected_text_attack_head": selected_text_attack,
        "selected_text_utility_head": selected_text_utility,
        "selected_probability_calibrators": selected_calibrators,
        "validation_text_counterbaseline": text_counterbaseline,
        "gate_status": gate_status,
        "gate_checks": gate_checks,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        **protocol_core,
        "protocol_sha256": protocol_hash,
        "train_attack_head_metrics": attack_metrics,
        "train_utility_head_metrics": utility_metrics,
        "train_probability_brier": probability_brier,
        "validation_results": results,
        "validation_comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("headwise_hybrid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("frozen_protocol.json").write_text(
        json.dumps({**protocol_core, "protocol_sha256": protocol_hash}, indent=2),
        encoding="utf-8",
    )
    test_pairs = []
    for index, row in enumerate(test_rows):
        test_pairs.append(
            {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "trajectory_id": row["trajectory_id"],
                "source_trace": row["source_trace"],
                "selected_world": BASE._serialize_method(
                    methods["test"][selected_name], index
                ),
                "text_counterbaseline": BASE._serialize_method(
                    methods["test"][text_counterbaseline], index
                ),
                "clean_raw": BASE._serialize_method(
                    methods["test"]["clean_raw"], index
                ),
            }
        )
    args.output_dir.joinpath("test_predictions_label_blind.json").write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "labels_included": False,
                "pairs": test_pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
