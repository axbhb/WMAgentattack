"""Clean-solvability-conditioned BUP probes for AgentDojo-v2.

The v2 dataset contains three clean runs per user task and five attacked runs
per attack configuration.  Existing preservation supervision first divides
the attacked utility posterior by the clean posterior.  This probe instead
fits attacked BUP directly, a probability uplift, or a logit residual while
keeping the clean posterior explicit.

Discovery mode reads train and validation only.  Frozen mode requires a fully
specified candidate and evaluates one held-out outer fold without retuning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROBE = _load_module(
    "semantic_probe_residual",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)
DUAL = _load_module(
    "dual_component_probe_residual",
    ROOT / "scripts" / "87_probe_v2_dual_component_value.py",
)


TARGET_KINDS = (
    "direct_probability",
    "uplift_probability",
    "logit_residual",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
UTILITY_WEIGHTS = (1.0, 2.0, 4.0)
UNCERTAINTY_PENALTIES = (0.0, 0.5, 1.0)
ATTACK_ALPHA = 10.0


def _jeffreys_posterior(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid Bernoulli count evidence")
    alpha = successes + 0.5
    beta = trials - successes + 0.5
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    return float(mean), float(variance)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _configuration_rows(steps: list[Any]) -> list[dict[str, Any]]:
    """Add clean and attacked count evidence to the existing semantic rows."""

    decision_steps = [steps[index] for index in PROBE._decision_indices(steps).values()]
    clean_by_task: dict[str, list[Any]] = defaultdict(list)
    attack_by_group: dict[str, list[Any]] = defaultdict(list)
    for step in decision_steps:
        group_id = str(step.multiseed_group_id or "")
        task_key = f"{step.domain}|{step.task_id}"
        if group_id.startswith("clean::"):
            clean_by_task[task_key].append(step)
        elif group_id.startswith("attack::"):
            attack_by_group[group_id].append(step)

    rows = PROBE._configuration_rows(steps)
    for row in rows:
        group_id = str(row["group_id"])
        task_key = str(row["task_key"])
        attacked = attack_by_group.get(group_id, [])
        clean = clean_by_task.get(task_key, [])
        if not attacked or not clean:
            raise ValueError(
                f"Missing clean or attacked evidence for {group_id}: "
                f"clean={len(clean)} attacked={len(attacked)}"
            )
        expected_attack_trials = {int(step.multiseed_trials or 0) for step in attacked}
        expected_clean_trials = {int(step.multiseed_trials or 0) for step in clean}
        if expected_attack_trials != {len(attacked)}:
            raise ValueError(f"Incomplete attacked evidence: {group_id}")
        if expected_clean_trials != {len(clean)}:
            raise ValueError(f"Incomplete clean evidence: {task_key}")

        clean_successes = sum(bool(step.task_success) for step in clean)
        utility_successes = sum(bool(step.task_success) for step in attacked)
        clean_mean, clean_variance = _jeffreys_posterior(
            clean_successes, len(clean)
        )
        utility_mean, utility_variance = _jeffreys_posterior(
            utility_successes, len(attacked)
        )
        base_rates = {float(step.base_task_success_rate or 0.0) for step in attacked}
        if len(base_rates) != 1:
            raise ValueError(f"Inconsistent clean base rate: {group_id}")
        empirical_clean = clean_successes / len(clean)
        if not math.isclose(next(iter(base_rates)), empirical_clean, abs_tol=1e-9):
            raise ValueError(f"Clean evidence does not match base rate: {group_id}")
        if not math.isclose(float(row["target_bup"]), utility_mean, abs_tol=1e-9):
            raise ValueError(f"Attacked evidence does not match BUP target: {group_id}")

        row.update(
            {
                "clean_successes": int(clean_successes),
                "clean_trials": int(len(clean)),
                "clean_probability": clean_mean,
                "clean_variance": clean_variance,
                "utility_successes": int(utility_successes),
                "utility_trials": int(len(attacked)),
                "utility_variance": utility_variance,
            }
        )
    return rows


def _target_and_variance(
    rows: list[dict[str, Any]], target_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray([float(row["clean_probability"]) for row in rows])
    clean_variance = np.asarray([float(row["clean_variance"]) for row in rows])
    utility = np.asarray([float(row["target_bup"]) for row in rows])
    utility_variance = np.asarray(
        [float(row["utility_variance"]) for row in rows]
    )
    if target_kind == "direct_probability":
        return utility, utility_variance
    if target_kind == "uplift_probability":
        return utility - clean, utility_variance + clean_variance
    if target_kind == "logit_residual":
        clean_scale = np.square(clean * (1.0 - clean))
        utility_scale = np.square(utility * (1.0 - utility))
        variance = utility_variance / np.maximum(utility_scale, 1e-8)
        variance += clean_variance / np.maximum(clean_scale, 1e-8)
        return _logit(utility) - _logit(clean), variance
    raise ValueError(target_kind)


def _precision_weights(variance: np.ndarray) -> np.ndarray:
    precision = 1.0 / np.maximum(variance, 1e-8)
    low, high = np.quantile(precision, (0.1, 0.9))
    precision = np.clip(precision, low, high)
    return precision / max(float(precision.mean()), 1e-12)


def _fit_residual_model(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    target_kind: str,
    alpha: float,
) -> dict[str, Any]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    target, target_variance = _target_and_variance(rows, target_kind)
    weights = _precision_weights(target_variance)
    feature_mean = np.average(matrix, axis=0, weights=weights)
    target_mean = float(np.average(target, weights=weights))
    centered = np.asarray(matrix, dtype=np.float64) - feature_mean
    square_root_weight = np.sqrt(weights)
    weighted_design = centered * square_root_weight[:, None]
    weighted_response = (target - target_mean) * square_root_weight
    gram = weighted_design @ weighted_design.T
    gram.flat[:: len(gram) + 1] += alpha
    dual = np.linalg.solve(gram, weighted_response)
    coefficient = weighted_design.T @ dual
    fitted = target_mean + centered @ coefficient
    residual = target - fitted
    residual_variance = max(
        float(np.average(np.square(residual), weights=weights)), 1e-6
    )
    return {
        "target_kind": target_kind,
        "alpha": float(alpha),
        "feature_mean": feature_mean,
        "target_mean": target_mean,
        "coefficient": coefficient,
        "weighted_design": weighted_design,
        "gram": gram,
        "residual_variance": residual_variance,
        "fit_summary": {
            "row_count": len(rows),
            "target_mean": target_mean,
            "target_std": float(target.std()),
            "posterior_variance_mean": float(target_variance.mean()),
            "precision_weight_min": float(weights.min()),
            "precision_weight_max": float(weights.max()),
            "weighted_residual_variance": residual_variance,
        },
    }


def _predict_residual_model(
    model: dict[str, Any],
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = np.asarray(matrix, dtype=np.float64) - model["feature_mean"]
    raw = model["target_mean"] + centered @ model["coefficient"]

    projected = model["weighted_design"] @ centered.T
    solved = np.linalg.solve(model["gram"], projected)
    leverage = np.sum(np.square(centered), axis=1) / model["alpha"]
    leverage -= np.sum(projected * solved, axis=0) / model["alpha"]
    latent_variance = model["residual_variance"] * np.maximum(leverage, 0.0)

    clean = np.asarray([float(row["clean_probability"]) for row in rows])
    clean_variance = np.asarray([float(row["clean_variance"]) for row in rows])
    if model["target_kind"] == "direct_probability":
        mean = np.clip(raw, 0.0, 1.0)
        standard_deviation = np.sqrt(latent_variance)
    elif model["target_kind"] == "uplift_probability":
        mean = np.clip(clean + raw, 0.0, 1.0)
        standard_deviation = np.sqrt(latent_variance + clean_variance)
    elif model["target_kind"] == "logit_residual":
        clean_logit_variance = clean_variance / np.maximum(
            np.square(clean * (1.0 - clean)), 1e-8
        )
        mean = _sigmoid(_logit(clean) + raw)
        eta_standard_deviation = np.sqrt(latent_variance + clean_logit_variance)
        standard_deviation = mean * (1.0 - mean) * eta_standard_deviation
    else:  # pragma: no cover - guarded during fitting
        raise ValueError(model["target_kind"])
    return mean, np.clip(standard_deviation, 0.0, 1.0), raw


def _fit_attack_model(matrix: np.ndarray, rows: list[dict[str, Any]]):
    return PROBE._ridge_fit(
        matrix,
        DUAL._component_rows(rows, "target_asr"),
        estimator="pairwise_ridge",
        alpha=ATTACK_ALPHA,
    )


def _predict_attack(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    _, prediction = PROBE._ridge_predict(model, matrix)
    return np.clip(prediction, 0.0, 1.0)


def _joint_control(
    train_matrix: np.ndarray,
    train_rows: list[dict[str, Any]],
    evaluation_matrix: np.ndarray,
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model = PROBE._ridge_fit(
        train_matrix,
        train_rows,
        estimator="pairwise_ridge",
        alpha=ATTACK_ALPHA,
    )
    rank_score, prediction = PROBE._ridge_predict(model, evaluation_matrix)
    return PROBE._evaluate(
        evaluation_rows,
        rank_scores=rank_score,
        predictions=prediction,
    )


def _evaluate_candidate(
    rows: list[dict[str, Any]],
    *,
    attack_prediction: np.ndarray,
    utility_prediction: np.ndarray,
    utility_standard_deviation: np.ndarray,
    utility_weight: float,
    uncertainty_penalty: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    conservative_utility = np.clip(
        utility_prediction - uncertainty_penalty * utility_standard_deviation,
        0.0,
        1.0,
    )
    rank_score = attack_prediction + utility_weight * conservative_utility
    joint_prediction = np.clip(attack_prediction + utility_prediction, 0.0, 2.0)
    metrics = PROBE._evaluate(
        rows,
        rank_scores=rank_score,
        predictions=joint_prediction,
    )
    return metrics, rank_score, conservative_utility


def _candidate_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["validation"]
    correlation = metrics["mean_task_spearman"]
    # Posterior means such as 7/12 can produce 1.0 versus
    # 0.9999999999999999 after task averaging.  Treat those as numerical ties
    # so the preregistered semantic tie-breaks, rather than IEEE noise, decide.
    stable = lambda value: round(float(value), 12)
    return (
        stable(metrics["top1_target_ASR_plus_BUP"]),
        stable(metrics["top1_target_BUP"]),
        stable(correlation) if correlation is not None else -math.inf,
        -stable(metrics["normalized_brier"]),
        -float(candidate["fixed_order"]),
    )


def _select_candidates(
    grid: list[dict[str, Any]], control_bup: float
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    for candidate in grid:
        candidate["BUP_constraint_pass"] = (
            float(candidate["validation"]["top1_target_BUP"]) + 1e-12
            >= control_bup
        )

    def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [row for row in rows if row["BUP_constraint_pass"]]
        pool = eligible or rows
        selected = max(pool, key=_candidate_key)
        return {**selected, "selection_used_fallback": not bool(eligible)}

    overall = select(grid)
    by_target = {
        target_kind: select(
            [row for row in grid if row["target_kind"] == target_kind]
        )
        for target_kind in TARGET_KINDS
    }
    return overall, by_target


def _core_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_kind": str(candidate["target_kind"]),
        "alpha": float(candidate["alpha"]),
        "utility_weight": float(candidate["utility_weight"]),
        "uncertainty_penalty": float(candidate["uncertainty_penalty"]),
    }


def discover(
    rows: dict[str, list[dict[str, Any]]],
    matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    attack_model = _fit_attack_model(matrices["train"], rows["train"])
    attack_prediction = _predict_attack(attack_model, matrices["val"])
    control = _joint_control(
        matrices["train"], rows["train"], matrices["val"], rows["val"]
    )

    grid = []
    order = 0
    fit_summaries = {}
    for target_kind in TARGET_KINDS:
        for alpha in ALPHAS:
            model = _fit_residual_model(
                matrices["train"],
                rows["train"],
                target_kind=target_kind,
                alpha=alpha,
            )
            utility_prediction, utility_std, _ = _predict_residual_model(
                model, matrices["val"], rows["val"]
            )
            fit_summaries[f"{target_kind}|alpha={alpha}"] = model["fit_summary"]
            for utility_weight in UTILITY_WEIGHTS:
                for uncertainty_penalty in UNCERTAINTY_PENALTIES:
                    metrics, _, _ = _evaluate_candidate(
                        rows["val"],
                        attack_prediction=attack_prediction,
                        utility_prediction=utility_prediction,
                        utility_standard_deviation=utility_std,
                        utility_weight=utility_weight,
                        uncertainty_penalty=uncertainty_penalty,
                    )
                    grid.append(
                        {
                            "target_kind": target_kind,
                            "alpha": alpha,
                            "utility_weight": utility_weight,
                            "uncertainty_penalty": uncertainty_penalty,
                            "fixed_order": order,
                            "validation": metrics,
                        }
                    )
                    order += 1
    overall, by_target = _select_candidates(
        grid, float(control["top1_target_BUP"])
    )
    return {
        "scope": "validation-only residual-preservation discovery",
        "protocol": {
            "fit_split": "train",
            "selection_split": "validation",
            "test_file_read": False,
            "target_kinds": list(TARGET_KINDS),
            "alphas": list(ALPHAS),
            "utility_weights": list(UTILITY_WEIGHTS),
            "uncertainty_penalties": list(UNCERTAINTY_PENALTIES),
            "attack_head": "pairwise_ridge_alpha_10",
            "selection_constraint": (
                "target BUP no lower than pairwise joint-control target BUP"
            ),
        },
        "counts": {
            split: {
                "configurations": len(rows[split]),
                "tasks": len({row["task_key"] for row in rows[split]}),
            }
            for split in ("train", "val")
        },
        "joint_control_validation": control,
        "grid": grid,
        "fit_summaries": fit_summaries,
        "overall_selected": overall,
        "overall_frozen_candidate": _core_candidate(overall),
        "best_by_target_kind": by_target,
        "frozen_candidates_by_target_kind": {
            key: _core_candidate(value) for key, value in by_target.items()
        },
    }


def frozen_evaluate(
    rows: dict[str, list[dict[str, Any]]],
    matrices: dict[str, np.ndarray],
    *,
    target_kind: str,
    alpha: float,
    utility_weight: float,
    uncertainty_penalty: float,
) -> dict[str, Any]:
    combined_rows = rows["train"] + rows["val"]
    combined_matrix = np.concatenate(
        (matrices["train"], matrices["val"]), axis=0
    )
    attack_model = _fit_attack_model(combined_matrix, combined_rows)
    utility_model = _fit_residual_model(
        combined_matrix,
        combined_rows,
        target_kind=target_kind,
        alpha=alpha,
    )
    attack_prediction = _predict_attack(attack_model, matrices["test"])
    utility_prediction, utility_std, utility_raw = _predict_residual_model(
        utility_model, matrices["test"], rows["test"]
    )
    metrics, rank_score, conservative_utility = _evaluate_candidate(
        rows["test"],
        attack_prediction=attack_prediction,
        utility_prediction=utility_prediction,
        utility_standard_deviation=utility_std,
        utility_weight=utility_weight,
        uncertainty_penalty=uncertainty_penalty,
    )
    frozen_candidate = {
        "target_kind": target_kind,
        "alpha": float(alpha),
        "utility_weight": float(utility_weight),
        "uncertainty_penalty": float(uncertainty_penalty),
    }
    return {
        "scope": "frozen clean-conditioned residual-preservation fold evaluation",
        "protocol": {
            "representation": "e5_structured/full",
            "attack_estimator": "pairwise_ridge",
            "attack_alpha": ATTACK_ALPHA,
            "frozen_candidate": frozen_candidate,
            "fit_scope": "train_plus_validation_after_global_freeze",
            "test_retuning": False,
            "clean_context": "three pre-attack clean task trials",
        },
        "counts": {
            split: {
                "configurations": len(rows[split]),
                "tasks": len({row["task_key"] for row in rows[split]}),
            }
            for split in ("train", "val", "test")
        },
        "utility_fit_summary": utility_model["fit_summary"],
        "test": metrics,
        "test_candidate_scores": [
            {
                "group_id": str(row["group_id"]),
                "task_key": str(row["task_key"]),
                "target": float(row["target"]),
                "target_asr": float(row["target_asr"]),
                "target_bup": float(row["target_bup"]),
                "observed": float(row["observed"]),
                "observed_asr": float(row["observed_asr"]),
                "observed_bup": float(row["observed_bup"]),
                "clean_probability": float(row["clean_probability"]),
                "clean_successes": int(row["clean_successes"]),
                "clean_trials": int(row["clean_trials"]),
                "utility_successes": int(row["utility_successes"]),
                "utility_trials": int(row["utility_trials"]),
                "rank_score": float(rank_score[index]),
                "prediction": float(
                    np.clip(
                        attack_prediction[index] + utility_prediction[index],
                        0.0,
                        2.0,
                    )
                ),
                "attack_prediction": float(attack_prediction[index]),
                "utility_prediction": float(utility_prediction[index]),
                "utility_standard_deviation": float(utility_std[index]),
                "conservative_utility": float(conservative_utility[index]),
                "utility_raw_target_space": float(utility_raw[index]),
            }
            for index, row in enumerate(rows["test"])
        ],
    }


def _build_matrices(
    rows: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return DUAL._build_matrices(
        rows,
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        domain_interactions=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=("discover", "frozen"), required=True)
    parser.add_argument("--frozen-target-kind", choices=TARGET_KINDS)
    parser.add_argument("--frozen-alpha", type=float)
    parser.add_argument("--frozen-utility-weight", type=float)
    parser.add_argument("--frozen-uncertainty-penalty", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_names = ("train", "val") if args.mode == "discover" else (
        "train",
        "val",
        "test",
    )
    rows = {
        split: _configuration_rows(PROBE._steps(args.data_root / f"{split}_steps.jsonl"))
        for split in split_names
    }
    matrix_rows = (
        {"train": rows["train"], "val": rows["val"], "test": rows["val"]}
        if args.mode == "discover"
        else rows
    )
    matrices, vocabulary = _build_matrices(
        matrix_rows,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
    )

    if args.mode == "discover":
        result = discover(rows, {"train": matrices["train"], "val": matrices["val"]})
    else:
        required = {
            "frozen_target_kind": args.frozen_target_kind,
            "frozen_alpha": args.frozen_alpha,
            "frozen_utility_weight": args.frozen_utility_weight,
            "frozen_uncertainty_penalty": args.frozen_uncertainty_penalty,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Frozen mode is missing arguments: {missing}")
        result = frozen_evaluate(
            rows,
            matrices,
            target_kind=str(args.frozen_target_kind),
            alpha=float(args.frozen_alpha),
            utility_weight=float(args.frozen_utility_weight),
            uncertainty_penalty=float(args.frozen_uncertainty_penalty),
        )
    result["provenance"] = {
        "data_root": str(args.data_root.resolve()),
        "model_name": args.model_name,
        "cache_dir": str(args.cache_dir.resolve()),
        "mode": args.mode,
        "structured_vocabulary": vocabulary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
