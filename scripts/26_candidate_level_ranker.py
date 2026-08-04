"""Leakage-safe candidate/trajectory-level ranking over cached imaginations.

This module deliberately keeps the world model frozen. It learns small
post-hoc scorers from validation candidates only and emits:

* grouped out-of-fold (OOF) predictions for validation selection;
* full-validation-fit predictions for the untouched test candidates;
* separate attack, utility, and conditional-preservation estimates.

The output keeps the original candidate schema and adds ``candidate_*``
scores consumed by ``18_pareto_utility_selection.py``. No task identifiers,
trajectory identifiers, labels, or source paths are model features.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_KEYS = ("observed_security", "observed_utility")
SCALAR_FEATURE_KEYS = (
    "risk_score",
    "rollout_mean_risk_score",
    "utility_score",
    "selection_utility_score",
    "preservation_score",
    "min_utility_score",
    "final_utility_score",
    "value_score",
    "reward_score",
    "target_skill_probability",
    "rollout_mean_target_skill_probability",
    "rollout_target_reached",
    "rollout_branch_count",
    "selection_score",
)
BRANCH_FEATURE_KEYS = (
    "risk_score",
    "utility_score",
    "preservation_score",
    "final_utility_score",
    "value_score",
    "reward_score",
    "target_skill_probability",
    "rollout_target_reached",
)
ENSEMBLE_FEATURE_KEYS = (
    "risk_score",
    "utility_score",
    "preservation_score",
    "final_utility_score",
    "value_score",
    "target_skill_probability",
)
SUITES = ("banking", "slack", "travel", "workspace")


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate source name: {name!r}")
        output[name] = Path(raw_path).expanduser()
    if not output:
        raise ValueError("At least one --source NAME=PATH is required")
    return output


def _parse_fixed_source_seeds(values: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=SEED, got: {value}")
        name, raw_seed = value.split("=", 1)
        output[name.strip()] = int(raw_seed)
    return output


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _group_key(row: dict[str, Any]) -> str:
    return f"{row['suite']}::{row['user_task_id']}"


def _safe_float(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    return float("nan")


def _load_clean_rates(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = {
        (str(row["suite"]), str(row["user_task_id"])): float(
            row["base_success_rate"]
        )
        for row in payload.get("tasks", [])
    }
    if not rates:
        raise ValueError(f"No task solvability rates found in {path}")
    return rates


def _load_candidate_payload(
    root: Path,
    seed: int,
    split: str,
    fixed_seed: int | None = None,
) -> dict[str, Any]:
    source_seed = seed if fixed_seed is None else fixed_seed
    path = root / f"seed{source_seed}_{split}_candidates.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate list missing or empty in {path}")
    return payload


def _align_source_rows(
    source_candidates: dict[str, list[dict[str, Any]]],
    primary_source: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if primary_source not in source_candidates:
        raise ValueError(f"Unknown primary source: {primary_source}")
    primary = source_candidates[primary_source]
    primary_keys = [_pair_key(row) for row in primary]
    if len(primary_keys) != len(set(primary_keys)):
        raise ValueError(f"Duplicate pair in primary source {primary_source}")

    aligned: dict[str, list[dict[str, Any]]] = {}
    for name, rows in source_candidates.items():
        mapping = {_pair_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate pair in source {name}")
        missing = set(primary_keys) - set(mapping)
        extra = set(mapping) - set(primary_keys)
        if missing or extra:
            raise ValueError(
                f"Candidate mismatch for {name}: missing={len(missing)} "
                f"extra={len(extra)}"
            )
        ordered = [mapping[key] for key in primary_keys]
        for reference, candidate in zip(primary, ordered, strict=True):
            if any(
                bool(reference[label]) != bool(candidate[label])
                for label in LABEL_KEYS
            ):
                raise ValueError(f"Label mismatch for {name} at {_pair_key(reference)}")
        aligned[name] = ordered
    return primary, aligned


def _branch_features(row: dict[str, Any], prefix: str) -> dict[str, float]:
    branches = row.get("rollout_top_branch_summaries")
    branches = branches if isinstance(branches, list) else []
    output: dict[str, float] = {f"{prefix}branch_count": float(len(branches))}
    for key in BRANCH_FEATURE_KEYS:
        values = np.asarray(
            [
                _safe_float(branch.get(key))
                for branch in branches
                if isinstance(branch, dict)
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        for statistic in ("mean", "std", "min", "max"):
            feature = f"{prefix}branch_{key}_{statistic}"
            if not len(values):
                output[feature] = float("nan")
            elif statistic == "mean":
                output[feature] = float(values.mean())
            elif statistic == "std":
                output[feature] = float(values.std())
            elif statistic == "min":
                output[feature] = float(values.min())
            else:
                output[feature] = float(values.max())
    return output


def _skill_path_features(row: dict[str, Any], prefix: str) -> dict[str, float]:
    skills = row.get("rollout_imagined_skills")
    skills = [str(item) for item in skills] if isinstance(skills, list) else []
    target = str(row.get("target_skill", ""))
    length = len(skills)
    denominator = max(1, length)
    return {
        f"{prefix}path_length": float(length),
        f"{prefix}path_unique_fraction": len(set(skills)) / denominator,
        f"{prefix}path_target_fraction": sum(skill == target for skill in skills)
        / denominator,
        f"{prefix}path_finish_fraction": sum(skill == "finish" for skill in skills)
        / denominator,
        f"{prefix}path_external_fraction": sum(
            "external" in skill for skill in skills
        )
        / denominator,
        f"{prefix}first_skill_is_target": float(bool(skills) and skills[0] == target),
    }


def _feature_dicts(
    primary: list[dict[str, Any]],
    aligned: dict[str, list[dict[str, Any]]],
    clean_rates: dict[tuple[str, str], float],
) -> tuple[np.ndarray, list[str]]:
    feature_rows: list[dict[str, float]] = []
    source_names = list(aligned)
    for index, primary_row in enumerate(primary):
        features: dict[str, float] = {}
        for name in source_names:
            row = aligned[name][index]
            prefix = f"{name}__"
            for key in SCALAR_FEATURE_KEYS:
                features[f"{prefix}{key}"] = _safe_float(row.get(key))
            features.update(_branch_features(row, prefix))
            features.update(_skill_path_features(row, prefix))

        for key in ENSEMBLE_FEATURE_KEYS:
            values = np.asarray(
                [_safe_float(aligned[name][index].get(key)) for name in source_names],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            features[f"ensemble__{key}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            features[f"ensemble__{key}_std"] = (
                float(values.std()) if len(values) else float("nan")
            )
            features[f"ensemble__{key}_range"] = (
                float(values.max() - values.min())
                if len(values)
                else float("nan")
            )

        clean_rate = clean_rates.get(
            (str(primary_row["suite"]), str(primary_row["user_task_id"]))
        )
        features["clean__base_success_rate"] = (
            float(clean_rate) if clean_rate is not None else float("nan")
        )
        features["clean__uncertainty"] = (
            4.0 * float(clean_rate) * (1.0 - float(clean_rate))
            if clean_rate is not None
            else float("nan")
        )
        for suite in SUITES:
            features[f"suite__{suite}"] = float(primary_row["suite"] == suite)
        feature_rows.append(features)

    feature_names = sorted({key for row in feature_rows for key in row})
    matrix = np.asarray(
        [[row.get(name, float("nan")) for name in feature_names] for row in feature_rows],
        dtype=float,
    )
    return matrix, feature_names


def _make_group_folds(
    joint_labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two task groups are required for OOF prediction")
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    output = list(splitter.split(np.zeros(len(groups)), joint_labels, groups))
    seen = np.zeros(len(groups), dtype=int)
    for train_index, valid_index in output:
        if set(groups[train_index]) & set(groups[valid_index]):
            raise AssertionError("Task group leaked across an OOF fold")
        seen[valid_index] += 1
    if not np.all(seen == 1):
        raise AssertionError("OOF folds did not cover each candidate exactly once")
    return output


def _constant_probability(labels: np.ndarray) -> float:
    return float((labels.sum() + 1.0) / (len(labels) + 2.0))


def _pointwise_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float,
    random_state: int,
) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(predict_x), _constant_probability(train_y), dtype=float)
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict_proba(predict_x)[:, 1], dtype=float)


def _pairwise_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float,
    random_state: int,
    max_pairs: int,
) -> np.ndarray:
    positives = np.flatnonzero(train_y == 1)
    negatives = np.flatnonzero(train_y == 0)
    if not len(positives) or not len(negatives):
        return np.full(len(predict_x), _constant_probability(train_y), dtype=float)

    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(imputer.fit_transform(train_x))
    transformed_predict = scaler.transform(imputer.transform(predict_x))

    pair_indices = np.asarray(
        [(positive, negative) for positive in positives for negative in negatives],
        dtype=int,
    )
    if len(pair_indices) > max_pairs:
        rng = np.random.default_rng(random_state)
        pair_indices = pair_indices[
            rng.choice(len(pair_indices), size=max_pairs, replace=False)
        ]
    differences = (
        transformed_train[pair_indices[:, 0]]
        - transformed_train[pair_indices[:, 1]]
    )
    pair_x = np.concatenate([differences, -differences], axis=0)
    pair_y = np.concatenate(
        [np.ones(len(differences), dtype=int), np.zeros(len(differences), dtype=int)]
    )
    model = LogisticRegression(
        C=c_value,
        fit_intercept=False,
        max_iter=3000,
        random_state=random_state,
        solver="liblinear",
    )
    model.fit(pair_x, pair_y)
    return expit(model.decision_function(transformed_predict))


def _ordinal_pairwise_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float,
    random_state: int,
    max_pairs: int,
) -> np.ndarray:
    """Rank candidates by the direct ASR+BUP target in {0, 1, 2}."""

    if len(np.unique(train_y)) < 2:
        return np.full(len(predict_x), float(np.mean(train_y)), dtype=float)
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(imputer.fit_transform(train_x))
    transformed_predict = scaler.transform(imputer.transform(predict_x))

    ordered_pairs = []
    pair_weights = []
    for left in range(len(train_y)):
        for right in range(left + 1, len(train_y)):
            difference = float(train_y[left] - train_y[right])
            if difference == 0.0:
                continue
            high, low = (left, right) if difference > 0 else (right, left)
            ordered_pairs.append((high, low))
            pair_weights.append(abs(difference))
    if not ordered_pairs:
        return np.full(len(predict_x), float(np.mean(train_y)), dtype=float)

    pair_indices = np.asarray(ordered_pairs, dtype=int)
    weights = np.asarray(pair_weights, dtype=float)
    if len(pair_indices) > max_pairs:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(pair_indices), size=max_pairs, replace=False)
        pair_indices = pair_indices[chosen]
        weights = weights[chosen]
    differences = (
        transformed_train[pair_indices[:, 0]]
        - transformed_train[pair_indices[:, 1]]
    )
    pair_x = np.concatenate([differences, -differences], axis=0)
    pair_y = np.concatenate(
        [np.ones(len(differences), dtype=int), np.zeros(len(differences), dtype=int)]
    )
    pair_weights_array = np.concatenate([weights, weights], axis=0)
    model = LogisticRegression(
        C=c_value,
        fit_intercept=False,
        max_iter=3000,
        random_state=random_state,
        solver="liblinear",
    )
    model.fit(pair_x, pair_y, sample_weight=pair_weights_array)
    return 2.0 * expit(model.decision_function(transformed_predict))


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float,
) -> np.ndarray:
    """Regularized direct regression for the expected ASR+BUP contribution."""

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0 / c_value)),
        ]
    )
    class_counts = Counter(train_y.tolist())
    sample_weight = np.asarray(
        [len(train_y) / (len(class_counts) * class_counts[value]) for value in train_y],
        dtype=float,
    )
    model.fit(train_x, train_y, model__sample_weight=sample_weight)
    return np.clip(np.asarray(model.predict(predict_x), dtype=float), 0.0, 2.0)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    estimator: str,
    c_value: float,
    random_state: int,
    max_pairs: int,
) -> np.ndarray:
    if estimator == "pointwise":
        return _pointwise_predict(
            train_x,
            train_y,
            predict_x,
            c_value=c_value,
            random_state=random_state,
        )
    if estimator == "pairwise":
        return _pairwise_predict(
            train_x,
            train_y,
            predict_x,
            c_value=c_value,
            random_state=random_state,
            max_pairs=max_pairs,
        )
    if estimator == "ordinal_pairwise":
        return _ordinal_pairwise_predict(
            train_x,
            train_y,
            predict_x,
            c_value=c_value,
            random_state=random_state,
            max_pairs=max_pairs,
        )
    if estimator == "ridge":
        return _ridge_predict(
            train_x,
            train_y,
            predict_x,
            c_value=c_value,
        )
    raise ValueError(f"Unsupported estimator: {estimator}")


def _oof_and_test_predictions(
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    fit_mask: np.ndarray | None,
    estimator: str,
    c_value: float,
    random_state: int,
    max_pairs: int,
    test_prediction_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(len(val_y), dtype=bool) if fit_mask is None else fit_mask
    oof = np.full(len(val_y), np.nan, dtype=float)
    fold_ids = np.full(len(val_y), -1, dtype=int)
    crossfit_test_predictions = []
    for fold_id, (train_index, valid_index) in enumerate(folds):
        fit_index = train_index[mask[train_index]]
        if not len(fit_index):
            raise ValueError(f"No eligible training rows in fold {fold_id}")
        combined_predict_x = np.concatenate([val_x[valid_index], test_x], axis=0)
        combined_predictions = _fit_predict(
            val_x[fit_index],
            val_y[fit_index],
            combined_predict_x,
            estimator=estimator,
            c_value=c_value,
            random_state=random_state + fold_id,
            max_pairs=max_pairs,
        )
        oof[valid_index] = combined_predictions[: len(valid_index)]
        crossfit_test_predictions.append(combined_predictions[len(valid_index) :])
        fold_ids[valid_index] = fold_id
    if not np.all(np.isfinite(oof)) or np.any(fold_ids < 0):
        raise AssertionError("OOF prediction is incomplete")

    if test_prediction_mode == "crossfit_ensemble":
        test = np.mean(np.stack(crossfit_test_predictions, axis=0), axis=0)
    elif test_prediction_mode == "full_fit":
        full_fit_index = np.flatnonzero(mask)
        test = _fit_predict(
            val_x[full_fit_index],
            val_y[full_fit_index],
            test_x,
            estimator=estimator,
            c_value=c_value,
            random_state=random_state + 1000,
            max_pairs=max_pairs,
        )
    else:
        raise ValueError(f"Unsupported test prediction mode: {test_prediction_mode}")
    return oof, test, fold_ids


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    active = np.ones(len(labels), dtype=bool) if mask is None else mask
    y = labels[active]
    p = np.clip(scores[active], 1e-6, 1.0 - 1e-6)
    output: dict[str, Any] = {
        "count": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
    }
    if len(y) and len(np.unique(y)) > 1:
        output["roc_auc"] = float(roc_auc_score(y, p))
        output["average_precision"] = float(average_precision_score(y, p))
    else:
        output["roc_auc"] = None
        output["average_precision"] = None
    return output


def _annotate_candidates(
    rows: list[dict[str, Any]],
    *,
    risk_scores: np.ndarray,
    utility_scores: np.ndarray,
    preservation_scores: np.ndarray,
    joint_scores: np.ndarray,
    clean_rates: dict[tuple[str, str], float],
    fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        clean_rate = clean_rates.get((str(row["suite"]), str(row["user_task_id"])))
        expected_utility = (
            float(clean_rate) * float(preservation_scores[index])
            if clean_rate is not None
            else float(utility_scores[index])
        )
        risk = float(risk_scores[index])
        utility = float(utility_scores[index])
        joint = float(joint_scores[index])
        output.append(
            {
                **row,
                "base_selection_score": row.get("selection_score"),
                "candidate_risk_score": risk,
                "candidate_utility_score": utility,
                "candidate_preservation_score": float(preservation_scores[index]),
                "candidate_expected_utility_score": expected_utility,
                "candidate_marginal_sum_score": risk + utility,
                "candidate_joint_score": joint,
                "candidate_conservative_joint_score": risk + expected_utility,
                "candidate_objective_score": risk,
                "candidate_ranker_fold": int(fold_ids[index]),
                # The standard weighted baseline evaluates the selected
                # direct or marginal estimate of expected ASR+BUP.
                "selection_score": joint,
            }
        )
    return output


def _write_payload(
    original: dict[str, Any],
    candidates: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    payload = {
        **original,
        "candidate_ranker": metadata,
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Candidate source as NAME=REPORT_ROOT; repeat for stacking.",
    )
    parser.add_argument("--primary-source", required=True)
    parser.add_argument(
        "--fixed-source-seed",
        action="append",
        default=[],
        help="Optional NAME=SEED for a source shared across model seeds.",
    )
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--min-base-success-rate", type=float, default=0.5)
    parser.add_argument("--estimator", choices=("pointwise", "pairwise"), required=True)
    parser.add_argument(
        "--joint-estimator",
        choices=("marginal_sum", "ordinal_pairwise", "ridge"),
        default="marginal_sum",
    )
    parser.add_argument(
        "--test-prediction-mode",
        choices=("full_fit", "crossfit_ensemble"),
        default="full_fit",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=20260710)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_roots = _parse_named_paths(args.source)
    fixed_source_seeds = _parse_fixed_source_seeds(args.fixed_source_seed)
    unknown_fixed = set(fixed_source_seeds) - set(source_roots)
    if unknown_fixed:
        raise ValueError(f"Fixed seeds reference unknown sources: {sorted(unknown_fixed)}")
    if args.primary_source not in source_roots:
        raise ValueError("--primary-source must match one of the --source names")
    if not 0.0 <= args.min_base_success_rate <= 1.0:
        raise ValueError("--min-base-success-rate must be in [0, 1]")
    if args.c_value <= 0 or args.max_pairs <= 0:
        raise ValueError("--c-value and --max-pairs must be positive")

    seeds = _parse_seeds(args.seeds)
    clean_rates = _load_clean_rates(args.clean_solvability_json)
    args.output_root.mkdir(parents=True, exist_ok=True)
    round_summary: dict[str, Any] = {
        "scope": "candidate_trajectory_ranker",
        "protocol": (
            "grouped validation OOF predictions; configured leakage-safe test "
            "prediction mode; "
            "no test labels used for fitting or configuration selection"
        ),
        "estimator": args.estimator,
        "joint_estimator": args.joint_estimator,
        "test_prediction_mode": args.test_prediction_mode,
        "primary_source": args.primary_source,
        "source_roots": {name: str(path.resolve()) for name, path in source_roots.items()},
        "fixed_source_seeds": fixed_source_seeds,
        "seeds": seeds,
        "cv_folds": args.cv_folds,
        "c_value": args.c_value,
        "max_pairs": args.max_pairs,
        "random_state": args.random_state,
        "min_base_success_rate": args.min_base_success_rate,
        "per_seed": [],
    }

    for seed in seeds:
        payloads: dict[str, dict[str, dict[str, Any]]] = {"val": {}, "test": {}}
        primary_rows: dict[str, list[dict[str, Any]]] = {}
        matrices: dict[str, np.ndarray] = {}
        feature_names: dict[str, list[str]] = {}

        for split in ("val", "test"):
            source_candidates: dict[str, list[dict[str, Any]]] = {}
            for name, root in source_roots.items():
                payload = _load_candidate_payload(
                    root,
                    seed,
                    split,
                    fixed_seed=fixed_source_seeds.get(name),
                )
                payloads[split][name] = payload
                source_candidates[name] = payload["candidates"]
            primary, aligned = _align_source_rows(
                source_candidates, args.primary_source
            )
            matrix, names = _feature_dicts(primary, aligned, clean_rates)
            primary_rows[split] = primary
            matrices[split] = matrix
            feature_names[split] = names

        if feature_names["val"] != feature_names["test"]:
            raise AssertionError("Validation and test feature schemas differ")

        val_rows = primary_rows["val"]
        test_rows = primary_rows["test"]
        val_security = np.asarray(
            [int(bool(row["observed_security"])) for row in val_rows], dtype=int
        )
        val_utility = np.asarray(
            [int(bool(row["observed_utility"])) for row in val_rows], dtype=int
        )
        test_security = np.asarray(
            [int(bool(row["observed_security"])) for row in test_rows], dtype=int
        )
        test_utility = np.asarray(
            [int(bool(row["observed_utility"])) for row in test_rows], dtype=int
        )
        val_groups = np.asarray([_group_key(row) for row in val_rows])
        joint_labels = 2 * val_security + val_utility
        val_joint_target = val_security + val_utility
        folds = _make_group_folds(
            joint_labels,
            val_groups,
            n_splits=args.cv_folds,
            random_state=args.random_state + seed,
        )
        val_clean = np.asarray(
            [
                clean_rates.get((str(row["suite"]), str(row["user_task_id"])), 0.0)
                for row in val_rows
            ],
            dtype=float,
        )
        test_clean = np.asarray(
            [
                clean_rates.get((str(row["suite"]), str(row["user_task_id"])), 0.0)
                for row in test_rows
            ],
            dtype=float,
        )
        val_eligible = val_clean >= args.min_base_success_rate
        test_eligible = test_clean >= args.min_base_success_rate

        risk_val, risk_test, fold_ids = _oof_and_test_predictions(
            matrices["val"],
            val_security,
            matrices["test"],
            folds,
            fit_mask=None,
            estimator=args.estimator,
            c_value=args.c_value,
            random_state=args.random_state + seed * 10,
            max_pairs=args.max_pairs,
            test_prediction_mode=args.test_prediction_mode,
        )
        utility_val, utility_test, utility_fold_ids = _oof_and_test_predictions(
            matrices["val"],
            val_utility,
            matrices["test"],
            folds,
            fit_mask=None,
            estimator=args.estimator,
            c_value=args.c_value,
            random_state=args.random_state + seed * 10 + 1,
            max_pairs=args.max_pairs,
            test_prediction_mode=args.test_prediction_mode,
        )
        preservation_val, preservation_test, preservation_fold_ids = (
            _oof_and_test_predictions(
                matrices["val"],
                val_utility,
                matrices["test"],
                folds,
                fit_mask=val_eligible,
                estimator=args.estimator,
                c_value=args.c_value,
                random_state=args.random_state + seed * 10 + 2,
                max_pairs=args.max_pairs,
                test_prediction_mode=args.test_prediction_mode,
            )
        )
        if args.joint_estimator == "marginal_sum":
            joint_val = risk_val + utility_val
            joint_test = risk_test + utility_test
            joint_fold_ids = fold_ids
        else:
            joint_val, joint_test, joint_fold_ids = _oof_and_test_predictions(
                matrices["val"],
                val_joint_target,
                matrices["test"],
                folds,
                fit_mask=None,
                estimator=args.joint_estimator,
                c_value=args.c_value,
                random_state=args.random_state + seed * 10 + 3,
                max_pairs=args.max_pairs,
                test_prediction_mode=args.test_prediction_mode,
            )
        if not (
            np.array_equal(fold_ids, utility_fold_ids)
            and np.array_equal(fold_ids, preservation_fold_ids)
            and np.array_equal(fold_ids, joint_fold_ids)
        ):
            raise AssertionError("Target heads used inconsistent OOF folds")

        metadata = {
            "estimator": args.estimator,
            "joint_estimator": args.joint_estimator,
            "test_prediction_mode": args.test_prediction_mode,
            "primary_source": args.primary_source,
            "sources": list(source_roots),
            "seed": seed,
            "feature_count": len(feature_names["val"]),
            "cv_folds": len(folds),
            "c_value": args.c_value,
            "max_pairs": args.max_pairs,
            "random_state": args.random_state,
        }
        val_annotated = _annotate_candidates(
            val_rows,
            risk_scores=risk_val,
            utility_scores=utility_val,
            preservation_scores=preservation_val,
            joint_scores=joint_val,
            clean_rates=clean_rates,
            fold_ids=fold_ids,
        )
        test_annotated = _annotate_candidates(
            test_rows,
            risk_scores=risk_test,
            utility_scores=utility_test,
            preservation_scores=preservation_test,
            joint_scores=joint_test,
            clean_rates=clean_rates,
            fold_ids=np.full(len(test_rows), -1, dtype=int),
        )
        _write_payload(
            payloads["val"][args.primary_source],
            val_annotated,
            args.output_root / f"seed{seed}_val_candidates.json",
            {**metadata, "prediction_scope": "grouped_oof_validation"},
        )
        _write_payload(
            payloads["test"][args.primary_source],
            test_annotated,
            args.output_root / f"seed{seed}_test_candidates.json",
            {
                **metadata,
                "prediction_scope": args.test_prediction_mode,
            },
        )

        seed_metrics = {
            **metadata,
            "validation_candidate_count": len(val_rows),
            "test_candidate_count": len(test_rows),
            "validation_group_count": len(set(val_groups.tolist())),
            "validation_joint_label_counts": dict(
                sorted(Counter(joint_labels.tolist()).items())
            ),
            "validation": {
                "risk": _binary_metrics(val_security, risk_val),
                "utility": _binary_metrics(val_utility, utility_val),
                "conditional_preservation": _binary_metrics(
                    val_utility, preservation_val, val_eligible
                ),
            },
            "test_diagnostic": {
                "risk": _binary_metrics(test_security, risk_test),
                "utility": _binary_metrics(test_utility, utility_test),
                "conditional_preservation": _binary_metrics(
                    test_utility, preservation_test, test_eligible
                ),
            },
        }
        (args.output_root / f"seed{seed}_ranker_metrics.json").write_text(
            json.dumps(seed_metrics, indent=2), encoding="utf-8"
        )
        round_summary["per_seed"].append(seed_metrics)

    args.output_root.joinpath("candidate_ranker_summary.json").write_text(
        json.dumps(round_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(round_summary, indent=2))


if __name__ == "__main__":
    main()
