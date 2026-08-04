"""Evaluate injection-aware probability models with leave-user-task-out CV."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUITES = ("banking", "slack", "travel", "workspace")
CV_SEEDS_DEFAULT = (101, 211, 307, 401, 503)
MODEL_SPECS = {
    "context_ridge": ("binary_ridge", "context"),
    "text_context_multinomial": ("joint", "context_text"),
    "clean_world_ridge": ("binary_ridge", "clean_world"),
    "injection_world_ridge": ("binary_ridge", "injection_world"),
    "hierarchical_beta_binomial": ("hierarchical", "injection_world"),
    "joint_text_multinomial": ("joint", "injection_text"),
}
DEPLOYABLE_MODELS = (
    "injection_world_ridge",
    "hierarchical_beta_binomial",
    "joint_text_multinomial",
)
CLEAN_WORLD_KEYS = (
    "risk_score",
    "rollout_mean_risk_score",
    "utility_score",
    "preservation_score",
    "final_utility_score",
    "value_score",
    "target_skill_probability",
    "rollout_target_reached",
)
EPSILON = 1e-5


def _parse_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("At least one integer is required")
    return output


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _clip_probability(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            return float(np.clip(number, EPSILON, 1.0 - EPSILON))
    return float("nan")


def _logit(value: Any) -> float:
    probability = _clip_probability(value)
    if not math.isfinite(probability):
        return float("nan")
    return float(math.log(probability / (1.0 - probability)))


def _hash_category(value: str, dim: int, namespace: str) -> np.ndarray:
    output = np.zeros(dim, dtype=float)
    digest = hashlib.blake2b(
        f"{namespace}|{value}".encode("utf-8"), digest_size=8
    ).digest()
    output[int.from_bytes(digest, "little") % dim] = 1.0
    return output


def _feature_matrix(
    rows: list[dict[str, Any]],
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:
    feature_rows: list[list[float]] = []
    names = [
        *[f"suite_{suite}" for suite in SUITES],
        "clean_solvability_logit",
    ]
    for row in rows:
        feature_rows.append(
            [
                *[float(row["suite"] == suite) for suite in SUITES],
                _logit(row.get("contrast_clean_solvability")),
            ]
        )
    if feature_set == "context":
        return np.asarray(feature_rows, dtype=float), names
    if feature_set == "context_text":
        vectorizer = HashingVectorizer(
            n_features=32,
            alternate_sign=True,
            norm="l2",
            ngram_range=(1, 2),
        )
        text = vectorizer.transform(
            [str(row.get("injection_text", "")) for row in rows]
        ).toarray()
        names.extend(
            [f"injection_text_hash_{index}" for index in range(text.shape[1])]
        )
        return np.column_stack(
            [np.asarray(feature_rows, dtype=float), text]
        ), names

    clean_names = []
    for key in CLEAN_WORLD_KEYS:
        for statistic in ("mean", "std"):
            field = f"contrast_{key}_{statistic}"
            clean_names.append(field)
            for index, row in enumerate(rows):
                feature_rows[index].append(float(row.get(field, float("nan"))))
    names.extend(clean_names)
    if feature_set == "clean_world":
        return np.asarray(feature_rows, dtype=float), names

    injection_names = []
    for key in CLEAN_WORLD_KEYS:
        for statistic in ("mean", "std"):
            field = f"injection_contrast_{key}_{statistic}"
            injection_names.append(field)
            for index, row in enumerate(rows):
                feature_rows[index].append(float(row.get(field, float("nan"))))
    names.extend(injection_names)
    for key in (
        "risk_score",
        "utility_score",
        "preservation_score",
        "final_utility_score",
        "target_skill_probability",
    ):
        name = f"injection_minus_clean_{key}"
        names.append(name)
        for index, row in enumerate(rows):
            injection = float(
                row.get(f"injection_contrast_{key}_mean", float("nan"))
            )
            clean = float(row.get(f"contrast_{key}_mean", float("nan")))
            feature_rows[index].append(injection - clean)

    category_names = [
        *[f"target_hash_{index}" for index in range(8)],
        *[f"location_hash_{index}" for index in range(8)],
    ]
    names.extend(category_names)
    for index, row in enumerate(rows):
        target = _hash_category(str(row.get("target_skill", "")), 8, "target")
        locations = "|".join(str(item) for item in row.get("injection_locations", []))
        location = _hash_category(locations, 8, "location")
        feature_rows[index].extend(target.tolist())
        feature_rows[index].extend(location.tolist())
    if feature_set == "injection_world":
        return np.asarray(feature_rows, dtype=float), names
    if feature_set != "injection_text":
        raise ValueError(f"Unknown feature set: {feature_set}")

    vectorizer = HashingVectorizer(
        n_features=32,
        alternate_sign=True,
        norm="l2",
        ngram_range=(1, 2),
    )
    text = vectorizer.transform(
        [str(row.get("injection_text", "")) for row in rows]
    ).toarray()
    names.extend([f"injection_text_hash_{index}" for index in range(text.shape[1])])
    return np.column_stack([np.asarray(feature_rows, dtype=float), text]), names


def _outcome_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    attempts = []
    for row in rows:
        values = [
            (int(bool(outcome["security"])), int(bool(outcome["utility"])))
            for outcome in row["outcomes"]
        ]
        attempts.append(values)
    count = np.asarray([len(values) for values in attempts], dtype=int)
    attack = np.asarray(
        [sum(value[0] for value in values) for values in attempts], dtype=int
    )
    utility = np.asarray(
        [sum(value[1] for value in values) for values in attempts], dtype=int
    )
    return attack, utility, count, attempts


def _suite_balanced_folds(
    rows: list[dict[str, Any]],
    *,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    tasks_by_suite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in sorted({_task_key(row) for row in rows}):
        tasks_by_suite[task[0]].append(task)
    counts = {suite: len(tasks_by_suite[suite]) for suite in SUITES}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Each suite must have the same task count: {counts}")
    fold_count = next(iter(counts.values()))
    if fold_count < 2:
        raise ValueError("At least two tasks per suite are required")
    rng = np.random.default_rng(random_state)
    shuffled = {}
    for suite in SUITES:
        tasks = tasks_by_suite[suite][:]
        rng.shuffle(tasks)
        shuffled[suite] = tasks
    groups = np.asarray([_task_key(row) for row in rows], dtype=object)
    output = []
    seen = np.zeros(len(rows), dtype=int)
    for fold in range(fold_count):
        valid_tasks = {shuffled[suite][fold] for suite in SUITES}
        valid = np.asarray(
            [index for index, task in enumerate(groups) if tuple(task) in valid_tasks],
            dtype=int,
        )
        train = np.asarray(
            [index for index, task in enumerate(groups) if tuple(task) not in valid_tasks],
            dtype=int,
        )
        if {_task_key(rows[index]) for index in train} & valid_tasks:
            raise AssertionError("User task leaked across fold")
        seen[valid] += 1
        output.append((train, valid))
    if not np.all(seen == 1):
        raise AssertionError("Folds did not cover every pair once")
    return output


def _expanded_binary(
    matrix: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    indices: np.ndarray,
    head: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    labels = []
    for index in indices:
        for outcome in attempts[int(index)]:
            x_rows.append(matrix[int(index)])
            labels.append(outcome[head])
    return np.asarray(x_rows, dtype=float), np.asarray(labels, dtype=int)


def _binary_ridge_predict(
    matrix: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    train: np.ndarray,
    valid: np.ndarray,
    head: int,
) -> np.ndarray:
    train_x, train_y = _expanded_binary(matrix, attempts, train, head)
    if len(np.unique(train_y)) < 2:
        probability = (train_y.sum() + 0.5) / (len(train_y) + 1.0)
        return np.full(len(valid), probability)
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict_proba(matrix[valid])[:, 1], EPSILON, 1 - EPSILON)


def _joint_predict(
    matrix: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    labels = []
    for index in train:
        for attack, utility in attempts[int(index)]:
            x_rows.append(matrix[int(index)])
            labels.append(2 * attack + utility)
    x_array = np.asarray(x_rows, dtype=float)
    y_array = np.asarray(labels, dtype=int)
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=3000,
                ),
            ),
        ]
    )
    model.fit(x_array, y_array)
    raw = model.predict_proba(matrix[valid])
    classes = model.named_steps["model"].classes_
    joint = np.zeros((len(valid), 4), dtype=float)
    for column, label in enumerate(classes):
        joint[:, int(label)] = raw[:, column]
    joint = np.clip(joint, EPSILON, 1.0)
    joint /= joint.sum(axis=1, keepdims=True)
    attack = joint[:, 2] + joint[:, 3]
    utility = joint[:, 1] + joint[:, 3]
    return attack, utility, joint


def _beta_binomial_value_gradient(
    parameters: np.ndarray,
    matrix: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    task_index: np.ndarray,
    task_count: int,
    *,
    fixed_l2: float,
    task_l2: float,
) -> tuple[float, np.ndarray]:
    dimension = matrix.shape[1]
    coefficients = parameters[:dimension]
    effects = parameters[dimension : dimension + task_count]
    log_concentration = parameters[-1]
    concentration_exp = math.exp(float(log_concentration))
    concentration = 2.0 + concentration_exp
    eta = matrix @ coefficients + effects[task_index]
    mean = np.clip(expit(eta), EPSILON, 1.0 - EPSILON)
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    log_likelihood = (
        gammaln(trials + 1)
        - gammaln(successes + 1)
        - gammaln(trials - successes + 1)
        + betaln(successes + alpha, trials - successes + beta)
        - betaln(alpha, beta)
    )
    value = float(-log_likelihood.sum())
    value += 0.5 * fixed_l2 * float(np.sum(coefficients[1:] ** 2))
    value += 0.5 * task_l2 * float(np.sum(effects**2))
    value += 0.005 * float(log_concentration**2)

    derivative_alpha = (
        digamma(successes + alpha)
        - digamma(trials + alpha + beta)
        - digamma(alpha)
        + digamma(alpha + beta)
    )
    derivative_beta = (
        digamma(trials - successes + beta)
        - digamma(trials + alpha + beta)
        - digamma(beta)
        + digamma(alpha + beta)
    )
    derivative_mean = concentration * (derivative_alpha - derivative_beta)
    derivative_eta = derivative_mean * mean * (1.0 - mean)
    coefficient_gradient = -(matrix.T @ derivative_eta)
    coefficient_gradient[1:] += fixed_l2 * coefficients[1:]
    effect_gradient = -np.bincount(
        task_index,
        weights=derivative_eta,
        minlength=task_count,
    )
    effect_gradient += task_l2 * effects
    derivative_concentration = (
        derivative_alpha * mean + derivative_beta * (1.0 - mean)
    )
    concentration_gradient = (
        -float(derivative_concentration.sum()) * concentration_exp
        + 0.01 * log_concentration
    )
    gradient = np.concatenate(
        [
            coefficient_gradient,
            effect_gradient,
            np.asarray([concentration_gradient]),
        ]
    )
    return value, gradient


def _hierarchical_predict_compatible(
    matrix: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    rows: list[dict[str, Any]],
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Call the optimizer with a closure compatible with all SciPy versions."""
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(matrix[train]))
    valid_x = scaler.transform(imputer.transform(matrix[valid]))
    train_x = np.column_stack([np.ones(len(train_x)), train_x])
    valid_x = np.column_stack([np.ones(len(valid_x)), valid_x])
    tasks = sorted({_task_key(rows[int(index)]) for index in train})
    task_to_index = {task: index for index, task in enumerate(tasks)}
    task_index = np.asarray(
        [task_to_index[_task_key(rows[int(index)])] for index in train],
        dtype=int,
    )
    initial = np.zeros(train_x.shape[1] + len(tasks) + 1, dtype=float)
    rate = (float(successes[train].sum()) + 0.5) / (
        float(trials[train].sum()) + 1.0
    )
    initial[0] = _logit(rate)
    initial[-1] = math.log(10.0)

    def objective(parameters):
        return _beta_binomial_value_gradient(
            parameters,
            train_x,
            successes[train].astype(float),
            trials[train].astype(float),
            task_index,
            len(tasks),
            fixed_l2=1.0,
            task_l2=4.0,
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[*([(None, None)] * (len(initial) - 1)), (-4.0, 6.0)],
        options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-6},
    )
    coefficients = result.x[: train_x.shape[1]]
    prediction = np.clip(expit(valid_x @ coefficients), EPSILON, 1 - EPSILON)
    return prediction, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "concentration": float(2.0 + math.exp(float(result.x[-1]))),
        "train_task_count": len(tasks),
    }


def _probability_metrics(
    rates: np.ndarray,
    predictions: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    head: int,
) -> dict[str, Any]:
    predictions = np.clip(predictions, EPSILON, 1.0 - EPSILON)
    correlation = None
    if len(np.unique(rates)) > 1 and len(np.unique(predictions)) > 1:
        value = spearmanr(rates, predictions).statistic
        if math.isfinite(float(value)):
            correlation = float(value)
    labels = []
    expanded_predictions = []
    for index, outcomes in enumerate(attempts):
        labels.extend(outcome[head] for outcome in outcomes)
        expanded_predictions.extend([float(predictions[index])] * len(outcomes))
    labels_array = np.asarray(labels, dtype=int)
    auc = (
        float(roc_auc_score(labels_array, expanded_predictions))
        if len(np.unique(labels_array)) > 1
        else None
    )
    order = np.argsort(predictions)
    bins = [
        indices
        for indices in np.array_split(order, min(5, len(order)))
        if len(indices)
    ]
    ece = float(
        sum(
            len(indices)
            / len(order)
            * abs(float(predictions[indices].mean() - rates[indices].mean()))
            for indices in bins
        )
    )
    return {
        "pair_soft_brier": float(np.mean((predictions - rates) ** 2)),
        "pair_binomial_log_loss": float(
            -np.mean(
                rates * np.log(predictions)
                + (1.0 - rates) * np.log(1.0 - predictions)
            )
        ),
        "pair_mae": float(np.mean(np.abs(predictions - rates))),
        "pair_spearman": correlation,
        "attempt_roc_auc": auc,
        "equal_count_ece_5bin": ece,
        "observed_mean": float(rates.mean()),
        "predicted_mean": float(predictions.mean()),
    }


def _within_task_metrics(
    rows: list[dict[str, Any]],
    rates: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_task_key(row)].append(index)
    total_pairs = 0
    correct = 0.0
    correlations = []
    per_task = {}
    for task, indices_list in sorted(grouped.items()):
        indices = np.asarray(indices_list, dtype=int)
        task_rates = rates[indices]
        task_predictions = predictions[indices]
        task_total = 0
        task_correct = 0.0
        for left, right in itertools.combinations(range(len(indices)), 2):
            observed_difference = task_rates[left] - task_rates[right]
            if abs(float(observed_difference)) < 1e-12:
                continue
            predicted_difference = task_predictions[left] - task_predictions[right]
            if abs(float(predicted_difference)) < 1e-12:
                score = 0.5
            else:
                score = float(
                    np.sign(observed_difference) == np.sign(predicted_difference)
                )
            task_correct += score
            task_total += 1
        correlation = None
        if len(np.unique(task_rates)) > 1 and len(np.unique(task_predictions)) > 1:
            value = spearmanr(task_rates, task_predictions).statistic
            if math.isfinite(float(value)):
                correlation = float(value)
                correlations.append(correlation)
        total_pairs += task_total
        correct += task_correct
        per_task[f"{task[0]}::{task[1]}"] = {
            "comparable_pair_count": task_total,
            "pairwise_accuracy": task_correct / task_total if task_total else None,
            "spearman": correlation,
            "brier": float(np.mean((task_predictions - task_rates) ** 2)),
        }
    return {
        "comparable_pair_count": total_pairs,
        "pairwise_accuracy": correct / total_pairs if total_pairs else None,
        "mean_task_spearman": (
            float(np.mean(correlations)) if correlations else None
        ),
        "per_task": per_task,
    }


def _evaluate_model(
    rows: list[dict[str, Any]],
    attempts: list[list[tuple[int, int]]],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    attack_predictions: np.ndarray,
    utility_predictions: np.ndarray,
) -> dict[str, Any]:
    attack = {
        "probability": _probability_metrics(
            attack_rates, attack_predictions, attempts, 0
        ),
        "within_task": _within_task_metrics(
            rows, attack_rates, attack_predictions
        ),
    }
    utility = {
        "probability": _probability_metrics(
            utility_rates, utility_predictions, attempts, 1
        ),
        "within_task": _within_task_metrics(
            rows, utility_rates, utility_predictions
        ),
    }
    pairwise_values = [
        attack["within_task"]["pairwise_accuracy"],
        utility["within_task"]["pairwise_accuracy"],
    ]
    return {
        "attack": attack,
        "utility": utility,
        "primary_mean_within_task_pairwise_accuracy": float(
            np.mean([value for value in pairwise_values if value is not None])
        ),
        "mean_pair_soft_brier": float(
            0.5
            * (
                attack["probability"]["pair_soft_brier"]
                + utility["probability"]["pair_soft_brier"]
            )
        ),
    }


def _run_repeated_oof(
    rows: list[dict[str, Any]],
    *,
    cv_seeds: list[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    attack_count, utility_count, trials, attempts = _outcome_arrays(rows)
    attack_rates = attack_count / trials
    utility_rates = utility_count / trials
    matrices = {}
    feature_names = {}
    for feature_set in (
        "context",
        "context_text",
        "clean_world",
        "injection_world",
        "injection_text",
    ):
        matrices[feature_set], feature_names[feature_set] = _feature_matrix(
            rows, feature_set
        )
    predictions = {
        model: {
            "attack": np.full((len(cv_seeds), len(rows)), np.nan),
            "utility": np.full((len(cv_seeds), len(rows)), np.nan),
        }
        for model in MODEL_SPECS
    }
    diagnostics = []
    for repeat, seed in enumerate(cv_seeds):
        folds = _suite_balanced_folds(rows, random_state=seed)
        for fold, (train, valid) in enumerate(folds):
            for model_name, (model_type, feature_set) in MODEL_SPECS.items():
                matrix = matrices[feature_set]
                if model_type == "binary_ridge":
                    predictions[model_name]["attack"][repeat, valid] = (
                        _binary_ridge_predict(matrix, attempts, train, valid, 0)
                    )
                    predictions[model_name]["utility"][repeat, valid] = (
                        _binary_ridge_predict(matrix, attempts, train, valid, 1)
                    )
                elif model_type == "joint":
                    attack, utility, _ = _joint_predict(
                        matrix, attempts, train, valid
                    )
                    predictions[model_name]["attack"][repeat, valid] = attack
                    predictions[model_name]["utility"][repeat, valid] = utility
                elif model_type == "hierarchical":
                    attack, attack_diag = _hierarchical_predict_compatible(
                        matrix,
                        attack_count,
                        trials,
                        rows,
                        train,
                        valid,
                    )
                    utility, utility_diag = _hierarchical_predict_compatible(
                        matrix,
                        utility_count,
                        trials,
                        rows,
                        train,
                        valid,
                    )
                    predictions[model_name]["attack"][repeat, valid] = attack
                    predictions[model_name]["utility"][repeat, valid] = utility
                    diagnostics.append(
                        {
                            "model": model_name,
                            "repeat_seed": seed,
                            "fold": fold,
                            "attack": attack_diag,
                            "utility": utility_diag,
                        }
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
    output = {}
    for model_name, values in predictions.items():
        if np.isnan(values["attack"]).any() or np.isnan(values["utility"]).any():
            raise AssertionError(f"Missing OOF prediction for {model_name}")
        attack_mean = values["attack"].mean(axis=0)
        utility_mean = values["utility"].mean(axis=0)
        output[model_name] = {
            "attack_mean": attack_mean,
            "attack_std": values["attack"].std(axis=0),
            "utility_mean": utility_mean,
            "utility_std": values["utility"].std(axis=0),
            "metrics": _evaluate_model(
                rows,
                attempts,
                attack_rates,
                utility_rates,
                attack_mean,
                utility_mean,
            ),
        }
    return output, feature_names, diagnostics


def _raw_models(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clean_attack = np.asarray(
        [_clip_probability(row.get("contrast_risk_score_mean")) for row in rows]
    )
    injection_attack = np.asarray(
        [
            _clip_probability(row.get("injection_contrast_risk_score_mean"))
            for row in rows
        ]
    )
    clean_utility = np.asarray(
        [
            _clip_probability(row.get("contrast_clean_solvability"))
            * _clip_probability(row.get("contrast_preservation_score_mean"))
            for row in rows
        ]
    )
    injection_utility = np.asarray(
        [
            _clip_probability(row.get("contrast_clean_solvability"))
            * _clip_probability(
                row.get("injection_contrast_preservation_score_mean")
            )
            for row in rows
        ]
    )
    return {
        "raw_clean_world": (clean_attack, clean_utility),
        "raw_injection_world": (injection_attack, injection_utility),
    }


def _task_bootstrap_difference(
    rows: list[dict[str, Any]],
    left_attack: np.ndarray,
    left_utility: np.ndarray,
    right_attack: np.ndarray,
    right_utility: np.ndarray,
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float] | float]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    task_indices: dict[tuple[str, str], np.ndarray] = {}
    for task in sorted({_task_key(row) for row in rows}):
        grouped[task[0]].append(task)
        task_indices[task] = np.asarray(
            [index for index, row in enumerate(rows) if _task_key(row) == task]
        )

    def task_pairwise(indices, predictions, rates):
        correct = 0.0
        total = 0
        for left, right in itertools.combinations(indices.tolist(), 2):
            observed = rates[left] - rates[right]
            if abs(float(observed)) < 1e-12:
                continue
            predicted = predictions[left] - predictions[right]
            correct += (
                0.5
                if abs(float(predicted)) < 1e-12
                else float(np.sign(observed) == np.sign(predicted))
            )
            total += 1
        return correct / total if total else np.nan

    task_values = {}
    for task, indices in task_indices.items():
        left_heads = np.asarray(
            [
                task_pairwise(indices, left_attack, attack_rates),
                task_pairwise(indices, left_utility, utility_rates),
            ],
            dtype=float,
        )
        right_heads = np.asarray(
            [
                task_pairwise(indices, right_attack, attack_rates),
                task_pairwise(indices, right_utility, utility_rates),
            ],
            dtype=float,
        )
        informative = np.isfinite(left_heads) & np.isfinite(right_heads)
        pairwise_difference = (
            float(np.mean(left_heads[informative] - right_heads[informative]))
            if informative.any()
            else float("nan")
        )
        left_brier = 0.5 * (
            np.mean((left_attack[indices] - attack_rates[indices]) ** 2)
            + np.mean((left_utility[indices] - utility_rates[indices]) ** 2)
        )
        right_brier = 0.5 * (
            np.mean((right_attack[indices] - attack_rates[indices]) ** 2)
            + np.mean((right_utility[indices] - utility_rates[indices]) ** 2)
        )
        task_values[task] = (
            pairwise_difference,
            left_brier - right_brier,
        )
    rng = np.random.default_rng(seed)
    values = np.empty((samples, 2), dtype=float)
    for sample in range(samples):
        chosen = []
        for suite in SUITES:
            suite_tasks = grouped[suite]
            chosen.extend(
                suite_tasks[index]
                for index in rng.integers(
                    0, len(suite_tasks), size=len(suite_tasks)
                )
            )
        task_tuples = chosen
        sampled = np.asarray(
            [task_values[task] for task in task_tuples], dtype=float
        )
        informative_sample = sampled[np.isfinite(sampled[:, 0]), 0]
        values[sample, 0] = (
            np.mean(informative_sample)
            if len(informative_sample)
            else float("nan")
        )
        values[sample, 1] = np.mean(sampled[:, 1])
    point_values = np.asarray(list(task_values.values()), dtype=float)
    point = np.asarray(
        [np.nanmean(point_values[:, 0]), np.mean(point_values[:, 1])]
    )
    finite_pairwise = values[np.isfinite(values[:, 0]), 0]
    if not len(finite_pairwise):
        raise ValueError("No informative task-level pairwise comparisons")
    return {
        "pairwise_accuracy_difference": float(point[0]),
        "pairwise_accuracy_difference_95ci": np.quantile(
            finite_pairwise, [0.025, 0.975]
        ).tolist(),
        "brier_difference": float(point[1]),
        "brier_difference_95ci": np.quantile(
            values[:, 1], [0.025, 0.975]
        ).tolist(),
        "informative_pairwise_task_count": int(
            np.isfinite(point_values[:, 0]).sum()
        ),
        "total_task_count": len(task_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cv-seeds", default="101,211,307,401,503")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Probability pairs missing in {args.dataset}")
    if len({_task_key(row) for row in rows}) != 24 or len(rows) != 96:
        raise ValueError("Frozen evaluation expects 24 tasks and 96 pairs")
    if any(int(row["replay_attempt_count"]) != 5 for row in rows):
        raise ValueError("Every pair must have five replay outcomes")
    required_injection_score = "injection_contrast_risk_score_mean"
    if any(required_injection_score not in row for row in rows):
        raise ValueError("Injection-conditioned score views are missing")

    cv_seeds = _parse_ints(args.cv_seeds)
    fitted, feature_names, diagnostics = _run_repeated_oof(
        rows, cv_seeds=cv_seeds
    )
    attack_count, utility_count, trials, attempts = _outcome_arrays(rows)
    attack_rates = attack_count / trials
    utility_rates = utility_count / trials
    results = {}
    all_predictions = {}
    for model_name, result in fitted.items():
        results[model_name] = result["metrics"]
        all_predictions[model_name] = (
            result["attack_mean"],
            result["utility_mean"],
            result["attack_std"],
            result["utility_std"],
        )
    for model_name, (attack, utility) in _raw_models(rows).items():
        results[model_name] = _evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            attack,
            utility,
        )
        all_predictions[model_name] = (
            attack,
            utility,
            np.zeros(len(rows)),
            np.zeros(len(rows)),
        )

    selected_model = max(
        DEPLOYABLE_MODELS,
        key=lambda name: (
            results[name]["primary_mean_within_task_pairwise_accuracy"],
            -results[name]["mean_pair_soft_brier"],
            name,
        ),
    )
    comparisons = {}
    for left, right in (
        ("joint_text_multinomial", "text_context_multinomial"),
        ("injection_world_ridge", "clean_world_ridge"),
        ("hierarchical_beta_binomial", "injection_world_ridge"),
        ("joint_text_multinomial", "injection_world_ridge"),
        ("raw_injection_world", "raw_clean_world"),
        (selected_model, "clean_world_ridge"),
        (selected_model, "text_context_multinomial"),
    ):
        left_values = all_predictions[left]
        right_values = all_predictions[right]
        comparisons[f"{left}__minus__{right}"] = _task_bootstrap_difference(
            rows,
            left_values[0],
            left_values[1],
            right_values[0],
            right_values[1],
            attack_rates,
            utility_rates,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    prediction_rows = []
    for index, row in enumerate(rows):
        item = {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
            "target_skill": row.get("target_skill"),
            "contrast_task_stratum": row["contrast_task_stratum"],
            "observed_attack_probability": float(attack_rates[index]),
            "observed_utility_probability": float(utility_rates[index]),
            "models": {},
        }
        for model_name, values in all_predictions.items():
            item["models"][model_name] = {
                "attack_probability_oof": float(values[0][index]),
                "utility_probability_oof": float(values[1][index]),
                "attack_uncertainty_oof": float(values[2][index]),
                "utility_uncertainty_oof": float(values[3][index]),
            }
        prediction_rows.append(item)

    summary = {
        "scope": "within_task_hierarchical_probability_evaluation",
        "dataset": str(args.dataset.resolve()),
        "evaluation": "five-repeat six-fold leave-user-task-out",
        "cv_seeds": cv_seeds,
        "primary_metric": "mean attack/utility within-task pairwise accuracy",
        "model_budget": list(MODEL_SPECS),
        "deployable_model_budget": list(DEPLOYABLE_MODELS),
        "model_selection_rule": (
            "highest primary within-task pairwise accuracy; "
            "mean pair Brier tie-break"
        ),
        "selected_model": selected_model,
        "feature_names": feature_names,
        "results": results,
        "comparisons": comparisons,
        "hierarchical_fit_diagnostics": diagnostics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("hierarchical_contrast_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("hierarchical_contrast_oof_predictions.json").write_text(
        json.dumps(
            {
                "scope": "within_task_contrast_oof_predictions",
                "selected_model": selected_model,
                "pairs": prediction_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
