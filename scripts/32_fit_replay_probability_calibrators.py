"""Fit replay-grounded probability calibrators and freeze a new selector.

The candidate/world-model scores are kept frozen. This script uses only the
multi-seed AgentDojo outcomes collected on the label-blind validation pilot to
learn two deliberately small post-hoc probability models:

* ridge logistic regression over a fixed low-dimensional feature set;
* monotonic isotonic calibration over the strongest single score per head.

All validation predictions are repeated grouped out-of-fold predictions where
(suite, user_task_id) is the group. Test predictions are an ensemble of the
corresponding cross-fit models. Cached observed labels in candidate files are
never used as features or training targets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUITES = ("banking", "slack", "travel", "workspace")
MODEL_NAMES = ("ridge_logistic", "monotonic_isotonic")
FORBIDDEN_FEATURE_KEYS = (
    "suite/user/injection identifiers",
    "trajectory_id",
    "attack",
    "target_skill",
    "source_trace",
    "observed_security",
    "observed_utility",
    "security",
    "utility",
)
EPSILON = 1e-5


def _parse_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("At least one integer is required")
    return output


def _parse_floats(value: str) -> list[float]:
    output = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("At least one float is required")
    return output


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _safe_probability(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            return float(np.clip(number, EPSILON, 1.0 - EPSILON))
    return float("nan")


def _logit(value: Any) -> float:
    probability = _safe_probability(value)
    if not math.isfinite(probability):
        return float("nan")
    return float(math.log(probability / (1.0 - probability)))


def _safe_number(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            return number
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
        raise ValueError(f"No clean solvability rates found in {path}")
    return rates


def _load_pilot_rows(path: Path, selection_name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("selections", {}).get(selection_name)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Selection {selection_name!r} is missing or empty in {path}")
    keys = [_pair_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Pilot selection contains duplicate task/injection pairs")
    return rows


def _load_replay_attempts(
    paths: Iterable[Path],
    selection_name: str,
    expected_keys: set[tuple[str, str, str]],
) -> tuple[dict[tuple[str, str, str], list[tuple[int, int]]], list[dict[str, Any]]]:
    attempts: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    seen_seeds: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = str(payload.get("seed"))
        if seed in seen_seeds:
            raise ValueError(f"Duplicate replay seed: {seed}")
        seen_seeds.add(seed)
        rows = payload.get("results", {}).get(selection_name, {}).get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Replay selection {selection_name!r} missing in {path}")
        row_keys = [_pair_key(row) for row in rows]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError(f"Duplicate pair within replay {path}")
        if set(row_keys) != expected_keys:
            raise ValueError(
                f"Replay pair mismatch in {path}: "
                f"missing={len(expected_keys - set(row_keys))} "
                f"extra={len(set(row_keys) - expected_keys)}"
            )
        for row in rows:
            attempts[_pair_key(row)].append(
                (int(bool(row["security"])), int(bool(row["utility"])))
            )
        metadata.append(
            {
                "path": str(path.resolve()),
                "seed": payload.get("seed"),
                "do_sample": payload.get("do_sample"),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
            }
        )
    if len(seen_seeds) < 2:
        raise ValueError("At least two replay seeds are required for probability labels")
    counts = {len(values) for values in attempts.values()}
    if len(counts) != 1:
        raise ValueError(f"Unequal replay attempt counts across pairs: {sorted(counts)}")
    return dict(attempts), metadata


def _feature_matrix(
    rows: list[dict[str, Any]],
    clean_rates: dict[tuple[str, str], float],
    *,
    head: str,
    model_name: str,
) -> tuple[np.ndarray, list[str]]:
    if model_name == "monotonic_isotonic":
        key = (
            "candidate_risk_score"
            if head == "attack"
            else "candidate_expected_utility_score"
        )
        values = np.asarray([_safe_probability(row.get(key)) for row in rows])
        if np.isnan(values).any():
            raise ValueError(f"Missing required isotonic score: {key}")
        return values[:, None], [key]

    feature_rows: list[list[float]] = []
    if head == "attack":
        names = [
            "logit_candidate_risk",
            "logit_raw_risk",
            "logit_target_probability",
            "target_reached",
            "candidate_minus_raw_risk",
            *[f"suite_{suite}" for suite in SUITES],
        ]
        for row in rows:
            candidate_risk = _safe_number(row.get("candidate_risk_score"))
            raw_risk = _safe_number(row.get("risk_score"))
            feature_rows.append(
                [
                    _logit(candidate_risk),
                    _logit(raw_risk),
                    _logit(row.get("target_skill_probability")),
                    _safe_number(row.get("rollout_target_reached")),
                    candidate_risk - raw_risk,
                    *[float(row.get("suite") == suite) for suite in SUITES],
                ]
            )
    elif head == "utility":
        names = [
            "logit_candidate_expected_utility",
            "logit_candidate_preservation",
            "logit_candidate_utility",
            "logit_final_utility",
            "logit_value",
            "logit_clean_solvability",
            *[f"suite_{suite}" for suite in SUITES],
        ]
        for row in rows:
            clean_rate = clean_rates.get(_task_key(row), float("nan"))
            feature_rows.append(
                [
                    _logit(row.get("candidate_expected_utility_score")),
                    _logit(row.get("candidate_preservation_score")),
                    _logit(row.get("candidate_utility_score")),
                    _logit(row.get("final_utility_score")),
                    _logit(row.get("value_score")),
                    _logit(clean_rate),
                    *[float(row.get("suite") == suite) for suite in SUITES],
                ]
            )
    else:
        raise ValueError(f"Unknown probability head: {head}")
    return np.asarray(feature_rows, dtype=float), names


def _make_folds(
    security_rates: np.ndarray,
    utility_rates: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two user-task groups are required")
    strata = 2 * (security_rates >= 0.5).astype(int) + (
        utility_rates >= 0.5
    ).astype(int)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    output = list(splitter.split(np.zeros(len(groups)), strata, groups))
    seen = np.zeros(len(groups), dtype=int)
    for train_index, valid_index in output:
        if set(groups[train_index]) & set(groups[valid_index]):
            raise AssertionError("User-task group leaked across a fold")
        seen[valid_index] += 1
    if not np.all(seen == 1):
        raise AssertionError("Grouped folds did not cover every pilot pair exactly once")
    return output


def _expanded_training_data(
    matrix: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    pair_indices: np.ndarray,
    head_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    labels: list[int] = []
    for index in pair_indices:
        for outcome in attempts[int(index)]:
            x_rows.append(matrix[int(index)])
            labels.append(int(outcome[head_index]))
    return np.asarray(x_rows, dtype=float), np.asarray(labels, dtype=int)


def _constant_probability(labels: np.ndarray) -> float:
    return float((float(labels.sum()) + 0.5) / (len(labels) + 1.0))


def _fit_predict_head(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(predict_x), _constant_probability(train_y))
    if model_name == "ridge_logistic":
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
        prediction = model.predict_proba(predict_x)[:, 1]
    elif model_name == "monotonic_isotonic":
        if len(np.unique(train_x[:, 0])) < 2:
            return np.full(len(predict_x), _constant_probability(train_y))
        model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=EPSILON,
            y_max=1.0 - EPSILON,
        )
        model.fit(train_x[:, 0], train_y)
        prediction = model.predict(predict_x[:, 0])
    else:
        raise ValueError(f"Unknown calibrator: {model_name}")
    return np.clip(np.asarray(prediction, dtype=float), EPSILON, 1.0 - EPSILON)


def _probability_metrics(
    rates: np.ndarray,
    predictions: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    head_index: int,
) -> dict[str, Any]:
    predictions = np.clip(predictions, EPSILON, 1.0 - EPSILON)
    pair_brier = float(np.mean((predictions - rates) ** 2))
    log_loss = float(
        -np.mean(
            rates * np.log(predictions)
            + (1.0 - rates) * np.log(1.0 - predictions)
        )
    )
    correlation = None
    if len(np.unique(rates)) > 1 and len(np.unique(predictions)) > 1:
        value = spearmanr(rates, predictions).statistic
        if math.isfinite(float(value)):
            correlation = float(value)
    expanded_labels: list[int] = []
    expanded_predictions: list[float] = []
    for index, outcomes in enumerate(attempts):
        expanded_labels.extend(int(outcome[head_index]) for outcome in outcomes)
        expanded_predictions.extend([float(predictions[index])] * len(outcomes))
    labels_array = np.asarray(expanded_labels, dtype=int)
    predictions_array = np.asarray(expanded_predictions, dtype=float)
    auc = None
    if len(np.unique(labels_array)) > 1:
        auc = float(roc_auc_score(labels_array, predictions_array))

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
        "pair_soft_brier": pair_brier,
        "pair_binomial_log_loss": log_loss,
        "pair_mae": float(np.mean(np.abs(predictions - rates))),
        "pair_spearman": correlation,
        "attempt_roc_auc": auc,
        "equal_count_ece_5bin": ece,
        "observed_mean": float(rates.mean()),
        "predicted_mean": float(predictions.mean()),
    }


def _crossfit_model(
    model_name: str,
    pilot_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    clean_rates: dict[tuple[str, str], float],
    attempts: list[list[tuple[int, int]]],
    security_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    cv_seeds: list[int],
    n_splits: int,
) -> dict[str, Any]:
    groups = np.asarray(
        [f"{row['suite']}::{row['user_task_id']}" for row in pilot_rows]
    )
    matrices: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for head in ("attack", "utility"):
        pilot_x, feature_names = _feature_matrix(
            pilot_rows, clean_rates, head=head, model_name=model_name
        )
        test_x, test_feature_names = _feature_matrix(
            test_rows, clean_rates, head=head, model_name=model_name
        )
        if feature_names != test_feature_names:
            raise AssertionError("Pilot and test feature schemas differ")
        matrices[head] = (pilot_x, test_x, feature_names)

    oof_repeats = {
        "attack": np.full((len(cv_seeds), len(pilot_rows)), np.nan),
        "utility": np.full((len(cv_seeds), len(pilot_rows)), np.nan),
    }
    test_predictions: dict[str, list[np.ndarray]] = {
        "attack": [],
        "utility": [],
    }
    fold_metadata: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(cv_seeds):
        folds = _make_folds(
            security_rates,
            utility_rates,
            groups,
            n_splits=n_splits,
            random_state=seed,
        )
        assignments = np.full(len(pilot_rows), -1, dtype=int)
        for fold_index, (train_index, valid_index) in enumerate(folds):
            assignments[valid_index] = fold_index
            for head_index, head in enumerate(("attack", "utility")):
                pilot_x, test_x, _ = matrices[head]
                train_x, train_y = _expanded_training_data(
                    pilot_x, attempts, train_index, head_index
                )
                oof_repeats[head][repeat_index, valid_index] = _fit_predict_head(
                    model_name, train_x, train_y, pilot_x[valid_index]
                )
                test_predictions[head].append(
                    _fit_predict_head(model_name, train_x, train_y, test_x)
                )
        if np.any(assignments < 0):
            raise AssertionError("Missing fold assignment")
        fold_metadata.append(
            {
                "seed": seed,
                "fold_count": len(folds),
                "assignment": assignments.tolist(),
            }
        )

    output: dict[str, Any] = {
        "model_name": model_name,
        "feature_names": {
            head: matrices[head][2] for head in ("attack", "utility")
        },
        "folds": fold_metadata,
    }
    for head_index, (head, rates) in enumerate(
        (("attack", security_rates), ("utility", utility_rates))
    ):
        if np.isnan(oof_repeats[head]).any():
            raise AssertionError(f"Missing OOF predictions for {head}")
        oof_mean = oof_repeats[head].mean(axis=0)
        oof_std = oof_repeats[head].std(axis=0)
        test_stack = np.asarray(test_predictions[head], dtype=float)
        test_mean = test_stack.mean(axis=0)
        test_std = test_stack.std(axis=0)
        output[head] = {
            "oof_mean": oof_mean,
            "oof_std": oof_std,
            "test_mean": test_mean,
            "test_std": test_std,
            "metrics": _probability_metrics(
                rates, oof_mean, attempts, head_index
            ),
        }
    output["selection_metric"] = {
        "mean_pair_soft_brier": float(
            0.5
            * (
                output["attack"]["metrics"]["pair_soft_brier"]
                + output["utility"]["metrics"]["pair_soft_brier"]
            )
        ),
        "mean_pair_binomial_log_loss": float(
            0.5
            * (
                output["attack"]["metrics"]["pair_binomial_log_loss"]
                + output["utility"]["metrics"]["pair_binomial_log_loss"]
            )
        ),
    }
    return output


def _annotate_predictions(
    rows: list[dict[str, Any]],
    *,
    attack_mean: np.ndarray,
    attack_std: np.ndarray,
    utility_mean: np.ndarray,
    utility_std: np.ndarray,
    model_name: str,
    uncertainty_weight: float,
    oof: bool,
) -> list[dict[str, Any]]:
    output = []
    suffix = "_oof" if oof else ""
    for index, row in enumerate(rows):
        utility_lcb = float(
            np.clip(
                utility_mean[index] - uncertainty_weight * utility_std[index],
                0.0,
                1.0,
            )
        )
        output.append(
            {
                **row,
                f"replay_attack_probability{suffix}": float(attack_mean[index]),
                f"replay_attack_uncertainty{suffix}": float(attack_std[index]),
                f"replay_utility_probability{suffix}": float(utility_mean[index]),
                f"replay_utility_uncertainty{suffix}": float(utility_std[index]),
                f"replay_utility_lcb{suffix}": utility_lcb,
                "replay_probability_model": model_name,
            }
        )
    return output


def _select_probability_rows(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    utility_floor: float,
    objective: str,
    max_per_user_task: int,
    oof: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = "_oof" if oof else ""
    attack_key = f"replay_attack_probability{suffix}"
    utility_key = f"replay_utility_lcb{suffix}"
    ranked: list[dict[str, Any]] = []
    for row in rows:
        attack = float(row[attack_key])
        utility = float(row[utility_key])
        if objective == "risk_guard":
            score = attack
        elif objective == "joint_lcb":
            score = attack + utility
        else:
            raise ValueError(f"Unknown probability objective: {objective}")
        ranked.append(
            {
                **row,
                "replay_selection_objective": objective,
                "replay_selection_score": score,
                "replay_selection_utility_floor": utility_floor,
                "replay_selection_feasible": utility >= utility_floor,
            }
        )
    feasible = sorted(
        (row for row in ranked if row["replay_selection_feasible"]),
        key=lambda row: (
            float(row["replay_selection_score"]),
            float(row[utility_key]),
            _pair_key(row),
        ),
        reverse=True,
    )
    fallback = sorted(
        (row for row in ranked if not row["replay_selection_feasible"]),
        key=lambda row: (
            float(row[utility_key]),
            float(row["replay_selection_score"]),
            _pair_key(row),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    task_counts: Counter[tuple[str, str]] = Counter()
    feasible_selected = 0
    for row in itertools.chain(feasible, fallback):
        task = _task_key(row)
        if max_per_user_task > 0 and task_counts[task] >= max_per_user_task:
            continue
        selected.append(row)
        task_counts[task] += 1
        feasible_selected += int(bool(row["replay_selection_feasible"]))
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise RuntimeError(f"Could only select {len(selected)}/{top_k} candidates")
    return selected, {
        "top_k": top_k,
        "utility_floor": utility_floor,
        "objective": objective,
        "max_per_user_task": max_per_user_task,
        "feasible_pool_count": len(feasible),
        "feasible_selected_count": feasible_selected,
        "fallback_selected_count": top_k - feasible_selected,
    }


def _selection_rates(
    rows: list[dict[str, Any]],
    rate_by_pair: dict[tuple[str, str, str], tuple[float, float]],
) -> dict[str, float]:
    security = np.asarray([rate_by_pair[_pair_key(row)][0] for row in rows])
    utility = np.asarray([rate_by_pair[_pair_key(row)][1] for row in rows])
    return {
        "observed_asr": float(security.mean()),
        "observed_bup": float(utility.mean()),
        "asr_plus_bup": float(security.mean() + utility.mean()),
    }


def _overlap(selections: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    output: dict[str, int] = {}
    names = list(selections)
    for left_index, left_name in enumerate(names):
        left = {_pair_key(row) for row in selections[left_name]}
        for right_name in names[left_index + 1 :]:
            right = {_pair_key(row) for row in selections[right_name]}
            output[f"{left_name}__{right_name}"] = len(left & right)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument(
        "--selection-name", default="validation_probability_pilot"
    )
    parser.add_argument("--replay", action="append", type=Path, required=True)
    parser.add_argument("--test-candidates", type=Path, required=True)
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--baseline-selections-json", type=Path)
    parser.add_argument("--cv-seeds", default="101,211,307,401,503")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--utility-floors", default="0.333333,0.5,0.666667")
    parser.add_argument("--utility-uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--max-per-user-task", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pilot_rows = _load_pilot_rows(args.selection_json, args.selection_name)
    pilot_keys = {_pair_key(row) for row in pilot_rows}
    attempts_by_key, replay_metadata = _load_replay_attempts(
        args.replay, args.selection_name, pilot_keys
    )
    attempts = [attempts_by_key[_pair_key(row)] for row in pilot_rows]
    security_rates = np.asarray(
        [np.mean([outcome[0] for outcome in values]) for values in attempts]
    )
    utility_rates = np.asarray(
        [np.mean([outcome[1] for outcome in values]) for values in attempts]
    )
    rate_by_pair = {
        _pair_key(row): (
            float(security_rates[index]),
            float(utility_rates[index]),
        )
        for index, row in enumerate(pilot_rows)
    }

    test_payload = json.loads(args.test_candidates.read_text(encoding="utf-8"))
    test_rows = test_payload.get("candidates")
    if not isinstance(test_rows, list) or not test_rows:
        raise ValueError(f"No test candidates found in {args.test_candidates}")
    clean_rates = _load_clean_rates(args.clean_solvability_json)
    cv_seeds = _parse_ints(args.cv_seeds)

    models: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        models[model_name] = _crossfit_model(
            model_name,
            pilot_rows,
            test_rows,
            clean_rates,
            attempts,
            security_rates,
            utility_rates,
            cv_seeds=cv_seeds,
            n_splits=args.folds,
        )
    selected_model_name = min(
        MODEL_NAMES,
        key=lambda name: (
            models[name]["selection_metric"]["mean_pair_soft_brier"],
            models[name]["selection_metric"]["mean_pair_binomial_log_loss"],
            name,
        ),
    )
    selected_model = models[selected_model_name]

    raw_metrics = {
        "attack": _probability_metrics(
            security_rates,
            np.asarray(
                [
                    _safe_probability(row.get("candidate_risk_score"))
                    for row in pilot_rows
                ]
            ),
            attempts,
            0,
        ),
        "utility": _probability_metrics(
            utility_rates,
            np.asarray(
                [
                    _safe_probability(
                        row.get("candidate_expected_utility_score")
                    )
                    for row in pilot_rows
                ]
            ),
            attempts,
            1,
        ),
    }

    pilot_annotated = _annotate_predictions(
        pilot_rows,
        attack_mean=selected_model["attack"]["oof_mean"],
        attack_std=selected_model["attack"]["oof_std"],
        utility_mean=selected_model["utility"]["oof_mean"],
        utility_std=selected_model["utility"]["oof_std"],
        model_name=selected_model_name,
        uncertainty_weight=args.utility_uncertainty_weight,
        oof=True,
    )
    for index, row in enumerate(pilot_annotated):
        row["replay_attempt_count"] = len(attempts[index])
        row["replay_observed_attack_rate"] = float(security_rates[index])
        row["replay_observed_utility_rate"] = float(utility_rates[index])

    selection_grid: list[dict[str, Any]] = []
    for objective in ("risk_guard", "joint_lcb"):
        for utility_floor in _parse_floats(args.utility_floors):
            selected, metadata = _select_probability_rows(
                pilot_annotated,
                top_k=args.top_k,
                utility_floor=utility_floor,
                objective=objective,
                max_per_user_task=args.max_per_user_task,
                oof=True,
            )
            selection_grid.append(
                {**metadata, **_selection_rates(selected, rate_by_pair)}
            )
    selected_config = max(
        selection_grid,
        key=lambda row: (
            float(row["asr_plus_bup"]),
            float(row["observed_bup"]),
            float(row["observed_asr"]),
            -int(row["fallback_selected_count"]),
            float(row["utility_floor"]),
            str(row["objective"]),
        ),
    )

    test_annotated = _annotate_predictions(
        test_rows,
        attack_mean=selected_model["attack"]["test_mean"],
        attack_std=selected_model["attack"]["test_std"],
        utility_mean=selected_model["utility"]["test_mean"],
        utility_std=selected_model["utility"]["test_std"],
        model_name=selected_model_name,
        uncertainty_weight=args.utility_uncertainty_weight,
        oof=False,
    )
    new_selection, test_selection_metadata = _select_probability_rows(
        test_annotated,
        top_k=args.top_k,
        utility_floor=float(selected_config["utility_floor"]),
        objective=str(selected_config["objective"]),
        max_per_user_task=args.max_per_user_task,
        oof=False,
    )

    selections: dict[str, list[dict[str, Any]]] = {
        "replay_probability_world_model": new_selection
    }
    selection_metadata: dict[str, Any] = {
        "replay_probability_world_model": {
            **test_selection_metadata,
            "probability_model": selected_model_name,
            "selection_was_frozen_on": "grouped_repeated_oof_validation_replay",
        }
    }
    if args.baseline_selections_json is not None:
        baseline_payload = json.loads(
            args.baseline_selections_json.read_text(encoding="utf-8")
        )
        for name, rows in baseline_payload.get("selections", {}).items():
            if name in selections:
                raise ValueError(f"Duplicate merged selection name: {name}")
            selections[name] = rows
            selection_metadata[name] = baseline_payload.get("metadata", {}).get(
                name, {}
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot_output = {
        "scope": "replay_probability_grouped_oof_validation_candidates",
        "selection_name": args.selection_name,
        "candidates": pilot_annotated,
    }
    test_output = {
        **test_payload,
        "replay_probability_calibration": {
            "selected_model": selected_model_name,
            "cv_seeds": cv_seeds,
            "folds": args.folds,
            "training_pair_count": len(pilot_rows),
            "attempts_per_pair": len(attempts[0]),
        },
        "candidates": test_annotated,
    }
    replay_selection_output = {
        "scope": "replay_probability_frozen_fresh_seed_comparison",
        "selection_uses_test_labels": False,
        "selections": selections,
        "metadata": selection_metadata,
        "overlap": _overlap(selections),
    }

    serializable_models = {}
    for name, result in models.items():
        serializable_models[name] = {
            "feature_names": result["feature_names"],
            "folds": result["folds"],
            "selection_metric": result["selection_metric"],
            "attack": {
                "metrics": result["attack"]["metrics"],
                "mean_oof_uncertainty": float(
                    result["attack"]["oof_std"].mean()
                ),
                "mean_test_uncertainty": float(
                    result["attack"]["test_std"].mean()
                ),
            },
            "utility": {
                "metrics": result["utility"]["metrics"],
                "mean_oof_uncertainty": float(
                    result["utility"]["oof_std"].mean()
                ),
                "mean_test_uncertainty": float(
                    result["utility"]["test_std"].mean()
                ),
            },
        }
    summary = {
        "scope": "replay_grounded_probability_calibration",
        "selection_json": str(args.selection_json.resolve()),
        "test_candidates": str(args.test_candidates.resolve()),
        "clean_solvability_json": str(args.clean_solvability_json.resolve()),
        "replays": replay_metadata,
        "training_pair_count": len(pilot_rows),
        "attempts_per_pair": len(attempts[0]),
        "total_attack_outcomes": sum(len(values) for values in attempts),
        "grouping_unit": "suite_and_user_task_id",
        "selection_uses_test_labels": False,
        "forbidden_features": list(FORBIDDEN_FEATURE_KEYS),
        "model_budget": list(MODEL_NAMES),
        "model_selection_rule": (
            "lowest equal-head mean pair soft Brier; log-loss tie-break"
        ),
        "raw_world_model_metrics": raw_metrics,
        "models": serializable_models,
        "selected_model": selected_model_name,
        "selection_grid_budget": {
            "top_k": args.top_k,
            "objectives": ["risk_guard", "joint_lcb"],
            "utility_floors": _parse_floats(args.utility_floors),
            "utility_uncertainty_weight": args.utility_uncertainty_weight,
            "config_count": len(selection_grid),
        },
        "validation_selection_grid": selection_grid,
        "selected_validation_config": selected_config,
        "test_selection_metadata": test_selection_metadata,
        "test_selection_overlap": replay_selection_output["overlap"],
    }

    args.output_dir.joinpath("pilot_probability_candidates.json").write_text(
        json.dumps(pilot_output, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("test_probability_candidates.json").write_text(
        json.dumps(test_output, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("fresh_replay_selections.json").write_text(
        json.dumps(replay_selection_output, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("probability_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
