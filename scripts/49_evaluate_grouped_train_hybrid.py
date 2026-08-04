"""Train grouped-task rank/probability models and evaluate on grouped validation.

The script deliberately decouples ordering from probability reporting.  Small
probability calibrators are selected by grouped OOF Brier on the training tasks;
pairwise rankers are trained only from within-user-task contrasts.  The grouped
validation labels select one model from a fixed, declared candidate budget.  Test
labels are never read for fitting, selection, or validation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 13, 21)
MODES = ("clean_prefix_rollout", "injection_conditioned_rollout")
WORLD_FIELDS = (
    "risk_score",
    "utility_score",
    "preservation_score",
    "min_utility_score",
    "final_utility_score",
    "value_score",
    "reward_score",
    "target_skill_probability",
    "rollout_target_reached",
)
C_VALUES = (0.03, 0.1, 0.3)
BLEND_ALPHAS = (0.25, 0.5, 0.75)
EPSILON = 1e-5


def _load_metric_module():
    path = ROOT / "scripts" / "38_evaluate_hierarchical_contrast_models.py"
    spec = importlib.util.spec_from_file_location("contrast_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_metric_module()


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (*_task_key(row), str(row["injection_task_id"]))


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidates missing in {path}")
    return rows


def _injection_context(path: Path) -> tuple[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    injections = raw.get("injections")
    if not isinstance(injections, dict) or not injections:
        return "", []
    locations = sorted(str(key) for key in injections)
    text = "\n".join(str(injections[key]) for key in locations)
    return text, locations


def _load_split(
    archive: Path,
    split: str,
    *,
    seeds: tuple[int, ...] = SEEDS,
    load_text: bool = True,
) -> list[dict[str, Any]]:
    mappings: dict[str, dict[int, dict[tuple[str, str, str], dict[str, Any]]]] = {
        mode: {} for mode in MODES
    }
    for mode in MODES:
        for seed in seeds:
            rows = _load_candidates(
                archive / f"seed{seed}" / f"{split}_{mode}_candidates.json"
            )
            mapping = {_pair_key(row): row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError(f"Duplicate pair in {split}/{mode}/seed{seed}")
            mappings[mode][seed] = mapping
    key_sets = [
        set(mapping)
        for mode_mappings in mappings.values()
        for mapping in mode_mappings.values()
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(f"Candidate sources do not align for {split}")

    output = []
    for key in sorted(key_sets[0]):
        reference = mappings[MODES[0]][seeds[0]][key]
        labels = {
            (
                bool(mappings[mode][seed][key]["observed_security"]),
                bool(mappings[mode][seed][key]["observed_utility"]),
            )
            for mode in MODES
            for seed in seeds
        }
        if len(labels) != 1:
            raise ValueError(f"Observed labels disagree for {key}")
        source_paths = {
            str(mappings[mode][seed][key]["source_trace"])
            for mode in MODES
            for seed in seeds
        }
        if len(source_paths) != 1:
            raise ValueError(f"Source traces disagree for {key}")
        source_trace = next(iter(source_paths))
        injection_text, locations = (
            _injection_context(Path(source_trace)) if load_text else ("", [])
        )
        seed_predictions = {
            str(seed): {
                mode: {
                    field: float(mappings[mode][seed][key][field])
                    for field in WORLD_FIELDS
                }
                for mode in MODES
            }
            for seed in seeds
        }
        output.append(
            {
                "suite": key[0],
                "user_task_id": key[1],
                "injection_task_id": key[2],
                "trajectory_id": str(reference["trajectory_id"]),
                "target_skill": str(reference.get("target_skill", "")),
                "source_trace": source_trace,
                "injection_text": injection_text,
                "injection_locations": locations,
                "observed_security": bool(reference["observed_security"]),
                "observed_utility": bool(reference["observed_utility"]),
                "seed_predictions": seed_predictions,
            }
        )
    return output


def _soft_targets(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("attack_action"):
                continue
            target = row.get("utility_probability_target")
            if target is None:
                continue
            confidence = float(row.get("probability_label_confidence", 1.0))
            grouped[str(row["trajectory_id"])].append((float(target), confidence))
    return {
        trajectory_id: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for trajectory_id, values in grouped.items()
    }


def _labels(
    rows: list[dict[str, Any]],
    soft_targets: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attack = np.asarray([float(row["observed_security"]) for row in rows])
    utility = []
    confidence = []
    for row in rows:
        soft = soft_targets.get(str(row["trajectory_id"]))
        if soft is None:
            utility.append(float(row["observed_utility"]))
            confidence.append(1.0)
        else:
            utility.append(float(soft[0]))
            confidence.append(float(soft[1]))
    return attack, np.asarray(utility), np.asarray(confidence)


def _text_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    documents = []
    for row in rows:
        context = " ".join(
            [
                str(row.get("injection_text", "")),
                f"__suite_{row['suite']}",
                f"__target_{row.get('target_skill', '')}",
                *[
                    f"__location_{location}"
                    for location in row.get("injection_locations", [])
                ],
            ]
        )
        documents.append(context)
    vectorizer = HashingVectorizer(
        n_features=256,
        alternate_sign=True,
        norm="l2",
        ngram_range=(1, 2),
    )
    return vectorizer.transform(documents).toarray()


def _seed_values(
    row: dict[str, Any], mode: str, field: str, seeds: tuple[int, ...] = SEEDS
) -> np.ndarray:
    return np.asarray(
        [row["seed_predictions"][str(seed)][mode][field] for seed in seeds],
        dtype=float,
    )


def _world_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    output = []
    for row in rows:
        features = []
        means: dict[tuple[str, str], float] = {}
        for mode in MODES:
            for field in WORLD_FIELDS:
                values = _seed_values(row, mode, field)
                means[(mode, field)] = float(values.mean())
                features.extend(
                    [
                        float(values.mean()),
                        float(values.std()),
                        float(values.min()),
                        float(values.max()),
                    ]
                )
        for field in WORLD_FIELDS:
            features.append(
                means[("injection_conditioned_rollout", field)]
                - means[("clean_prefix_rollout", field)]
            )
        output.append(features)
    return np.asarray(output, dtype=float)


def _raw_scores(
    rows: list[dict[str, Any]], mode: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [_seed_values(row, mode, field) for row in rows], dtype=float
    )
    return matrix.mean(axis=1), matrix.std(axis=1)


def _make_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000),
            ),
        ]
    )


def _fit_soft_probability(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float = 0.1,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(train_y), dtype=float)
    positive_weight = np.asarray(weights) * np.clip(train_y, 0.0, 1.0)
    negative_weight = np.asarray(weights) * (1.0 - np.clip(train_y, 0.0, 1.0))
    expanded_x = np.concatenate([train_x, train_x], axis=0)
    expanded_y = np.concatenate(
        [np.ones(len(train_y), dtype=int), np.zeros(len(train_y), dtype=int)]
    )
    expanded_weight = np.concatenate([positive_weight, negative_weight])
    keep = expanded_weight > 1e-9
    if len(np.unique(expanded_y[keep])) < 2:
        probability = float(
            (np.sum(train_y * weights) + 0.5) / (np.sum(weights) + 1.0)
        )
        return np.full(len(predict_x), probability)
    model = _make_pipeline(c_value)
    model.fit(
        expanded_x[keep],
        expanded_y[keep],
        model__sample_weight=expanded_weight[keep],
    )
    return np.clip(model.predict_proba(predict_x)[:, 1], EPSILON, 1 - EPSILON)


def _grouped_oof_probability(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int = 5,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    predictions = np.full(len(labels), np.nan)
    for train, valid in splitter.split(matrix, labels, groups):
        predictions[valid] = _fit_soft_probability(
            matrix[train],
            labels[train],
            matrix[valid],
            weights=weights[train],
        )
    if np.isnan(predictions).any():
        raise AssertionError("Grouped OOF probabilities are incomplete")
    return predictions


def _pairwise_rank_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    groups: np.ndarray,
    predict_x: np.ndarray,
    *,
    c_value: float,
    minimum_gap: float = 0.05,
) -> tuple[np.ndarray, int]:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(imputer.fit_transform(train_x))
    transformed_predict = scaler.transform(imputer.transform(predict_x))
    ordered_pairs = []
    pair_weights = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        for left, right in itertools.combinations(indices.tolist(), 2):
            difference = float(train_y[left] - train_y[right])
            if abs(difference) < minimum_gap:
                continue
            high, low = (left, right) if difference > 0 else (right, left)
            ordered_pairs.append((high, low))
            pair_weights.append(abs(difference))
    if not ordered_pairs:
        return np.zeros(len(predict_x), dtype=float), 0
    pair_indices = np.asarray(ordered_pairs, dtype=int)
    differences = (
        transformed_train[pair_indices[:, 0]]
        - transformed_train[pair_indices[:, 1]]
    )
    pair_x = np.concatenate([differences, -differences], axis=0)
    pair_y = np.concatenate(
        [np.ones(len(differences), dtype=int), np.zeros(len(differences), dtype=int)]
    )
    pair_weight = np.concatenate(
        [np.asarray(pair_weights), np.asarray(pair_weights)]
    )
    model = LogisticRegression(
        C=c_value,
        fit_intercept=False,
        solver="liblinear",
        max_iter=3000,
    )
    model.fit(pair_x, pair_y, sample_weight=pair_weight)
    return np.asarray(model.decision_function(transformed_predict)), len(ordered_pairs)


def _within_task_rank(
    rows: list[dict[str, Any]], values: np.ndarray
) -> np.ndarray:
    output = np.zeros(len(rows), dtype=float)
    for task in sorted({_task_key(row) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if _task_key(row) == task]
        )
        denominator = max(len(indices) - 1, 1)
        output[indices] = (rankdata(values[indices], method="average") - 1) / denominator
    return output


def _borda_attack(rows: list[dict[str, Any]]) -> np.ndarray:
    seed_ranks = []
    for seed in SEEDS:
        values = np.asarray(
            [
                row["seed_predictions"][str(seed)][
                    "injection_conditioned_rollout"
                ]["risk_score"]
                for row in rows
            ],
            dtype=float,
        )
        seed_ranks.append(_within_task_rank(rows, values))
    return np.mean(seed_ranks, axis=0)


def _method_metrics(
    rows: list[dict[str, Any]],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    method: dict[str, np.ndarray],
) -> dict[str, Any]:
    attack_pairwise = METRICS._within_task_metrics(
        rows, attack_rates, method["attack_rank"]
    )
    utility_pairwise = METRICS._within_task_metrics(
        rows, utility_rates, method["utility_rank"]
    )
    pairwise = [
        value
        for value in (
            attack_pairwise["pairwise_accuracy"],
            utility_pairwise["pairwise_accuracy"],
        )
        if value is not None
    ]
    attack_brier = float(
        np.mean((method["attack_probability"] - attack_rates) ** 2)
    )
    utility_brier = float(
        np.mean((method["utility_probability"] - utility_rates) ** 2)
    )
    return {
        "primary_mean_within_task_pairwise_accuracy": float(np.mean(pairwise)),
        "mean_pair_soft_brier": 0.5 * (attack_brier + utility_brier),
        "attack": {
            "within_task": attack_pairwise,
            "pair_soft_brier": attack_brier,
        },
        "utility": {
            "within_task": utility_pairwise,
            "pair_soft_brier": utility_brier,
        },
    }


def _bootstrap_difference(
    rows: list[dict[str, Any]],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    task_indices = {
        task: np.asarray(
            [index for index, row in enumerate(rows) if _task_key(row) == task]
        )
        for task in sorted({_task_key(row) for row in rows})
    }
    by_suite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in task_indices:
        by_suite[task[0]].append(task)

    def pairwise(indices: np.ndarray, prediction: np.ndarray, rates: np.ndarray):
        correct = 0.0
        total = 0
        for first, second in itertools.combinations(indices.tolist(), 2):
            observed = float(rates[first] - rates[second])
            if abs(observed) < 1e-12:
                continue
            predicted = float(prediction[first] - prediction[second])
            correct += (
                0.5
                if abs(predicted) < 1e-12
                else float(np.sign(observed) == np.sign(predicted))
            )
            total += 1
        return correct / total if total else float("nan")

    task_values = {}
    for task, indices in task_indices.items():
        differences = []
        for head, rates in (("attack", attack_rates), ("utility", utility_rates)):
            left_value = pairwise(indices, left[f"{head}_rank"], rates)
            right_value = pairwise(indices, right[f"{head}_rank"], rates)
            if math.isfinite(left_value) and math.isfinite(right_value):
                differences.append(left_value - right_value)
        pairwise_difference = (
            float(np.mean(differences)) if differences else float("nan")
        )
        left_brier = 0.5 * (
            np.mean((left["attack_probability"][indices] - attack_rates[indices]) ** 2)
            + np.mean(
                (left["utility_probability"][indices] - utility_rates[indices]) ** 2
            )
        )
        right_brier = 0.5 * (
            np.mean(
                (right["attack_probability"][indices] - attack_rates[indices]) ** 2
            )
            + np.mean(
                (right["utility_probability"][indices] - utility_rates[indices]) ** 2
            )
        )
        task_values[task] = (pairwise_difference, float(left_brier - right_brier))

    point_values = np.asarray(list(task_values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.full((samples, 2), np.nan)
    for sample in range(samples):
        selected = []
        for suite, tasks in sorted(by_suite.items()):
            selected.extend(
                tasks[index]
                for index in rng.integers(0, len(tasks), size=len(tasks))
            )
        values = np.asarray([task_values[task] for task in selected], dtype=float)
        finite = values[np.isfinite(values[:, 0]), 0]
        bootstrap[sample, 0] = float(np.mean(finite)) if len(finite) else np.nan
        bootstrap[sample, 1] = float(np.mean(values[:, 1]))
    finite_bootstrap = bootstrap[np.isfinite(bootstrap[:, 0]), 0]
    return {
        "pairwise_accuracy_difference": float(np.nanmean(point_values[:, 0])),
        "pairwise_accuracy_difference_95ci": np.quantile(
            finite_bootstrap, [0.025, 0.975]
        ).tolist(),
        "brier_difference": float(np.mean(point_values[:, 1])),
        "brier_difference_95ci": np.quantile(
            bootstrap[:, 1], [0.025, 0.975]
        ).tolist(),
        "informative_pairwise_task_count": int(
            np.isfinite(point_values[:, 0]).sum()
        ),
        "total_task_count": len(task_values),
    }


def _method(
    attack_rank: np.ndarray,
    utility_rank: np.ndarray,
    attack_probability: np.ndarray,
    utility_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "attack_rank": np.asarray(attack_rank, dtype=float),
        "utility_rank": np.asarray(utility_rank, dtype=float),
        "attack_probability": np.clip(attack_probability, EPSILON, 1 - EPSILON),
        "utility_probability": np.clip(utility_probability, EPSILON, 1 - EPSILON),
    }


def _c_name(value: float) -> str:
    return str(value).replace(".", "p")


def _build_methods(
    train_rows: list[dict[str, Any]],
    predict_rows: list[dict[str, Any]],
    train_labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_matrices: dict[str, np.ndarray],
    predict_matrices: dict[str, np.ndarray],
    pointwise_predictions: dict[str, dict[str, np.ndarray]],
    selected_calibrators: dict[str, str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    attack_y, utility_y, _ = train_labels
    groups = np.asarray(["::".join(_task_key(row)) for row in train_rows])
    clean_attack, _ = _raw_scores(
        predict_rows, "clean_prefix_rollout", "risk_score"
    )
    clean_utility, _ = _raw_scores(
        predict_rows, "clean_prefix_rollout", "utility_score"
    )
    injection_attack, injection_std = _raw_scores(
        predict_rows, "injection_conditioned_rollout", "risk_score"
    )
    calibrated_attack = pointwise_predictions[selected_calibrators["attack"]][
        "attack"
    ]
    calibrated_utility = pointwise_predictions[selected_calibrators["utility"]][
        "utility"
    ]
    text_attack = pointwise_predictions["text"]["attack"]
    text_utility = pointwise_predictions["text"]["utility"]
    borda = _borda_attack(predict_rows)

    methods = {
        "clean_raw": _method(
            clean_attack, clean_utility, clean_attack, clean_utility
        ),
        "text_pointwise": _method(
            text_attack, text_utility, text_attack, text_utility
        ),
        "dual_raw": _method(
            injection_attack, clean_utility, injection_attack, clean_utility
        ),
        "dual_lcb_1std": _method(
            injection_attack - injection_std,
            clean_utility,
            injection_attack,
            clean_utility,
        ),
        "dual_borda": _method(
            borda,
            clean_utility,
            calibrated_attack,
            calibrated_utility,
        ),
    }
    pair_counts = {}
    pairwise_scores: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for feature_set in ("text", "world", "combined"):
        for c_value in C_VALUES:
            attack_score, attack_pairs = _pairwise_rank_predict(
                train_matrices[feature_set],
                attack_y,
                groups,
                predict_matrices[feature_set],
                c_value=c_value,
            )
            utility_score, utility_pairs = _pairwise_rank_predict(
                train_matrices[feature_set],
                utility_y,
                groups,
                predict_matrices[feature_set],
                c_value=c_value,
            )
            pairwise_scores[(feature_set, c_value)] = (
                attack_score,
                utility_score,
            )
            pair_counts[f"{feature_set}_c{_c_name(c_value)}"] = (
                attack_pairs + utility_pairs
            )
            name = f"{feature_set}_pairwise_c{_c_name(c_value)}"
            if feature_set == "text":
                methods[name] = _method(
                    attack_score,
                    utility_score,
                    text_attack,
                    text_utility,
                )
            else:
                methods[name] = _method(
                    attack_score,
                    clean_utility,
                    calibrated_attack,
                    calibrated_utility,
                )
    text_rank = _within_task_rank(predict_rows, text_attack)
    for alpha in BLEND_ALPHAS:
        name = f"text_borda_alpha_{str(alpha).replace('.', 'p')}"
        methods[name] = _method(
            (1.0 - alpha) * text_rank + alpha * borda,
            clean_utility,
            calibrated_attack,
            calibrated_utility,
        )
    return methods, pair_counts


def _serialize_method(method: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(values[index]) for name, values in method.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--eval-archive", type=Path, required=True)
    parser.add_argument("--train-steps", type=Path)
    parser.add_argument("--validation-steps", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows = _load_split(args.train_archive, "train")
    validation_rows = _load_split(args.eval_archive, "val")
    test_rows = _load_split(args.eval_archive, "test")
    task_sets = {
        name: {_task_key(row) for row in rows}
        for name, rows in (
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        )
    }
    if task_sets["train"] & task_sets["validation"]:
        raise ValueError("Training and validation tasks overlap")
    if task_sets["train"] & task_sets["test"]:
        raise ValueError("Training and test tasks overlap")
    if task_sets["validation"] & task_sets["test"]:
        raise ValueError("Validation and test tasks overlap")

    train_labels = _labels(train_rows, _soft_targets(args.train_steps))
    validation_labels = _labels(
        validation_rows, _soft_targets(args.validation_steps)
    )
    matrices = {}
    for split, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        text = _text_matrix(rows)
        world = _world_matrix(rows)
        matrices[split] = {
            "text": text,
            "world": world,
            "combined": np.column_stack([text, world]),
        }

    attack_y, utility_y, utility_weight = train_labels
    groups = np.asarray(["::".join(_task_key(row)) for row in train_rows])
    oof = {}
    oof_brier = {"attack": {}, "utility": {}}
    clean_train_attack, _ = _raw_scores(
        train_rows, "clean_prefix_rollout", "risk_score"
    )
    clean_train_utility, _ = _raw_scores(
        train_rows, "clean_prefix_rollout", "utility_score"
    )
    oof["clean_raw"] = {
        "attack": clean_train_attack,
        "utility": clean_train_utility,
    }
    for feature_set in ("text", "world", "combined"):
        oof[feature_set] = {
            "attack": _grouped_oof_probability(
                matrices["train"][feature_set],
                attack_y,
                np.ones(len(attack_y)),
                groups,
            ),
            "utility": _grouped_oof_probability(
                matrices["train"][feature_set],
                utility_y,
                utility_weight,
                groups,
            ),
        }
    for model, predictions in oof.items():
        oof_brier["attack"][model] = float(
            np.mean((predictions["attack"] - attack_y) ** 2)
        )
        oof_brier["utility"][model] = float(
            np.average(
                (predictions["utility"] - utility_y) ** 2,
                weights=utility_weight,
            )
        )
    selected_calibrators = {
        head: min(values, key=lambda name: (values[name], name))
        for head, values in oof_brier.items()
    }

    pointwise: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "validation": {},
        "test": {},
    }
    for model in ("text", "world", "combined"):
        combined_predict = np.concatenate(
            [matrices["validation"][model], matrices["test"][model]], axis=0
        )
        attack_prediction = _fit_soft_probability(
            matrices["train"][model],
            attack_y,
            combined_predict,
            weights=np.ones(len(attack_y)),
        )
        utility_prediction = _fit_soft_probability(
            matrices["train"][model],
            utility_y,
            combined_predict,
            weights=utility_weight,
        )
        boundary = len(validation_rows)
        pointwise["validation"][model] = {
            "attack": attack_prediction[:boundary],
            "utility": utility_prediction[:boundary],
        }
        pointwise["test"][model] = {
            "attack": attack_prediction[boundary:],
            "utility": utility_prediction[boundary:],
        }
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        clean_attack, _ = _raw_scores(
            rows, "clean_prefix_rollout", "risk_score"
        )
        clean_utility, _ = _raw_scores(
            rows, "clean_prefix_rollout", "utility_score"
        )
        pointwise[split]["clean_raw"] = {
            "attack": clean_attack,
            "utility": clean_utility,
        }

    validation_methods, pair_counts = _build_methods(
        train_rows,
        validation_rows,
        train_labels,
        matrices["train"],
        matrices["validation"],
        pointwise["validation"],
        selected_calibrators,
    )
    test_methods, _ = _build_methods(
        train_rows,
        test_rows,
        train_labels,
        matrices["train"],
        matrices["test"],
        pointwise["test"],
        selected_calibrators,
    )
    expected_budget = 17
    if len(validation_methods) != expected_budget:
        raise AssertionError(
            f"Fixed candidate budget changed: {len(validation_methods)} != {expected_budget}"
        )

    validation_attack, validation_utility, _ = validation_labels
    results = {
        name: _method_metrics(
            validation_rows,
            validation_attack,
            validation_utility,
            method,
        )
        for name, method in validation_methods.items()
    }
    text_candidates = [
        "text_pointwise",
        *[
            f"text_pairwise_c{_c_name(c_value)}" for c_value in C_VALUES
        ],
    ]
    world_candidates = [
        "dual_raw",
        "dual_lcb_1std",
        "dual_borda",
        *[
            f"world_pairwise_c{_c_name(c_value)}" for c_value in C_VALUES
        ],
        *[
            f"combined_pairwise_c{_c_name(c_value)}" for c_value in C_VALUES
        ],
        *[
            f"text_borda_alpha_{str(alpha).replace('.', 'p')}"
            for alpha in BLEND_ALPHAS
        ],
    ]

    def selection_key(name: str):
        return (
            results[name]["primary_mean_within_task_pairwise_accuracy"],
            -results[name]["mean_pair_soft_brier"],
            name,
        )

    selected_text = max(text_candidates, key=selection_key)
    selected_world = max(world_candidates, key=selection_key)
    comparisons = {
        f"{selected_world}__minus__clean_raw": _bootstrap_difference(
            validation_rows,
            validation_methods[selected_world],
            validation_methods["clean_raw"],
            validation_attack,
            validation_utility,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        f"{selected_world}__minus__{selected_text}": _bootstrap_difference(
            validation_rows,
            validation_methods[selected_world],
            validation_methods[selected_text],
            validation_attack,
            validation_utility,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        ),
    }
    clean_comparison = comparisons[f"{selected_world}__minus__clean_raw"]
    text_comparison = comparisons[f"{selected_world}__minus__{selected_text}"]
    selected_brier = results[selected_world]["mean_pair_soft_brier"]
    reference_brier = min(
        results["clean_raw"]["mean_pair_soft_brier"],
        results[selected_text]["mean_pair_soft_brier"],
    )
    gate_checks = {
        "pairwise_gain_vs_clean_at_least_0p03": (
            clean_comparison["pairwise_accuracy_difference"] >= 0.03
        ),
        "pairwise_gain_vs_best_text_at_least_0p03": (
            text_comparison["pairwise_accuracy_difference"] >= 0.03
        ),
        "pairwise_ci_lower_vs_clean_at_least_minus_0p02": (
            clean_comparison["pairwise_accuracy_difference_95ci"][0] >= -0.02
        ),
        "pairwise_ci_lower_vs_text_at_least_minus_0p02": (
            text_comparison["pairwise_accuracy_difference_95ci"][0] >= -0.02
        ),
        "brier_within_0p01_of_best_reference": (
            selected_brier <= reference_brier + 0.01
        ),
        "at_least_6_informative_tasks": min(
            clean_comparison["informative_pairwise_task_count"],
            text_comparison["informative_pairwise_task_count"],
        )
        >= 6,
    }
    gate_status = "GO" if all(gate_checks.values()) else "NO-GO"

    protocol_core = {
        "scope": "grouped_train_pairwise_rank_probability_decoupling",
        "fixed_candidate_budget": expected_budget,
        "training_task_count": len(task_sets["train"]),
        "validation_task_count": len(task_sets["validation"]),
        "test_task_count": len(task_sets["test"]),
        "training_validation_test_task_overlap": 0,
        "validation_labels_used_for_candidate_selection": True,
        "test_labels_used_for_fit_selection_or_metrics": False,
        "training_attack_target": "single historical AgentDojo security outcome",
        "training_utility_target": (
            "continuous utility_probability_target when available, otherwise "
            "historical utility outcome"
        ),
        "probability_calibrator_selection": (
            "five-fold user-task-grouped OOF minimum Brier on training tasks"
        ),
        "ranking_training": "within-user-task pairwise logistic differences only",
        "selected_probability_calibrators": selected_calibrators,
        "selected_text_counterbaseline": selected_text,
        "selected_world_method": selected_world,
        "gate_status": gate_status,
        "gate_checks": gate_checks,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        **protocol_core,
        "protocol_sha256": protocol_hash,
        "train_pair_count": len(train_rows),
        "validation_pair_count": len(validation_rows),
        "test_pair_count": len(test_rows),
        "probability_oof_brier": oof_brier,
        "pairwise_training_pair_counts": pair_counts,
        "validation_results": results,
        "validation_comparisons": comparisons,
        "gate_rule": (
            "GO iff selected world-assisted method gains >=0.03 pairwise against "
            "clean and strongest text-only validation comparator, both CI lower "
            ">=-0.02, Brier <= best reference +0.01, and >=6 tasks informative"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("grouped_train_hybrid_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("frozen_protocol.json").write_text(
        json.dumps({**protocol_core, "protocol_sha256": protocol_hash}, indent=2),
        encoding="utf-8",
    )

    def prediction_payload(
        rows: list[dict[str, Any]],
        methods: dict[str, dict[str, np.ndarray]],
        *,
        include_labels: bool,
    ) -> dict[str, Any]:
        pairs = []
        for index, row in enumerate(rows):
            item: dict[str, Any] = {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "trajectory_id": row["trajectory_id"],
                "source_trace": row["source_trace"],
                "selected_world_method": _serialize_method(
                    methods[selected_world], index
                ),
                "clean_raw": _serialize_method(methods["clean_raw"], index),
                "selected_text_counterbaseline": _serialize_method(
                    methods[selected_text], index
                ),
                "all_methods": {
                    name: _serialize_method(method, index)
                    for name, method in methods.items()
                },
            }
            if include_labels:
                item["observed_attack_target"] = float(validation_attack[index])
                item["observed_utility_target"] = float(validation_utility[index])
            pairs.append(item)
        return {
            "protocol_sha256": protocol_hash,
            "selected_world_method": selected_world,
            "selected_text_counterbaseline": selected_text,
            "labels_included": include_labels,
            "pairs": pairs,
        }

    args.output_dir.joinpath("validation_predictions.json").write_text(
        json.dumps(
            prediction_payload(
                validation_rows, validation_methods, include_labels=True
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_dir.joinpath("test_predictions_label_blind.json").write_text(
        json.dumps(
            prediction_payload(test_rows, test_methods, include_labels=False),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
