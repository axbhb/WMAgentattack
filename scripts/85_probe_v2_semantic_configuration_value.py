"""Leakage-safe semantic representation probe for configuration value ranking.

The fixed grid compares the current 768-d hashed text representation with one
pretrained 768-d E5 representation.  Hyperparameters are selected only on
validation user tasks, then the chosen model is refit on train+validation and
evaluated once on held-out test tasks.
"""

from __future__ import annotations

import argparse
import hashlib
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

from wmagentattack.dreamer_world_model import hash_text_features
from wmagentattack.dreamer_world_model import step_to_dreamer_text
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


RANK_PATH = ROOT / "scripts" / "83_diagnose_v2_group_utility_rank_stability.py"
RANK_SPEC = importlib.util.spec_from_file_location("rank_stability", RANK_PATH)
RANK = importlib.util.module_from_spec(RANK_SPEC)
assert RANK_SPEC.loader is not None
RANK_SPEC.loader.exec_module(RANK)

ALPHAS = (0.1, 1.0, 10.0, 100.0)
REPRESENTATIONS = (
    "hash",
    "e5",
    "structured",
    "hash_structured",
    "e5_structured",
)
TEXT_VIEWS = ("full", "transferable")
REPRESENTATION_VIEWS = {
    "hash": TEXT_VIEWS,
    "e5": TEXT_VIEWS,
    "structured": ("structured",),
    "hash_structured": TEXT_VIEWS,
    "e5_structured": TEXT_VIEWS,
}
ESTIMATORS = ("pointwise_ridge", "pairwise_ridge")


def _steps(path: Path) -> list[StepRecord]:
    return [StepRecord.model_validate(row) for row in read_jsonl(path)]


def _decision_indices(steps: list[StepRecord]) -> dict[str, int]:
    selected: dict[str, int] = {}
    for index, step in enumerate(steps):
        previous = selected.get(step.trajectory_id)
        if previous is None or step.step_id < steps[previous].step_id:
            selected[step.trajectory_id] = index
    return selected


def _transferable_text(step: StepRecord) -> str:
    descriptions = " ".join(
        f"{skill}: {step.candidate_skill_descriptions.get(skill, '')}"
        for skill in step.candidate_skills
    )
    return "\n".join(
        [
            f"domain: {step.domain}",
            f"goal: {step.user_goal}",
            f"trusted: {step.trusted_instruction}",
            f"observation: {step.current_observation}",
            f"untrusted: {step.untrusted_content or ''}",
            f"previous_skills: {' '.join(step.previous_skills)}",
            f"candidates: {' '.join(step.candidate_skills)}",
            f"candidate_descriptions: {descriptions}",
            f"attack: {step.attack_action or 'NONE'}",
            f"target: {step.target_skill or 'NONE'}",
        ]
    )


def _configuration_rows(steps: list[StepRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[StepRecord]] = defaultdict(list)
    for index in _decision_indices(steps).values():
        step = steps[index]
        if (
            step.multiseed_group_id is not None
            and step.attack_probability_target is not None
            and step.utility_probability_target is not None
        ):
            grouped[str(step.multiseed_group_id)].append(step)
    rows = []
    for group_id in sorted(grouped):
        records = grouped[group_id]
        expected = {int(record.multiseed_trials or 0) for record in records}
        if len(expected) != 1 or len(records) != next(iter(expected)):
            raise ValueError(f"Incomplete configuration group: {group_id}")
        tasks = {(record.domain, record.task_id) for record in records}
        if len(tasks) != 1:
            raise ValueError(f"Configuration spans tasks: {group_id}")
        attack_targets = {float(record.attack_probability_target) for record in records}
        utility_targets = {float(record.utility_probability_target) for record in records}
        if len(attack_targets) != 1 or len(utility_targets) != 1:
            raise ValueError(f"Inconsistent targets: {group_id}")
        domain, task_id = next(iter(tasks))
        target_asr = next(iter(attack_targets))
        target_bup = next(iter(utility_targets))
        observed_asr = float(np.mean([float(record.attack_success) for record in records]))
        observed_bup = float(np.mean([float(record.task_success) for record in records]))
        rows.append(
            {
                "group_id": group_id,
                "task_key": f"{domain}|{task_id}",
                "target_asr": target_asr,
                "target_bup": target_bup,
                "target": target_asr + target_bup,
                "observed_asr": observed_asr,
                "observed_bup": observed_bup,
                "observed": observed_asr + observed_bup,
                "texts": {
                    "full": [step_to_dreamer_text(record) for record in records],
                    "transferable": [
                        _transferable_text(record) for record in records
                    ],
                },
            }
        )
    if not rows:
        raise ValueError("No complete attack configuration groups found")
    return rows


def _mean_group_embeddings(
    rows: list[dict[str, Any]],
    text_embeddings: dict[str, np.ndarray],
    *,
    view: str,
) -> np.ndarray:
    matrix = []
    for row in rows:
        values = [text_embeddings[text] for text in row["texts"][view]]
        mean = np.mean(values, axis=0)
        norm = np.linalg.norm(mean)
        matrix.append(mean / norm if norm > 0 else mean)
    return np.asarray(matrix, dtype=np.float64)


def _configuration_structure(group_id: str) -> tuple[str, str, str]:
    parts = group_id.split("__")
    injection = next(
        (part for part in parts if part.startswith("injection_task_")),
        "injection_task_UNKNOWN",
    )
    family = parts[-2] if len(parts) >= 2 else "family_UNKNOWN"
    return family, injection, f"{family}|{injection}"


def _structured_vocab(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    values = [_configuration_structure(str(row["group_id"])) for row in rows]
    return {
        "family": sorted({value[0] for value in values}),
        "injection": sorted({value[1] for value in values}),
        "combination": sorted({value[2] for value in values}),
    }


def _structured_matrix(
    rows: list[dict[str, Any]], vocab: dict[str, list[str]]
) -> np.ndarray:
    offsets = {
        "family": 0,
        "injection": len(vocab["family"]),
        "combination": len(vocab["family"]) + len(vocab["injection"]),
    }
    width = sum(len(vocab[key]) for key in ("family", "injection", "combination"))
    lookups = {
        key: {value: offsets[key] + index for index, value in enumerate(vocab[key])}
        for key in offsets
    }
    matrix = np.zeros((len(rows), width), dtype=np.float64)
    for row_index, row in enumerate(rows):
        family, injection, combination = _configuration_structure(
            str(row["group_id"])
        )
        for key, value in (
            ("family", family),
            ("injection", injection),
            ("combination", combination),
        ):
            column = lookups[key].get(value)
            if column is not None:
                matrix[row_index, column] = 1.0
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _concatenate_normalized(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    matrix = np.concatenate((left, right), axis=1)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def _hash_embeddings(texts: list[str], dim: int = 768) -> dict[str, np.ndarray]:
    return {text: hash_text_features(text, dim) for text in texts}


def _e5_embeddings(
    texts: list[str],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (model_name + "\n" + "\n".join(texts)).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = cache_dir / f"e5_text_embeddings_{digest}.npz"
    if cache_path.is_file():
        payload = np.load(cache_path)
        matrix = payload["embeddings"]
        return {text: matrix[index] for index, text in enumerate(texts)}

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_text = [f"query: {text}" for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = F.normalize(pooled, p=2, dim=1)
            output.append(pooled.float().cpu().numpy())
    matrix = np.concatenate(output, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=matrix)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {text: matrix[index] for index, text in enumerate(texts)}


def _largest_pairs(
    rows: list[dict[str, Any]], *, min_gap: float = 0.1, max_per_task: int = 64
) -> list[tuple[int, int, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["task_key"])].append(index)
    selected = []
    for task in sorted(grouped):
        candidates = []
        indices = grouped[task]
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                difference = float(rows[left]["target"] - rows[right]["target"])
                if abs(difference) + 1e-12 < min_gap:
                    continue
                high, low = (left, right) if difference > 0 else (right, left)
                candidates.append((high, low, abs(difference)))
        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        selected.extend(candidates[:max_per_task])
    return selected


def _ridge_fit(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    estimator: str,
    alpha: float,
) -> dict[str, Any]:
    target = np.asarray([float(row["target"]) / 2.0 for row in rows])
    feature_mean = matrix.mean(axis=0)
    centered = matrix - feature_mean
    if estimator == "pointwise_ridge":
        design = centered
        response = target - target.mean()
        target_mean = float(target.mean())
    elif estimator == "pairwise_ridge":
        pairs = _largest_pairs(rows)
        if not pairs:
            raise ValueError("Pairwise ridge found no within-task target pairs")
        design = np.stack(
            [centered[high] - centered[low] for high, low, _ in pairs]
        )
        response = np.asarray([gap / 2.0 for _, _, gap in pairs])
        target_mean = float(target.mean())
    else:
        raise ValueError(estimator)
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    weight = vt.T @ (
        (singular / (singular**2 + alpha)) * (u.T @ response)
    )
    raw_train = centered @ weight
    if estimator == "pointwise_ridge":
        calibration_slope = 1.0
        calibration_intercept = target_mean
    else:
        variance = float(np.sum((raw_train - raw_train.mean()) ** 2))
        covariance = float(
            np.sum((raw_train - raw_train.mean()) * (target - target.mean()))
        )
        calibration_slope = max(0.0, covariance / variance) if variance > 1e-12 else 0.0
        calibration_intercept = float(
            target.mean() - calibration_slope * raw_train.mean()
        )
    return {
        "feature_mean": feature_mean,
        "weight": weight,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "pair_count": len(_largest_pairs(rows)) if estimator == "pairwise_ridge" else 0,
    }


def _ridge_predict(model: dict[str, Any], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = (matrix - model["feature_mean"]) @ model["weight"]
    normalized = (
        model["calibration_intercept"] + model["calibration_slope"] * raw
    )
    return raw, np.clip(2.0 * normalized, 0.0, 2.0)


def _spearman(left: list[float], right: list[float]) -> float | None:
    return RANK._spearman(left, right)


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    rank_scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[str(row["task_key"])].append(index)
    per_task = {}
    selected = []
    for task in sorted(by_task):
        indices = by_task[task]
        ranked = sorted(indices, key=lambda index: (-rank_scores[index], rows[index]["group_id"]))
        winner = ranked[0]
        selected.append(winner)
        targets = [float(rows[index]["target"]) for index in indices]
        observed = [float(rows[index]["observed"]) for index in indices]
        per_task[task] = {
            "spearman_target": _spearman(
                [float(rank_scores[index]) for index in indices], targets
            ),
            "top1_target_regret": max(targets) - float(rows[winner]["target"]),
            "top1_observed_regret": max(observed) - float(rows[winner]["observed"]),
            "selected_target": float(rows[winner]["target"]),
            "selected_target_ASR": float(rows[winner]["target_asr"]),
            "selected_target_BUP": float(rows[winner]["target_bup"]),
            "selected_observed": float(rows[winner]["observed"]),
            "selected_ASR": float(rows[winner]["observed_asr"]),
            "selected_BUP": float(rows[winner]["observed_bup"]),
            "selected_group_id": str(rows[winner]["group_id"]),
            "top1_margin": float(rank_scores[winner] - rank_scores[ranked[1]])
            if len(ranked) > 1
            else 0.0,
            "top1_tie_count": sum(
                abs(float(rank_scores[index] - rank_scores[winner])) <= 1e-12
                for index in indices
            ),
        }
    correlations = [
        float(row["spearman_target"])
        for row in per_task.values()
        if row["spearman_target"] is not None
    ]
    target = np.asarray([float(row["target"]) for row in rows])
    return {
        "configuration_count": len(rows),
        "task_count": len(by_task),
        "normalized_brier": float(np.mean(((predictions - target) / 2.0) ** 2)),
        "mae": float(np.mean(np.abs(predictions - target))),
        "mean_task_spearman": float(np.mean(correlations)) if correlations else None,
        "mean_top1_target_regret": float(
            np.mean([row["top1_target_regret"] for row in per_task.values()])
        ),
        "mean_top1_observed_regret": float(
            np.mean([row["top1_observed_regret"] for row in per_task.values()])
        ),
        "top1_target_oracle_rate": float(
            np.mean(
                [row["top1_target_regret"] <= 1e-12 for row in per_task.values()]
            )
        ),
        "top1_target_ASR": float(
            np.mean([rows[index]["target_asr"] for index in selected])
        ),
        "top1_target_BUP": float(
            np.mean([rows[index]["target_bup"] for index in selected])
        ),
        "top1_target_ASR_plus_BUP": float(
            np.mean([rows[index]["target"] for index in selected])
        ),
        "top1_ASR_plus_BUP": float(
            np.mean([rows[index]["observed"] for index in selected])
        ),
        "top1_ASR": float(
            np.mean([rows[index]["observed_asr"] for index in selected])
        ),
        "top1_BUP": float(
            np.mean([rows[index]["observed_bup"] for index in selected])
        ),
        "mean_top1_margin": float(
            np.mean([row["top1_margin"] for row in per_task.values()])
        ),
        "mean_top1_tie_count": float(
            np.mean([row["top1_tie_count"] for row in per_task.values()])
        ),
        "unique_top1_rate": float(
            np.mean([row["top1_tie_count"] == 1 for row in per_task.values()])
        ),
        "per_task": per_task,
    }


def _selection_key(row: dict[str, Any], order: int) -> tuple[float, ...]:
    metrics = row["validation"]
    correlation = metrics["mean_task_spearman"]
    return (
        float(correlation) if correlation is not None else -math.inf,
        -float(metrics["mean_top1_target_regret"]),
        -float(metrics["normalized_brier"]),
        -float(order),
    )


def _fit_frozen_method(
    *,
    rows: dict[str, list[dict[str, Any]]],
    matrices: dict[str, dict[str, dict[str, np.ndarray]]],
    representation: str,
    view: str,
    estimator: str,
    alpha: float,
) -> dict[str, Any]:
    combined_rows = rows["train"] + rows["val"]
    combined_matrix = np.concatenate(
        [
            matrices[representation][view]["train"],
            matrices[representation][view]["val"],
        ],
        axis=0,
    )
    model = _ridge_fit(
        combined_matrix,
        combined_rows,
        estimator=estimator,
        alpha=alpha,
    )
    rank_score, prediction = _ridge_predict(
        model, matrices[representation][view]["test"]
    )
    return {
        "frozen_candidate": {
            "representation": representation,
            "view": view,
            "estimator": estimator,
            "alpha": alpha,
        },
        "fit_scope": "train_plus_validation with method frozen before test",
        "test": _evaluate(
            rows["test"], rank_scores=rank_score, predictions=prediction
        ),
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
                "rank_score": float(rank_score[index]),
                "prediction": float(prediction[index]),
            }
            for index, row in enumerate(rows["test"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--frozen-only", action="store_true")
    parser.add_argument(
        "--frozen-representation", choices=REPRESENTATIONS, default="e5_structured"
    )
    parser.add_argument("--frozen-view", default="full")
    parser.add_argument("--frozen-estimator", choices=ESTIMATORS, default="pairwise_ridge")
    parser.add_argument("--frozen-alpha", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        split: _configuration_rows(_steps(args.data_root / f"{split}_steps.jsonl"))
        for split in ("train", "val", "test")
    }
    all_texts = {
        view: sorted(
            {
                text
                for split_rows in rows.values()
                for row in split_rows
                for text in row["texts"][view]
            }
        )
        for view in TEXT_VIEWS
    }
    embedding_maps: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "hash": {
            view: _hash_embeddings(all_texts[view]) for view in TEXT_VIEWS
        },
        "e5": {},
    }
    for view in TEXT_VIEWS:
        embedding_maps["e5"][view] = _e5_embeddings(
            all_texts[view],
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            batch_size=args.embedding_batch_size,
        )
    matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for representation in ("hash", "e5"):
        matrices[representation] = {
            view: {
                split: _mean_group_embeddings(
                    rows[split], embedding_maps[representation][view], view=view
                )
                for split in ("train", "val", "test")
            }
            for view in TEXT_VIEWS
        }
    structured_vocab = _structured_vocab(rows["train"])
    structured = {
        split: _structured_matrix(rows[split], structured_vocab)
        for split in ("train", "val", "test")
    }
    matrices["structured"] = {"structured": structured}
    for base in ("hash", "e5"):
        matrices[f"{base}_structured"] = {
            view: {
                split: _concatenate_normalized(
                    matrices[base][view][split], structured[split]
                )
                for split in ("train", "val", "test")
            }
            for view in TEXT_VIEWS
        }

    frozen_view = args.frozen_view
    if frozen_view not in REPRESENTATION_VIEWS[args.frozen_representation]:
        parser.error(
            f"{args.frozen_representation} does not support view {frozen_view}"
        )
    frozen_method = _fit_frozen_method(
        rows=rows,
        matrices=matrices,
        representation=args.frozen_representation,
        view=frozen_view,
        estimator=args.frozen_estimator,
        alpha=args.frozen_alpha,
    )

    grid = []
    order = 0
    if not args.frozen_only:
        for representation in REPRESENTATIONS:
            for view in REPRESENTATION_VIEWS[representation]:
                for estimator in ESTIMATORS:
                    for alpha in ALPHAS:
                        model = _ridge_fit(
                            matrices[representation][view]["train"],
                            rows["train"],
                            estimator=estimator,
                            alpha=alpha,
                        )
                        rank_score, prediction = _ridge_predict(
                            model, matrices[representation][view]["val"]
                        )
                        grid.append(
                            {
                                "representation": representation,
                                "view": view,
                                "estimator": estimator,
                                "alpha": alpha,
                                "pair_count": model["pair_count"],
                                "validation": _evaluate(
                                    rows["val"],
                                    rank_scores=rank_score,
                                    predictions=prediction,
                                ),
                                "fixed_order": order,
                            }
                        )
                        order += 1

    selected_by_representation = {}
    for representation in (() if args.frozen_only else REPRESENTATIONS):
        candidates = [
            row for row in grid if row["representation"] == representation
        ]
        selected = max(
            candidates,
            key=lambda row: _selection_key(row, row["fixed_order"]),
        )
        combined_rows = rows["train"] + rows["val"]
        combined_matrix = np.concatenate(
            [
                matrices[representation][selected["view"]]["train"],
                matrices[representation][selected["view"]]["val"],
            ],
            axis=0,
        )
        final_model = _ridge_fit(
            combined_matrix,
            combined_rows,
            estimator=selected["estimator"],
            alpha=float(selected["alpha"]),
        )
        rank_score, prediction = _ridge_predict(
            final_model,
            matrices[representation][selected["view"]]["test"],
        )
        selected_by_representation[representation] = {
            "selected_from_validation": selected,
            "refit_scope": "train_plus_validation_after_hyperparameter_freeze",
            "test": _evaluate(
                rows["test"],
                rank_scores=rank_score,
                predictions=prediction,
            ),
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
                    "rank_score": float(rank_score[index]),
                    "prediction": float(prediction[index]),
                }
                for index, row in enumerate(rows["test"])
            ],
        }

    overall = (
        max(grid, key=lambda row: _selection_key(row, row["fixed_order"]))
        if grid
        else None
    )
    result = {
        "scope": "AgentDojo-v2 semantic configuration-value representation probe",
        "protocol": {
            "split_unit": "held-out user task",
            "decision_step": "first",
            "target": "continuous attack_probability + utility_probability",
            "fixed_representations": list(REPRESENTATIONS),
            "structured_fields": [
                "attack_family",
                "injection_task",
                "attack_family_x_injection_task",
            ],
            "structured_vocabulary_fitted_on": "train only",
            "fixed_text_views": list(TEXT_VIEWS),
            "fixed_estimators": list(ESTIMATORS),
            "fixed_alphas": list(ALPHAS),
            "selection_rule": (
                "maximize validation mean within-task Spearman; tie-break by "
                "lower Top-1 target regret, lower normalized Brier, fixed order"
            ),
            "test_rule": "single frozen refit; no test retuning",
            "frozen_only": args.frozen_only,
            "e5_prefix": "query: (recommended by model card for feature probes)",
        },
        "provenance": {
            "data_root": str(args.data_root.resolve()),
            "model_name": args.model_name,
            "cache_dir": str(args.cache_dir.resolve()),
        },
        "counts": {
            split: {
                "configurations": len(rows[split]),
                "tasks": len({row["task_key"] for row in rows[split]}),
            }
            for split in rows
        },
        "grid": grid,
        "overall_validation_selected_candidate": overall,
        "selected_by_representation": selected_by_representation,
        "frozen_method": frozen_method,
        "semantic_minus_hash_test": {
            metric: float(
                selected_by_representation["e5"]["test"][metric]
                - selected_by_representation["hash"]["test"][metric]
            )
            for metric in (
                "mean_task_spearman",
                "top1_ASR_plus_BUP",
                "mean_top1_target_regret",
                "normalized_brier",
            )
        } if not args.frozen_only else None,
        "structured_augmentation_test": {
            base: {
                metric: float(
                    selected_by_representation[f"{base}_structured"]["test"][metric]
                    - selected_by_representation[base]["test"][metric]
                )
                for metric in (
                    "mean_task_spearman",
                    "top1_ASR_plus_BUP",
                    "mean_top1_target_regret",
                    "normalized_brier",
                    "unique_top1_rate",
                )
            }
            for base in ("hash", "e5")
        } if not args.frozen_only else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
