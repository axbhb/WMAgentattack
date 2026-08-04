"""Frozen top-heavy learning-to-rank probes for AgentDojo-v2.

The existing E5 control spends its fixed pair budget on the largest target
gaps.  This probe keeps the representation and ridge alpha fixed and changes
only which within-task comparisons receive training weight.  Every invocation
fits train+validation and evaluates one held-out outer fold without retuning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROBE = _load_module(
    "semantic_probe_top_heavy",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)
DUAL = _load_module(
    "dual_probe_top_heavy",
    ROOT / "scripts" / "87_probe_v2_dual_component_value.py",
)


PAIR_SCHEMES = ("largest_gap_control", "top_anchor", "lambda_ndcg3")
ALPHA = 10.0
MIN_GAP = 0.1
MAX_PAIRS_PER_TASK = 64
NDCG_K = 3


def _task_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["task_key"])].append(index)
    return dict(grouped)


def _ndcg_pair_weights(
    indices: list[int], rows: list[dict[str, Any]], *, k: int
) -> dict[tuple[int, int], float]:
    ideal = sorted(
        indices,
        key=lambda index: (-float(rows[index]["target"]), str(rows[index]["group_id"])),
    )
    position = {index: rank for rank, index in enumerate(ideal)}
    gains = {
        index: math.pow(2.0, float(rows[index]["target"])) - 1.0
        for index in indices
    }

    def discount(rank: int) -> float:
        return 1.0 / math.log2(rank + 2.0) if rank < k else 0.0

    ideal_dcg = sum(gains[index] * discount(rank) for rank, index in enumerate(ideal))
    if ideal_dcg <= 1e-12:
        return {}
    weights = {}
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            difference = float(rows[left]["target"] - rows[right]["target"])
            if abs(difference) + 1e-12 < MIN_GAP:
                continue
            high, low = (left, right) if difference > 0 else (right, left)
            delta = abs(
                (gains[high] - gains[low])
                * (discount(position[high]) - discount(position[low]))
                / ideal_dcg
            )
            if delta > 1e-12:
                weights[(high, low)] = delta
    return weights


def _pair_examples(
    rows: list[dict[str, Any]],
    scheme: str,
    *,
    max_per_task: int = MAX_PAIRS_PER_TASK,
) -> tuple[list[tuple[int, int, float, float]], dict[str, int]]:
    if scheme not in PAIR_SCHEMES:
        raise ValueError(scheme)
    examples: list[tuple[int, int, float, float]] = []
    counts: dict[str, int] = {}
    for task, indices in sorted(_task_indices(rows).items()):
        candidates: list[tuple[int, int, float, float]] = []
        maximum = max(float(rows[index]["target"]) for index in indices)
        ndcg_weights = (
            _ndcg_pair_weights(indices, rows, k=NDCG_K)
            if scheme == "lambda_ndcg3"
            else {}
        )
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                difference = float(rows[left]["target"] - rows[right]["target"])
                if abs(difference) + 1e-12 < MIN_GAP:
                    continue
                high, low = (left, right) if difference > 0 else (right, left)
                gap = abs(difference)
                if scheme == "top_anchor" and not math.isclose(
                    float(rows[high]["target"]), maximum, abs_tol=1e-12
                ):
                    continue
                weight = ndcg_weights.get((high, low), 0.0) if ndcg_weights else 1.0
                if scheme == "lambda_ndcg3" and weight <= 1e-12:
                    continue
                candidates.append((high, low, gap / 2.0, weight))
        if scheme == "lambda_ndcg3":
            candidates.sort(key=lambda item: (-item[3], -item[2], item[0], item[1]))
        else:
            candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        candidates = candidates[:max_per_task]
        if scheme == "lambda_ndcg3" and candidates:
            mean_weight = float(np.mean([row[3] for row in candidates]))
            candidates = [
                (high, low, response, weight / mean_weight)
                for high, low, response, weight in candidates
            ]
        examples.extend(candidates)
        counts[task] = len(candidates)
    if not examples:
        raise ValueError(f"No pair examples found for {scheme}")
    return examples, counts


def _fit(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    scheme: str,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    target = np.asarray([float(row["target"]) / 2.0 for row in rows])
    feature_mean = matrix.mean(axis=0)
    centered = np.asarray(matrix, dtype=np.float64) - feature_mean
    pairs, counts = _pair_examples(rows, scheme)
    design = np.stack([centered[high] - centered[low] for high, low, _, _ in pairs])
    response = np.asarray([value for _, _, value, _ in pairs])
    weights = np.asarray([weight for _, _, _, weight in pairs])
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_response = response * root_weight
    u, singular, vt = np.linalg.svd(weighted_design, full_matrices=False)
    coefficient = vt.T @ (
        (singular / (np.square(singular) + alpha)) * (u.T @ weighted_response)
    )

    raw_train = centered @ coefficient
    raw_centered = raw_train - raw_train.mean()
    variance = float(np.sum(np.square(raw_centered)))
    covariance = float(np.sum(raw_centered * (target - target.mean())))
    calibration_slope = max(0.0, covariance / variance) if variance > 1e-12 else 0.0
    calibration_intercept = float(target.mean() - calibration_slope * raw_train.mean())
    return {
        "scheme": scheme,
        "alpha": float(alpha),
        "feature_mean": feature_mean,
        "coefficient": coefficient,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "fit_summary": {
            "pair_count": len(pairs),
            "pairs_per_task": counts,
            "pair_weight_min": float(weights.min()),
            "pair_weight_mean": float(weights.mean()),
            "pair_weight_max": float(weights.max()),
        },
    }


def _predict(model: dict[str, Any], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = (
        np.asarray(matrix, dtype=np.float64) - model["feature_mean"]
    ) @ model["coefficient"]
    normalized = model["calibration_intercept"] + model["calibration_slope"] * raw
    return raw, np.clip(2.0 * normalized, 0.0, 2.0)


def evaluate_fold(
    rows: dict[str, list[dict[str, Any]]],
    matrices: dict[str, np.ndarray],
    *,
    scheme: str,
) -> dict[str, Any]:
    fit_rows = rows["train"] + rows["val"]
    fit_matrix = np.concatenate((matrices["train"], matrices["val"]), axis=0)
    model = _fit(fit_matrix, fit_rows, scheme=scheme)
    rank_scores, predictions = _predict(model, matrices["test"])
    metrics = PROBE._evaluate(
        rows["test"], rank_scores=rank_scores, predictions=predictions
    )
    candidate = {
        "pair_scheme": scheme,
        "alpha": ALPHA,
        "representation": "e5_structured",
        "view": "full",
    }
    return {
        "scope": "frozen top-heavy E5 ranking fold evaluation",
        "protocol": {
            "frozen_candidate": candidate,
            "fit_scope": "train_plus_validation",
            "test_retuning": False,
            "decision_step": "first",
            "ndcg_k": NDCG_K if scheme == "lambda_ndcg3" else None,
        },
        "fit_summary": model["fit_summary"],
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
                "rank_score": float(rank_scores[index]),
                "prediction": float(predictions[index]),
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
    parser.add_argument("--pair-scheme", choices=PAIR_SCHEMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        split: PROBE._configuration_rows(
            PROBE._steps(args.data_root / f"{split}_steps.jsonl")
        )
        for split in ("train", "val", "test")
    }
    matrices, structured_vocab = DUAL._build_matrices(
        rows,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
        domain_interactions=False,
    )
    result = evaluate_fold(rows, matrices, scheme=args.pair_scheme)
    result["counts"] = {
        split: {
            "configurations": len(rows[split]),
            "tasks": len({row["task_key"] for row in rows[split]}),
        }
        for split in ("train", "val", "test")
    }
    result["structured_vocabulary"] = structured_vocab
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
