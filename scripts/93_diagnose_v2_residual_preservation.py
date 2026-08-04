"""Post-hoc mechanism diagnostics for frozen residual-preservation probes.

This script never refits a model.  It replays a small, fixed uncertainty-
penalty ablation from the candidate predictions saved by the five frozen OOF
folds.  Because held-out outcomes are read, every result is diagnostic only
and must not be reported as a prospectively selected replacement method.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    "semantic_probe_residual_diagnostic",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _metric_subset(
    rows: list[dict[str, Any]], rank_scores: np.ndarray, indices: list[int]
) -> dict[str, Any]:
    subset = [rows[index] for index in indices]
    scores = rank_scores[np.asarray(indices, dtype=np.int64)]
    predictions = np.asarray([float(row["prediction"]) for row in subset])
    metrics = PROBE._evaluate(subset, rank_scores=scores, predictions=predictions)
    return {
        "tasks": int(metrics["task_count"]),
        "ASR": float(metrics["top1_ASR"]),
        "BUP": float(metrics["top1_BUP"]),
        "ASR_plus_BUP": float(metrics["top1_ASR_plus_BUP"]),
        "target_ASR_plus_BUP": float(metrics["top1_target_ASR_plus_BUP"]),
        "mean_task_spearman": metrics["mean_task_spearman"],
    }


def _selection_map(metrics: dict[str, Any]) -> dict[str, str]:
    return {
        str(task): str(row["selected_group_id"])
        for task, row in metrics["per_task"].items()
    }


def _diagnose_family(
    paths: list[Path], penalties: tuple[float, ...]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidates: set[str] = set()
    task_to_fold: dict[str, int] = {}
    for fold, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["protocol"]["frozen_candidate"]
        candidates.add(json.dumps(candidate, sort_keys=True))
        for row in payload["test_candidate_scores"]:
            task = str(row["task_key"])
            previous = task_to_fold.setdefault(task, fold)
            if previous != fold:
                raise ValueError(f"Task occurs in multiple folds: {task}")
            rows.append(row)
    if len(candidates) != 1:
        raise ValueError("Fold files do not share one frozen candidate")
    candidate = json.loads(next(iter(candidates)))
    weight = float(candidate["utility_weight"])

    utility = np.asarray([float(row["utility_prediction"]) for row in rows])
    standard_deviation = np.asarray(
        [float(row["utility_standard_deviation"]) for row in rows]
    )
    attacked_target = np.asarray([float(row["target_bup"]) for row in rows])
    attack = np.asarray([float(row["attack_prediction"]) for row in rows])
    absolute_error = np.abs(utility - attacked_target)

    by_penalty: dict[str, Any] = {}
    selections: dict[str, dict[str, str]] = {}
    for penalty in penalties:
        conservative = np.clip(utility - penalty * standard_deviation, 0.0, 1.0)
        rank_scores = attack + weight * conservative
        metrics = PROBE._evaluate(
            rows,
            rank_scores=rank_scores,
            predictions=np.asarray([float(row["prediction"]) for row in rows]),
        )
        key = str(penalty)
        selections[key] = _selection_map(metrics)

        domain_indices: dict[str, list[int]] = defaultdict(list)
        clean_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            domain_indices[str(row["task_key"]).split("|", 1)[0]].append(index)
            clean_indices[str(int(row["clean_successes"]))].append(index)
        by_penalty[key] = {
            "ASR": float(metrics["top1_ASR"]),
            "BUP": float(metrics["top1_BUP"]),
            "ASR_plus_BUP": float(metrics["top1_ASR_plus_BUP"]),
            "target_ASR_plus_BUP": float(metrics["top1_target_ASR_plus_BUP"]),
            "mean_task_spearman": metrics["mean_task_spearman"],
            "mean_top1_margin": float(metrics["mean_top1_margin"]),
            "selected_group_by_task": selections[key],
            "by_domain": {
                name: _metric_subset(rows, rank_scores, indices)
                for name, indices in sorted(domain_indices.items())
            },
            "by_clean_success_count": {
                name: _metric_subset(rows, rank_scores, indices)
                for name, indices in sorted(clean_indices.items())
            },
        }

    formal_key = str(float(candidate["uncertainty_penalty"]))
    if formal_key not in selections:
        raise ValueError("Penalty grid must include the frozen formal penalty")
    formal_selection = selections[formal_key]
    selection_switches = {}
    for key, selected in selections.items():
        switched = [
            task for task in sorted(formal_selection) if selected[task] != formal_selection[task]
        ]
        selection_switches[key] = {
            "task_count": len(switched),
            "fraction": len(switched) / len(formal_selection),
            "tasks": switched,
        }

    quantiles = np.quantile(standard_deviation, (0.0, 0.25, 0.5, 0.75, 1.0))
    return {
        "frozen_candidate": candidate,
        "counts": {
            "folds": len(paths),
            "tasks": len(task_to_fold),
            "configurations": len(rows),
        },
        "uncertainty_diagnostic": {
            "mean": float(standard_deviation.mean()),
            "quantiles_min_q25_q50_q75_max": [float(value) for value in quantiles],
            "utility_absolute_error_mean": float(absolute_error.mean()),
            "uncertainty_error_pearson": _correlation(
                standard_deviation, absolute_error
            ),
        },
        "penalty_ablation": by_penalty,
        "selection_switches_vs_formal_penalty": selection_switches,
    }


def diagnose(
    archive_root: Path, penalties: tuple[float, ...] = (0.0, 0.5, 1.0)
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family_dir in sorted((archive_root / "frozen").iterdir()):
        if not family_dir.is_dir():
            continue
        paths = sorted(family_dir.glob("fold*/result.json"))
        if not paths:
            continue
        families[family_dir.name] = _diagnose_family(paths, penalties)
    if not families:
        raise ValueError(f"No frozen fold results found under {archive_root}")
    return {
        "scope": "post-hoc held-out residual-preservation mechanism diagnostic",
        "claim_boundary": (
            "Held-out labels are used to diagnose a completed frozen experiment. "
            "No ablation row is a prospectively selected replacement result."
        ),
        "fixed_diagnostic_grid": {
            "uncertainty_penalties": list(penalties),
            "model_predictions_reused": True,
            "model_refit": False,
            "utility_weight": "the previously frozen per-family value",
        },
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.archive_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
