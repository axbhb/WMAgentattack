"""Evaluate clean, injection-conditioned, and dual-view task transfer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_metric_module():
    path = ROOT / "scripts" / "38_evaluate_hierarchical_contrast_models.py"
    spec = importlib.util.spec_from_file_location("contrast_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_metric_module()
SEEDS = (7, 13, 21)
MODES = ("clean_prefix_rollout", "injection_conditioned_rollout")


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidates missing in {path}")
    return rows


def _align(
    archive: Path, split: str
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[tuple[str, str, str], dict[str, Any]]]]]:
    mappings: dict[
        str, dict[int, dict[tuple[str, str, str], dict[str, Any]]]
    ] = {mode: {} for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            rows = _load_candidates(
                archive / f"seed{seed}" / f"{split}_{mode}_candidates.json"
            )
            mapping = {_pair_key(row): row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError(f"Duplicate candidate key for {split}/{mode}/{seed}")
            mappings[mode][seed] = mapping
    key_sets = [
        set(mapping)
        for mode_mappings in mappings.values()
        for mapping in mode_mappings.values()
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(f"Candidate sources do not align for {split}")
    reference = mappings[MODES[0]][SEEDS[0]]
    rows = []
    for key in sorted(key_sets[0]):
        row = reference[key]
        observed = {
            (
                bool(mappings[mode][seed][key]["observed_security"]),
                bool(mappings[mode][seed][key]["observed_utility"]),
            )
            for mode in MODES
            for seed in SEEDS
        }
        if len(observed) != 1:
            raise ValueError(f"Observed labels disagree at {key}")
        rows.append(
            {
                "suite": key[0],
                "user_task_id": key[1],
                "injection_task_id": key[2],
                "observed_security": bool(row["observed_security"]),
                "observed_utility": bool(row["observed_utility"]),
            }
        )
    return rows, mappings


def _predictions(
    rows: list[dict[str, Any]],
    mappings: dict[str, dict[int, dict[tuple[str, str, str], dict[str, Any]]]],
    seeds: tuple[int, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def mean_score(mode: str, field: str) -> np.ndarray:
        return np.asarray(
            [
                np.mean(
                    [mappings[mode][seed][_pair_key(row)][field] for seed in seeds]
                )
                for row in rows
            ],
            dtype=float,
        )

    clean_attack = mean_score("clean_prefix_rollout", "risk_score")
    clean_utility = mean_score("clean_prefix_rollout", "utility_score")
    injection_attack = mean_score(
        "injection_conditioned_rollout", "risk_score"
    )
    injection_utility = mean_score(
        "injection_conditioned_rollout", "utility_score"
    )
    return {
        "clean_view": (clean_attack, clean_utility),
        "injection_view": (injection_attack, injection_utility),
        "dual_view": (injection_attack, clean_utility),
        "symmetric_shrinkage_alpha_0_5": (
            0.5 * (clean_attack + injection_attack),
            0.5 * (clean_utility + injection_utility),
        ),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    attack_rates = np.asarray(
        [float(row["observed_security"]) for row in rows]
    )
    utility_rates = np.asarray(
        [float(row["observed_utility"]) for row in rows]
    )
    attempts = [
        [(int(row["observed_security"]), int(row["observed_utility"]))]
        for row in rows
    ]
    results = {
        name: METRICS._evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            values[0],
            values[1],
        )
        for name, values in predictions.items()
    }

    def compare(left: str, right: str) -> dict[str, Any]:
        left_values = predictions[left]
        right_values = predictions[right]
        return METRICS._task_bootstrap_difference(
            rows,
            left_values[0],
            left_values[1],
            right_values[0],
            right_values[1],
            attack_rates,
            utility_rates,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    return {
        "results": results,
        "comparisons": {
            "dual_view__minus__clean_view": compare(
                "dual_view", "clean_view"
            ),
            "injection_view__minus__clean_view": compare(
                "injection_view", "clean_view"
            ),
            "symmetric_shrinkage_alpha_0_5__minus__clean_view": compare(
                "symmetric_shrinkage_alpha_0_5", "clean_view"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_results = {}
    for split in ("val", "test"):
        rows, mappings = _align(args.archive, split)
        ensemble = _predictions(rows, mappings, SEEDS)
        per_seed = {
            str(seed): _evaluate(
                rows,
                _predictions(rows, mappings, (seed,)),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + seed,
            )
            for seed in SEEDS
        }
        split_results[split] = {
            "pair_count": len(rows),
            "task_count": len(
                {(row["suite"], row["user_task_id"]) for row in rows}
            ),
            "ensemble": _evaluate(
                rows,
                ensemble,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "per_seed": per_seed,
        }
    output = {
        "scope": "grouped_unseen_user_task_transfer_analysis",
        "world_model_training_task_overlap": 0,
        "seeds": list(SEEDS),
        "fixed_views": {
            "clean_view": "clean-prefix risk and utility heads",
            "injection_view": "injection-conditioned risk and utility heads",
            "dual_view": "injection-conditioned risk plus clean-prefix utility",
            "symmetric_shrinkage_alpha_0_5": "equal probability blend per head",
        },
        "status": "exploratory_after_inspecting_test_candidate_labels",
        "splits": split_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
