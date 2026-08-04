"""Post-hoc aggregation and label-stability diagnostics for experiment 0713.

This script is intentionally separate from the confirmatory evaluator.  None of
its alternatives may be reported as a confirmation result because they were
examined after the fresh replay outcomes were available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 13, 21)
PROBABILITY_METHODS = (
    "mean_probability",
    "median_probability",
    "logit_mean_probability",
    "lower_confidence_bound_1std",
)


def _load_metrics_module():
    path = ROOT / "scripts" / "38_evaluate_hierarchical_contrast_models.py"
    spec = importlib.util.spec_from_file_location("contrast_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_metrics_module()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (*_task_key(row), str(row["injection_task_id"]))


def _seed_matrix(
    rows: list[dict[str, Any]], model: str, head: str
) -> np.ndarray:
    field = f"{head}_probability"
    return np.asarray(
        [
            [
                row["confirmation_seed_predictions"][str(seed)][model][field]
                for row in rows
            ]
            for seed in SEEDS
        ],
        dtype=float,
    )


def _aggregate_probability(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "mean_probability":
        output = matrix.mean(axis=0)
    elif method == "median_probability":
        output = np.median(matrix, axis=0)
    elif method == "logit_mean_probability":
        output = expit(logit(np.clip(matrix, 1e-5, 1.0 - 1e-5)).mean(axis=0))
    elif method == "lower_confidence_bound_1std":
        output = matrix.mean(axis=0) - matrix.std(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return np.clip(output, 1e-5, 1.0 - 1e-5)


def _aggregate_borda(
    rows: list[dict[str, Any]], matrix: np.ndarray
) -> np.ndarray:
    output = np.zeros(len(rows), dtype=float)
    for task in sorted({_task_key(row) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if _task_key(row) == task]
        )
        seed_ranks = []
        for values in matrix[:, indices]:
            denominator = max(len(indices) - 1, 1)
            seed_ranks.append((rankdata(values, method="average") - 1) / denominator)
        output[indices] = np.mean(seed_ranks, axis=0)
    return output


def _method_predictions(
    rows: list[dict[str, Any]], method: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for model in ("clean_view", "dual_view"):
        attack = _seed_matrix(rows, model, "attack")
        utility = _seed_matrix(rows, model, "utility")
        if method == "borda_within_task_rank":
            output[model] = (
                _aggregate_borda(rows, attack),
                _aggregate_borda(rows, utility),
            )
        else:
            output[model] = (
                _aggregate_probability(attack, method),
                _aggregate_probability(utility, method),
            )
    return output


def _comparison(
    rows: list[dict[str, Any]],
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    return METRICS._task_bootstrap_difference(
        rows,
        left[0],
        left[1],
        right[0],
        right[1],
        attack_rates,
        utility_rates,
        samples=samples,
        seed=seed,
    )


def _probability_diagnostic(
    rows: list[dict[str, Any]],
    method: str,
    frozen_clean: tuple[np.ndarray, np.ndarray],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    attempts: list[list[tuple[int, int]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    predictions = _method_predictions(rows, method)
    risk_only_dual = (
        predictions["dual_view"][0],
        frozen_clean[1],
    )
    return {
        "clean": METRICS._evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            predictions["clean_view"][0],
            predictions["clean_view"][1],
        ),
        "dual": METRICS._evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            predictions["dual_view"][0],
            predictions["dual_view"][1],
        ),
        "risk_only_dual_clean_utility": METRICS._evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            risk_only_dual[0],
            risk_only_dual[1],
        ),
        "dual_minus_same_aggregation_clean": _comparison(
            rows,
            predictions["dual_view"],
            predictions["clean_view"],
            attack_rates,
            utility_rates,
            samples=samples,
            seed=seed,
        ),
        "dual_minus_frozen_mean_clean": _comparison(
            rows,
            predictions["dual_view"],
            frozen_clean,
            attack_rates,
            utility_rates,
            samples=samples,
            seed=seed + 100,
        ),
        "risk_only_dual_minus_frozen_mean_clean": _comparison(
            rows,
            risk_only_dual,
            frozen_clean,
            attack_rates,
            utility_rates,
            samples=samples,
            seed=seed + 200,
        ),
    }


def _ranking_diagnostic(
    rows: list[dict[str, Any]],
    frozen_clean: tuple[np.ndarray, np.ndarray],
    attack_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    predictions = _method_predictions(rows, "borda_within_task_rank")
    risk_only_dual = (predictions["dual_view"][0], frozen_clean[1])
    same = _comparison(
        rows,
        predictions["dual_view"],
        predictions["clean_view"],
        attack_rates,
        utility_rates,
        samples=samples,
        seed=seed,
    )
    frozen = _comparison(
        rows,
        risk_only_dual,
        frozen_clean,
        attack_rates,
        utility_rates,
        samples=samples,
        seed=seed + 100,
    )
    for comparison in (same, frozen):
        comparison.pop("brier_difference", None)
        comparison.pop("brier_difference_95ci", None)
    return {
        "calibration_interpretation": (
            "Borda values are ordering scores, not probabilities; Brier is omitted."
        ),
        "dual_minus_same_aggregation_clean": same,
        "dual_minus_frozen_mean_clean": frozen,
        "decoupled_selector": {
            "ordering": (
                "dual-view Borda risk score plus frozen clean-view utility score"
            ),
            "reported_probability": "frozen clean-view probability",
            "pairwise_comparison": frozen,
            "brier_difference_by_construction": 0.0,
            "status": "post_hoc_hypothesis_only",
        },
    }


def _historical_diagnostic(
    rows: list[dict[str, Any]], historical_path: Path
) -> dict[str, Any]:
    historical_rows = _load(historical_path).get("candidates")
    if not isinstance(historical_rows, list):
        raise ValueError("Historical candidates are missing")
    mapping = {_pair_key(row): row for row in historical_rows}
    old_attack = np.asarray(
        [float(mapping[_pair_key(row)]["observed_security"]) for row in rows]
    )
    old_utility = np.asarray(
        [float(mapping[_pair_key(row)]["observed_utility"]) for row in rows]
    )
    new_attack = np.asarray(
        [float(row["observed_attack_probability"]) for row in rows]
    )
    new_utility = np.asarray(
        [float(row["observed_utility_probability"]) for row in rows]
    )

    def informative_task_count(
        attack: np.ndarray, utility: np.ndarray
    ) -> int:
        count = 0
        for task in sorted({_task_key(row) for row in rows}):
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(rows)
                    if _task_key(row) == task
                ]
            )
            if len(np.unique(attack[indices])) > 1 or len(
                np.unique(utility[indices])
            ) > 1:
                count += 1
        return count

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        value = spearmanr(left, right).statistic
        return float(value) if np.isfinite(value) else None

    return {
        "historical_single_attack_mean": float(old_attack.mean()),
        "fresh_five_repeat_attack_mean": float(new_attack.mean()),
        "historical_single_utility_mean": float(old_utility.mean()),
        "fresh_five_repeat_utility_mean": float(new_utility.mean()),
        "historical_label_brier_against_fresh_rate": {
            "attack": float(np.mean((old_attack - new_attack) ** 2)),
            "utility": float(np.mean((old_utility - new_utility) ** 2)),
        },
        "historical_fresh_spearman": {
            "attack": correlation(old_attack, new_attack),
            "utility": correlation(old_utility, new_utility),
        },
        "historical_matches_fresh_majority_rate": {
            "attack": float(np.mean(old_attack == (new_attack > 0.5))),
            "utility": float(np.mean(old_utility == (new_utility > 0.5))),
        },
        "fresh_variable_pair_count": {
            "attack": int(np.sum((new_attack > 0.0) & (new_attack < 1.0))),
            "utility": int(np.sum((new_utility > 0.0) & (new_utility < 1.0))),
        },
        "within_task_ranking_support": {
            "historical_single_informative_task_count": informative_task_count(
                old_attack, old_utility
            ),
            "fresh_five_repeat_informative_task_count": informative_task_count(
                new_attack, new_utility
            ),
            "total_task_count": len({_task_key(row) for row in rows}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--historical-candidates", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load(args.dataset).get("pairs")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("Expected the 32-pair grouped confirmation dataset")
    attack_count, utility_count, trials, attempts = METRICS._outcome_arrays(rows)
    attack_rates = attack_count / trials
    utility_rates = utility_count / trials
    frozen = _method_predictions(rows, "mean_probability")
    frozen_clean = frozen["clean_view"]
    methods = {
        method: _probability_diagnostic(
            rows,
            method,
            frozen_clean,
            attack_rates,
            utility_rates,
            attempts,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + index * 1000,
        )
        for index, method in enumerate(PROBABILITY_METHODS)
    }
    output = {
        "scope": "post_hoc_grouped_confirmation_aggregation_diagnostic",
        "status": "exploratory_after_fresh_outcomes_not_confirmatory",
        "multiple_comparison_warning": (
            "Alternatives were inspected after labels; no best method is selected "
            "and no pass/fail claim is permitted from this file."
        ),
        "probability_aggregations": methods,
        "rank_aggregation": _ranking_diagnostic(
            rows,
            frozen_clean,
            attack_rates,
            utility_rates,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 10000,
        ),
        "historical_label_stability": _historical_diagnostic(
            rows, args.historical_candidates
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
