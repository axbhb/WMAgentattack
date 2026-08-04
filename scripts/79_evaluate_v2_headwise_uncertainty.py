"""Evaluate frozen head-wise UCB/LCB seed aggregation on AgentDojo-v2."""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_stability():
    path = ROOT / "scripts" / "77_evaluate_v2_stability_ensemble.py"
    spec = importlib.util.spec_from_file_location("v2_stability_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STABILITY = _load_stability()
BASE = STABILITY.BASE
AGGREGATOR_ORDER = (
    "mean_score",
    "risk_ucb_0p5",
    "utility_lcb_0p5",
    "asymmetric_ucb_lcb_0p5",
)
UNCERTAINTY_COEFFICIENT = 0.5


def _aggregate_rows(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    *,
    risk_key: str,
    recipe: str,
    budget: int,
) -> dict[str, list[dict[str, Any]]]:
    del budget
    seeds = sorted(rows_by_seed)
    if len(seeds) < 2:
        raise ValueError("Uncertainty aggregation requires at least two seeds")
    maps = {
        seed: {row["group_id"]: row for row in rows_by_seed[seed]}
        for seed in seeds
    }
    group_ids = set(maps[seeds[0]])
    if any(set(maps[seed]) != group_ids for seed in seeds[1:]):
        raise ValueError("Checkpoint seeds produced different configuration ids")
    utility_weight, preservation_weight = BASE.RECIPES[recipe]

    output = {aggregator: [] for aggregator in AGGREGATOR_ORDER}
    for group_id in sorted(group_ids):
        reference = maps[seeds[0]][group_id]
        values = {
            "risk": [float(maps[seed][group_id][risk_key]) for seed in seeds],
            "utility": [
                float(maps[seed][group_id]["utility_score"]) for seed in seeds
            ],
            "preservation": [
                float(maps[seed][group_id]["preservation_score"])
                for seed in seeds
            ],
        }
        means = {key: statistics.fmean(head) for key, head in values.items()}
        stds = {key: statistics.pstdev(head) for key, head in values.items()}
        risk_ucb = means["risk"] + UNCERTAINTY_COEFFICIENT * stds["risk"]
        utility_lcb = (
            means["utility"]
            - UNCERTAINTY_COEFFICIENT * stds["utility"]
        )
        preservation_lcb = (
            means["preservation"]
            - UNCERTAINTY_COEFFICIENT * stds["preservation"]
        )
        aggregate_scores = {
            "mean_score": (
                means["risk"]
                + utility_weight * means["utility"]
                + preservation_weight * means["preservation"]
            ),
            "risk_ucb_0p5": (
                risk_ucb
                + utility_weight * means["utility"]
                + preservation_weight * means["preservation"]
            ),
            "utility_lcb_0p5": (
                means["risk"]
                + utility_weight * utility_lcb
                + preservation_weight * preservation_lcb
            ),
            "asymmetric_ucb_lcb_0p5": (
                risk_ucb
                + utility_weight * utility_lcb
                + preservation_weight * preservation_lcb
            ),
        }
        for aggregator, score in aggregate_scores.items():
            output[aggregator].append(
                {
                    **reference,
                    "decision_score": float(score),
                    "aggregation_score": float(score),
                    "head_seed_values": {
                        key: {
                            str(seed): float(value)
                            for seed, value in zip(seeds, head, strict=True)
                        }
                        for key, head in values.items()
                    },
                    "head_means": {key: float(value) for key, value in means.items()},
                    "head_stds": {key: float(value) for key, value in stds.items()},
                }
            )
    return output


STABILITY.AGGREGATOR_ORDER = AGGREGATOR_ORDER
STABILITY._aggregate_rows = _aggregate_rows
STABILITY.EXPERIMENT_SCOPE = "frozen head-wise uncertainty ensemble evaluation"
STABILITY.AGGREGATOR_PARAMETERS = {
    "uncertainty_coefficient": UNCERTAINTY_COEFFICIENT,
    "score_clipping": False,
    "risk_rule": "mean plus coefficient times population standard deviation",
    "utility_rule": "mean minus coefficient times population standard deviation",
    "preservation_rule": "mean minus coefficient times population standard deviation",
}


if __name__ == "__main__":
    STABILITY.main()
