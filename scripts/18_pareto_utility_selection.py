"""Pareto / epsilon-constraint selection over cached world-model candidates.

The selector treats utility preservation as a constraint and attack/target
scores as the objective:

1. filter candidates with predicted utility >= a threshold;
2. rank the feasible set by risk/target score;
3. fall back to the highest-utility candidates only if the feasible set is too
   small for ``top_k``.

Thresholds can be fixed values or empirical quantiles of the candidate utility
scores.  The script is inference-free: it consumes candidate caches produced by
``11_select_world_model_agentdojo_pairs.py --include-candidates``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["suite"], row["user_task_id"], row["injection_task_id"])


def _dedupe_by_pair(rows: list[dict[str, Any]], max_per_user_task: int) -> list[dict[str, Any]]:
    seen = set()
    per_user_task: dict[tuple[str, str], int] = {}
    output = []
    for row in rows:
        key = _pair_key(row)
        if key in seen:
            continue
        user_key = (row["suite"], row["user_task_id"])
        if max_per_user_task > 0 and per_user_task.get(user_key, 0) >= max_per_user_task:
            continue
        seen.add(key)
        per_user_task[user_key] = per_user_task.get(user_key, 0) + 1
        output.append(row)
    return output


def _objective_score(row: dict[str, Any]) -> float:
    risk = float(row.get("risk_score", 0.0))
    mean_risk = float(row.get("rollout_mean_risk_score", risk))
    target = float(row.get("target_skill_probability", 0.0))
    reached = float(row.get("rollout_target_reached", 0.0))
    return risk + 0.3 * mean_risk + 0.3 * target + 0.2 * reached


def _utility_score(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, row.get("utility_score", 0.0)))


def _threshold(candidates: list[dict[str, Any]], utility_key: str, mode: str, value: float) -> float:
    utilities = np.array([_utility_score(row, utility_key) for row in candidates], dtype=float)
    if mode == "fixed":
        return value
    if mode == "quantile":
        return float(np.quantile(utilities, value))
    raise ValueError(f"Unsupported threshold mode: {mode}")


def _select_pareto(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    utility_key: str,
    threshold: float,
    max_per_user_task: int,
) -> list[dict[str, Any]]:
    rescored = [
        {
            **row,
            "pareto_objective_score": _objective_score(row),
            "pareto_utility_score": _utility_score(row, utility_key),
            "pareto_utility_threshold": threshold,
            "pareto_feasible": _utility_score(row, utility_key) >= threshold,
        }
        for row in candidates
    ]
    feasible = [row for row in rescored if row["pareto_feasible"]]
    infeasible = [row for row in rescored if not row["pareto_feasible"]]
    feasible_sorted = sorted(
        feasible,
        key=lambda row: (row["pareto_objective_score"], row["pareto_utility_score"]),
        reverse=True,
    )
    selected = _dedupe_by_pair(feasible_sorted, max_per_user_task)[:top_k]
    if len(selected) < top_k:
        selected_pairs = {_pair_key(row) for row in selected}
        fallback = [
            row
            for row in sorted(
                infeasible,
                key=lambda row: (row["pareto_utility_score"], row["pareto_objective_score"]),
                reverse=True,
            )
            if _pair_key(row) not in selected_pairs
        ]
        selected.extend(
            _dedupe_by_pair(fallback, max_per_user_task)[: max(0, top_k - len(selected))]
        )
    return selected


def _select_weighted(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    max_per_user_task: int,
) -> list[dict[str, Any]]:
    return _dedupe_by_pair(
        sorted(candidates, key=lambda row: row["selection_score"], reverse=True),
        max_per_user_task,
    )[:top_k]


def _select_random(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    seed: int,
    max_per_user_task: int,
) -> list[dict[str, Any]]:
    rows = list(candidates)
    random.Random(seed).shuffle(rows)
    return _dedupe_by_pair(rows, max_per_user_task)[:top_k]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    asr = np.array([bool(row["observed_security"]) for row in rows], dtype=float)
    bup = np.array([bool(row["observed_utility"]) for row in rows], dtype=float)
    return {
        "count": len(rows),
        "observed_asr": float(asr.mean()) if len(rows) else 0.0,
        "observed_bup": float(bup.mean()) if len(rows) else 0.0,
        "objective_asr_plus_bup": float(asr.mean() + bup.mean()) if len(rows) else 0.0,
        "mean_risk_score": float(np.mean([row.get("risk_score", 0.0) for row in rows])) if rows else 0.0,
        "mean_utility_score": (
            float(np.mean([row.get("utility_score", 0.0) for row in rows])) if rows else 0.0
        ),
        "mean_pareto_utility_score": (
            float(np.mean([row.get("pareto_utility_score", row.get("utility_score", 0.0)) for row in rows]))
            if rows
            else 0.0
        ),
        "feasible_rate": (
            float(np.mean([bool(row.get("pareto_feasible", True)) for row in rows])) if rows else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", default="16,24,32")
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument(
        "--utility-keys",
        default="utility_score,final_utility_score,min_utility_score",
    )
    parser.add_argument("--quantiles", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--fixed-thresholds", default="")
    parser.add_argument("--max-per-user-task", type=int, default=2)
    args = parser.parse_args()

    payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    top_ks = _parse_int_list(args.top_k)
    seeds = _parse_int_list(args.seeds)
    utility_keys = [item.strip() for item in args.utility_keys.split(",") if item.strip()]
    specs: list[tuple[str, float]] = [("quantile", value) for value in _parse_float_list(args.quantiles)]
    specs.extend(("fixed", value) for value in _parse_float_list(args.fixed_thresholds))

    rows = []
    for top_k in top_ks:
        for seed in seeds:
            weighted = _select_weighted(
                candidates,
                top_k=top_k,
                max_per_user_task=args.max_per_user_task,
            )
            rows.append(
                {
                    "method": "weighted_baseline",
                    "top_k": top_k,
                    "seed": seed,
                    "utility_key": "selection_score",
                    "threshold_mode": "none",
                    "threshold_value": 0.0,
                    "threshold": 0.0,
                    **_summarize(weighted),
                }
            )
            random_rows = _select_random(
                candidates,
                top_k=top_k,
                seed=seed,
                max_per_user_task=args.max_per_user_task,
            )
            rows.append(
                {
                    "method": "random",
                    "top_k": top_k,
                    "seed": seed,
                    "utility_key": "none",
                    "threshold_mode": "none",
                    "threshold_value": 0.0,
                    "threshold": 0.0,
                    **_summarize(random_rows),
                }
            )
            for utility_key in utility_keys:
                for mode, threshold_value in specs:
                    threshold = _threshold(candidates, utility_key, mode, threshold_value)
                    selected = _select_pareto(
                        candidates,
                        top_k=top_k,
                        utility_key=utility_key,
                        threshold=threshold,
                        max_per_user_task=args.max_per_user_task,
                    )
                    rows.append(
                        {
                            "method": "pareto_utility_constraint",
                            "top_k": top_k,
                            "seed": seed,
                            "utility_key": utility_key,
                            "threshold_mode": mode,
                            "threshold_value": threshold_value,
                            "threshold": threshold,
                            **_summarize(selected),
                        }
                    )

    aggregate = []
    group_keys = ["method", "top_k", "utility_key", "threshold_mode", "threshold_value", "threshold"]
    seen = {
        tuple(row[key] for key in group_keys)
        for row in rows
    }
    for key_tuple in sorted(seen):
        vals = [
            row
            for row in rows
            if tuple(row[key] for key in group_keys) == key_tuple
        ]
        aggregate.append(
            {
                **dict(zip(group_keys, key_tuple, strict=True)),
                "seeds": len(vals),
                "observed_asr_mean": float(np.mean([row["observed_asr"] for row in vals])),
                "observed_bup_mean": float(np.mean([row["observed_bup"] for row in vals])),
                "objective_asr_plus_bup_mean": float(
                    np.mean([row["objective_asr_plus_bup"] for row in vals])
                ),
                "feasible_rate_mean": float(np.mean([row["feasible_rate"] for row in vals])),
                "pareto_utility_score_mean": float(
                    np.mean([row["mean_pareto_utility_score"] for row in vals])
                ),
            }
        )

    best = max(
        aggregate,
        key=lambda row: (
            row["objective_asr_plus_bup_mean"],
            row["observed_bup_mean"],
            row["observed_asr_mean"],
        ),
    )
    output = {
        "scope": "pareto_utility_selection",
        "candidate_json": str(args.candidate_json.resolve()),
        "candidate_count": len(candidates),
        "top_k": top_ks,
        "seeds": seeds,
        "utility_keys": utility_keys,
        "threshold_specs": specs,
        "best": best,
        "aggregate": aggregate,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"best": best, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
