"""Offline score-weight sweep for already-scored world-model candidates.

This script is deliberately separated from model inference.  First run
``11_select_world_model_agentdojo_pairs.py --include-candidates`` once to cache
the expensive Dreamer/RSSM scores.  Then sweep scalar selection weights here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


DEFAULT_PRESETS = {
    "default": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.5,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "utility_1x": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 1.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "utility_1p5x": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 1.5,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "utility_2x": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 2.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "balanced_utility": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 0.8,
        "mean_risk": 0.2,
        "utility": 1.5,
        "target": 0.2,
        "target_reached": 0.1,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "target_light_utility": {
        "objective": "weighted",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.2,
        "utility": 1.2,
        "target": 0.1,
        "target_reached": 0.0,
        "utility_threshold": 0.0,
        "utility_shortfall_penalty": 2.0,
    },
    "constrained_u08_p2": {
        "objective": "utility_constrained",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.08,
        "utility_shortfall_penalty": 2.0,
    },
    "constrained_u10_p2": {
        "objective": "utility_constrained",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.10,
        "utility_shortfall_penalty": 2.0,
    },
    "constrained_u12_p4": {
        "objective": "utility_constrained",
        "utility_score_key": "utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.12,
        "utility_shortfall_penalty": 4.0,
    },
    "terminal_constrained_u10_p2": {
        "objective": "utility_constrained",
        "utility_score_key": "final_utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.10,
        "utility_shortfall_penalty": 2.0,
    },
    "min_constrained_u10_p2": {
        "objective": "utility_constrained",
        "utility_score_key": "min_utility_score",
        "risk": 1.0,
        "mean_risk": 0.3,
        "utility": 0.0,
        "target": 0.3,
        "target_reached": 0.2,
        "utility_threshold": 0.10,
        "utility_shortfall_penalty": 2.0,
    },
}


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _pair_key(row: dict) -> tuple[str, str, str]:
    return (row["suite"], row["user_task_id"], row["injection_task_id"])


def _score(row: dict, weights: dict) -> float:
    risk = float(row.get("risk_score", 0.0))
    mean_risk = float(row.get("rollout_mean_risk_score", risk))
    utility_score_key = str(weights.get("utility_score_key", "utility_score"))
    utility = float(row.get(utility_score_key, row.get("utility_score", 0.0)))
    target = float(row.get("target_skill_probability", 0.0))
    reached = float(row.get("rollout_target_reached", 0.0))
    risk_target_score = (
        weights["risk"] * risk
        + weights["mean_risk"] * mean_risk
        + weights["target"] * target
        + weights["target_reached"] * reached
    )
    if weights.get("objective", "weighted") == "utility_constrained":
        return risk_target_score - weights["utility_shortfall_penalty"] * max(
            0.0, weights["utility_threshold"] - utility
        )
    return risk_target_score + weights["utility"] * utility


def _dedupe_by_pair(rows: list[dict], max_per_user_task: int) -> list[dict]:
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


def _without_pairs(rows: list[dict], excluded: set[tuple[str, str, str]]) -> list[dict]:
    return [row for row in rows if _pair_key(row) not in excluded]


def _summarize(rows: list[dict]) -> dict:
    asr = np.array([bool(row["observed_security"]) for row in rows], dtype=float)
    bup = np.array([bool(row["observed_utility"]) for row in rows], dtype=float)
    return {
        "count": len(rows),
        "observed_asr": float(asr.mean()) if len(rows) else 0.0,
        "observed_bup": float(bup.mean()) if len(rows) else 0.0,
        "objective_asr_plus_bup": float(asr.mean() + bup.mean()) if len(rows) else 0.0,
        "mean_selection_score": float(np.mean([row["selection_score"] for row in rows])) if rows else 0.0,
        "mean_risk_score": float(np.mean([row["risk_score"] for row in rows])) if rows else 0.0,
        "mean_utility_score": float(np.mean([row["utility_score"] for row in rows])) if rows else 0.0,
        "mean_selection_utility_score": (
            float(np.mean([row.get("selection_utility_score", row["utility_score"]) for row in rows]))
            if rows
            else 0.0
        ),
        "mean_target_skill_probability": (
            float(np.mean([row["target_skill_probability"] for row in rows])) if rows else 0.0
        ),
    }


def _select(candidates: list[dict], top_k: int, seed: int, max_per_user_task: int) -> dict[str, list[dict]]:
    sorted_high = sorted(candidates, key=lambda row: row["selection_score"], reverse=True)
    sorted_risk = sorted(candidates, key=lambda row: row["risk_score"], reverse=True)
    sorted_utility = sorted(candidates, key=lambda row: row["utility_score"], reverse=True)
    sorted_target = sorted(candidates, key=lambda row: row["target_skill_probability"], reverse=True)
    sorted_low = sorted(candidates, key=lambda row: row["selection_score"])
    rng = random.Random(seed)
    random_rows = candidates[:]
    rng.shuffle(random_rows)

    world_model_top = _dedupe_by_pair(sorted_high, max_per_user_task)[:top_k]
    excluded = {_pair_key(row) for row in world_model_top}
    return {
        "world_model_top": world_model_top,
        "risk_only_top": _dedupe_by_pair(_without_pairs(sorted_risk, excluded), max_per_user_task)[:top_k],
        "utility_only_top": _dedupe_by_pair(_without_pairs(sorted_utility, excluded), max_per_user_task)[:top_k],
        "target_skill_top": _dedupe_by_pair(_without_pairs(sorted_target, excluded), max_per_user_task)[:top_k],
        "low_score": _dedupe_by_pair(_without_pairs(sorted_low, excluded), max_per_user_task)[:top_k],
        "random": _dedupe_by_pair(_without_pairs(random_rows, excluded), max_per_user_task)[:top_k],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", default="16,24,32")
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--max-per-user-task", type=int, default=2)
    args = parser.parse_args()

    payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    top_ks = _parse_int_list(args.top_k)
    seeds = _parse_int_list(args.seeds)

    rows = []
    for preset_name, weights in DEFAULT_PRESETS.items():
        utility_score_key = str(weights.get("utility_score_key", "utility_score"))
        rescored = [
            {
                **row,
                "selection_score": _score(row, weights),
                "selection_utility_score": float(row.get(utility_score_key, row.get("utility_score", 0.0))),
            }
            for row in candidates
        ]
        for top_k in top_ks:
            for seed in seeds:
                selections = _select(rescored, top_k, seed, args.max_per_user_task)
                for method, selected in selections.items():
                    rows.append(
                        {
                            "preset": preset_name,
                            "top_k": top_k,
                            "seed": seed,
                            "method": method,
                            **weights,
                            **_summarize(selected),
                            "candidate_count": len(candidates),
                            "max_per_user_task": args.max_per_user_task,
                        }
                    )

    aggregate = []
    for preset_name in DEFAULT_PRESETS:
        for top_k in top_ks:
            for method in ["world_model_top", "random", "low_score", "risk_only_top", "utility_only_top", "target_skill_top"]:
                vals = [
                    row
                    for row in rows
                    if row["preset"] == preset_name and row["top_k"] == top_k and row["method"] == method
                ]
                if not vals:
                    continue
                aggregate.append(
                    {
                        "preset": preset_name,
                        "top_k": top_k,
                        "method": method,
                        "seeds": len(vals),
                        "observed_asr_mean": float(np.mean([row["observed_asr"] for row in vals])),
                        "observed_bup_mean": float(np.mean([row["observed_bup"] for row in vals])),
                        "selection_utility_score_mean": float(
                            np.mean([row["mean_selection_utility_score"] for row in vals])
                        ),
                        "objective_asr_plus_bup_mean": float(
                            np.mean([row["objective_asr_plus_bup"] for row in vals])
                        ),
                        **DEFAULT_PRESETS[preset_name],
                    }
                )

    best = max(
        [row for row in aggregate if row["method"] == "world_model_top"],
        key=lambda row: (row["objective_asr_plus_bup_mean"], row["observed_asr_mean"], row["observed_bup_mean"]),
    )
    output = {
        "scope": "selection_weight_sweep",
        "candidate_json": str(args.candidate_json.resolve()),
        "top_k": top_ks,
        "seeds": seeds,
        "presets": DEFAULT_PRESETS,
        "best_world_model_top": best,
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
    print(json.dumps({"best_world_model_top": best, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
