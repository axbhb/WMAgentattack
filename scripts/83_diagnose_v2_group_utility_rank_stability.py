"""Diagnose candidate-level rank stability for grouped utility models.

This script reuses validation-frozen calibration and scoring recipes from the
strict downstream evaluation.  Test labels are used only for diagnostics; no
recipe, calibration, or checkpoint is selected from test outcomes.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "70_evaluate_v2_downstream_selection.py"
BASE_SPEC = importlib.util.spec_from_file_location("v2_downstream", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)


ScoreFn = Callable[[dict[str, Any]], float]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rankdata(values: list[float]) -> np.ndarray:
    """Return average ranks with deterministic handling of exact ties."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _score_specs(recipe: str) -> dict[str, tuple[ScoreFn, str, str]]:
    return {
        "risk": (
            lambda row: float(row["calibrated_risk_score"]),
            "target_asr",
            "observed_asr",
        ),
        "utility": (
            lambda row: float(row["utility_score"]),
            "target_bup",
            "observed_bup",
        ),
        "preservation": (
            lambda row: float(row["preservation_score"]),
            "target_bup",
            "observed_bup",
        ),
        "critic_value": (
            lambda row: float(row["critic_value_score"]),
            "target_joint",
            "observed_joint",
        ),
        "configuration_value": (
            lambda row: float(row["configuration_value_score"]),
            "target_joint",
            "observed_joint",
        ),
        "composite": (
            lambda row: BASE._decision_score(
                row, "calibrated_risk_score", recipe
            ),
            "target_joint",
            "observed_joint",
        ),
    }


def _with_joint_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        row["target_joint"] = float(row["target_asr"] + row["target_bup"])
        row["observed_joint"] = float(
            row["observed_asr"] + row["observed_bup"]
        )
        output.append(row)
    return output


def _by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_key"])].append(row)
    return {key: grouped[key] for key in sorted(grouped)}


def _component_task_metrics(
    rows: list[dict[str, Any]],
    *,
    score: ScoreFn,
    target_key: str,
    observed_key: str,
) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (-score(row), str(row["group_id"])),
    )
    selected = ranked[0]
    top_score = score(selected)
    runner_up_score = score(ranked[1]) if len(ranked) > 1 else top_score
    target_values = [float(row[target_key]) for row in rows]
    observed_values = [float(row[observed_key]) for row in rows]
    return {
        "candidate_count": len(rows),
        "selected_group_id": str(selected["group_id"]),
        "spearman_target": _spearman(
            [score(row) for row in rows], target_values
        ),
        "top1_target_regret": max(target_values)
        - float(selected[target_key]),
        "top1_observed_regret": max(observed_values)
        - float(selected[observed_key]),
        "top1_margin": float(top_score - runner_up_score),
        "selected_target": float(selected[target_key]),
        "selected_observed": float(selected[observed_key]),
    }


def _aggregate_task_metrics(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    correlations = [
        float(row["spearman_target"])
        for row in per_task.values()
        if row["spearman_target"] is not None
        and math.isfinite(float(row["spearman_target"]))
    ]
    return {
        "task_count": len(per_task),
        "spearman_defined_task_count": len(correlations),
        "mean_task_spearman_target": _mean(correlations),
        "mean_top1_target_regret": _mean(
            [float(row["top1_target_regret"]) for row in per_task.values()]
        ),
        "mean_top1_observed_regret": _mean(
            [float(row["top1_observed_regret"]) for row in per_task.values()]
        ),
        "target_oracle_top1_rate": _mean(
            [
                float(abs(float(row["top1_target_regret"])) <= 1e-12)
                for row in per_task.values()
            ]
        ),
        "observed_oracle_top1_rate": _mean(
            [
                float(abs(float(row["top1_observed_regret"])) <= 1e-12)
                for row in per_task.values()
            ]
        ),
        "mean_top1_margin": _mean(
            [float(row["top1_margin"]) for row in per_task.values()]
        ),
        "mean_selected_target": _mean(
            [float(row["selected_target"]) for row in per_task.values()]
        ),
        "mean_selected_observed": _mean(
            [float(row["selected_observed"]) for row in per_task.values()]
        ),
    }


def _rank_summary(rows: list[dict[str, Any]], *, recipe: str) -> dict[str, Any]:
    grouped = _by_task(_with_joint_targets(rows))
    output: dict[str, Any] = {}
    for component, (score, target_key, observed_key) in _score_specs(
        recipe
    ).items():
        per_task = {
            task: _component_task_metrics(
                task_rows,
                score=score,
                target_key=target_key,
                observed_key=observed_key,
            )
            for task, task_rows in grouped.items()
        }
        output[component] = {
            "aggregate": _aggregate_task_metrics(per_task),
            "per_task": per_task,
        }
    return output


def _cross_seed_stability(
    rows_by_seed: dict[int, list[dict[str, Any]]], *, recipe: str
) -> dict[str, Any]:
    seeds = sorted(rows_by_seed)
    maps = {
        seed: {str(row["group_id"]): row for row in _with_joint_targets(rows)}
        for seed, rows in rows_by_seed.items()
    }
    group_ids = set(maps[seeds[0]])
    if any(set(maps[seed]) != group_ids for seed in seeds[1:]):
        raise ValueError("Checkpoint seeds produced different configuration ids")
    task_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, row in maps[seeds[0]].items():
        task_groups[str(row["task_key"])].append(group_id)

    result: dict[str, Any] = {}
    for component, (score, _, _) in _score_specs(recipe).items():
        per_task = {}
        for task, ids in sorted(task_groups.items()):
            ids = sorted(ids)
            pair_correlations = []
            for left_index, left_seed in enumerate(seeds):
                for right_seed in seeds[left_index + 1 :]:
                    correlation = _spearman(
                        [score(maps[left_seed][group_id]) for group_id in ids],
                        [score(maps[right_seed][group_id]) for group_id in ids],
                    )
                    if correlation is not None and math.isfinite(correlation):
                        pair_correlations.append(correlation)

            winners = []
            for seed in seeds:
                winners.append(
                    min(
                        ids,
                        key=lambda group_id: (
                            -score(maps[seed][group_id]),
                            group_id,
                        ),
                    )
                )
            winner_counts = {winner: winners.count(winner) for winner in set(winners)}
            ensemble_scores = {
                group_id: statistics.fmean(
                    score(maps[seed][group_id]) for seed in seeds
                )
                for group_id in ids
            }
            ranked_ids = sorted(
                ids, key=lambda group_id: (-ensemble_scores[group_id], group_id)
            )
            winner = ranked_ids[0]
            runner_up = ranked_ids[1] if len(ranked_ids) > 1 else winner
            gap_by_seed = [
                score(maps[seed][winner]) - score(maps[seed][runner_up])
                for seed in seeds
            ]
            gap_mean = statistics.fmean(gap_by_seed)
            gap_std = statistics.pstdev(gap_by_seed) if len(seeds) > 1 else 0.0
            ratio = (
                abs(gap_mean) / gap_std
                if gap_std > 1e-12
                else (float("inf") if abs(gap_mean) > 1e-12 else 0.0)
            )
            per_task[task] = {
                "candidate_count": len(ids),
                "mean_pairwise_seed_spearman": _mean(pair_correlations),
                "defined_seed_pair_count": len(pair_correlations),
                "unique_seed_top1_count": len(set(winners)),
                "top1_consensus_fraction": max(winner_counts.values()) / len(seeds),
                "seed_top1_group_ids": {
                    str(seed): winner_id
                    for seed, winner_id in zip(seeds, winners)
                },
                "ensemble_top1_group_id": winner,
                "ensemble_top1_margin": float(gap_mean),
                "top1_gap_across_seed_std": float(gap_std),
                "margin_to_seed_uncertainty": float(ratio),
            }
        correlations = [
            float(row["mean_pairwise_seed_spearman"])
            for row in per_task.values()
            if row["mean_pairwise_seed_spearman"] is not None
        ]
        finite_ratios = [
            float(row["margin_to_seed_uncertainty"])
            for row in per_task.values()
            if math.isfinite(float(row["margin_to_seed_uncertainty"]))
        ]
        result[component] = {
            "aggregate": {
                "task_count": len(per_task),
                "mean_pairwise_seed_spearman": _mean(correlations),
                "complete_top1_agreement_rate": _mean(
                    [
                        float(row["unique_seed_top1_count"] == 1)
                        for row in per_task.values()
                    ]
                ),
                "mean_top1_consensus_fraction": _mean(
                    [
                        float(row["top1_consensus_fraction"])
                        for row in per_task.values()
                    ]
                ),
                "mean_ensemble_top1_margin": _mean(
                    [
                        float(row["ensemble_top1_margin"])
                        for row in per_task.values()
                    ]
                ),
                "mean_margin_to_seed_uncertainty": _mean(finite_ratios),
                "fraction_tasks_margin_below_seed_uncertainty": _mean(
                    [
                        float(row["margin_to_seed_uncertainty"] < 1.0)
                        for row in per_task.values()
                    ]
                ),
            },
            "per_task": per_task,
        }
    return result


def _view_summary(
    rows_by_seed: dict[int, list[dict[str, Any]]], *, recipe: str
) -> dict[str, Any]:
    ensemble = BASE._ensemble(rows_by_seed)
    return {
        "recipe": recipe,
        "per_seed": {
            str(seed): {
                component: values["aggregate"]
                for component, values in _rank_summary(
                    rows, recipe=recipe
                ).items()
            }
            for seed, rows in sorted(rows_by_seed.items())
        },
        "ensemble": _rank_summary(ensemble, recipe=recipe),
        "cross_seed_stability": _cross_seed_stability(
            rows_by_seed, recipe=recipe
        ),
    }


def _load_frozen_calibrator(payload: dict[str, Any], seed: int) -> Any:
    if payload["selected_calibration"] == "identity":
        return BASE.MonotonicAffineRiskCalibrator()
    return BASE.MonotonicAffineRiskCalibrator.from_dict(
        payload["calibrators"][str(seed)]
    )


def _load_method_rows(
    *,
    model_root: Path,
    downstream: dict[str, Any],
    split_steps: dict[str, list[Any]],
    seeds: tuple[int, ...],
    decision_step: str,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    output: dict[str, dict[int, list[dict[str, Any]]]] = {
        "val": {},
        "test": {},
    }
    for seed in seeds:
        calibrator = _load_frozen_calibrator(downstream, seed)
        model = BASE.FullSheepRLDreamerV3.load(
            model_root / f"seed{seed}" / "model"
        )
        for split, steps in split_steps.items():
            output[split][seed] = BASE._configuration_rows(
                steps,
                model.predict(steps),
                calibrator,
                decision_step=decision_step,
            )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return output


def _selection_comparison(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    *,
    baseline_recipe: str,
    variant_recipe: str,
) -> dict[str, Any]:
    baseline_tasks = _by_task(_with_joint_targets(baseline_rows))
    variant_tasks = _by_task(_with_joint_targets(variant_rows))
    if set(baseline_tasks) != set(variant_tasks):
        raise ValueError("Methods cover different task sets")
    per_task = {}
    for task in sorted(baseline_tasks):
        baseline_candidates = baseline_tasks[task]
        variant_candidates = variant_tasks[task]
        baseline_ids = {str(row["group_id"]) for row in baseline_candidates}
        variant_ids = {str(row["group_id"]) for row in variant_candidates}
        if baseline_ids != variant_ids:
            raise ValueError(f"Methods cover different candidates for {task}")
        baseline_selected = min(
            baseline_candidates,
            key=lambda row: (
                -BASE._decision_score(
                    row, "calibrated_risk_score", baseline_recipe
                ),
                str(row["group_id"]),
            ),
        )
        variant_selected = min(
            variant_candidates,
            key=lambda row: (
                -BASE._decision_score(
                    row, "calibrated_risk_score", variant_recipe
                ),
                str(row["group_id"]),
            ),
        )
        changed = str(baseline_selected["group_id"]) != str(
            variant_selected["group_id"]
        )
        observed_delta = float(
            variant_selected["observed_joint"]
            - baseline_selected["observed_joint"]
        )
        target_delta = float(
            variant_selected["target_joint"]
            - baseline_selected["target_joint"]
        )
        per_task[task] = {
            "baseline_group_id": str(baseline_selected["group_id"]),
            "variant_group_id": str(variant_selected["group_id"]),
            "selection_changed": changed,
            "changed_but_observed_outcome_tied": bool(
                changed and abs(observed_delta) <= 1e-12
            ),
            "selected_observed_joint_delta": observed_delta,
            "selected_target_joint_delta": target_delta,
        }
    changed_rows = [row for row in per_task.values() if row["selection_changed"]]
    return {
        "baseline_recipe": baseline_recipe,
        "variant_recipe": variant_recipe,
        "aggregate": {
            "task_count": len(per_task),
            "changed_selection_count": len(changed_rows),
            "selection_overlap_rate": 1.0 - len(changed_rows) / len(per_task),
            "changed_but_observed_outcome_tied_count": sum(
                row["changed_but_observed_outcome_tied"] for row in changed_rows
            ),
            "mean_selected_observed_joint_delta": _mean(
                [
                    float(row["selected_observed_joint_delta"])
                    for row in per_task.values()
                ]
            ),
            "mean_selected_target_joint_delta": _mean(
                [
                    float(row["selected_target_joint_delta"])
                    for row in per_task.values()
                ]
            ),
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--variant-model-root", type=Path, required=True)
    parser.add_argument("--baseline-downstream", type=Path, required=True)
    parser.add_argument("--variant-downstream", type=Path, required=True)
    parser.add_argument("--decision-step", choices=["first", "final"], default="first")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    split_steps = {
        split: BASE._steps(args.data_root / f"{split}_steps.jsonl")
        for split in ("val", "test")
    }
    downstream = {
        "baseline": json.loads(
            args.baseline_downstream.read_text(encoding="utf-8")
        ),
        "variant": json.loads(
            args.variant_downstream.read_text(encoding="utf-8")
        ),
    }
    roots = {
        "baseline": args.baseline_model_root,
        "variant": args.variant_model_root,
    }
    rows = {
        method: _load_method_rows(
            model_root=roots[method],
            downstream=downstream[method],
            split_steps=split_steps,
            seeds=seeds,
            decision_step=args.decision_step,
        )
        for method in ("baseline", "variant")
    }

    recipes = {
        method: downstream[method]["test"]["1"]["calibrated"][
            "frozen_validation_recipe"
        ]
        for method in ("baseline", "variant")
    }
    methods = {}
    for method in ("baseline", "variant"):
        methods[method] = {
            "selected_calibration": downstream[method]["selected_calibration"],
            "frozen_top1_recipe": recipes[method],
            "splits": {
                split: {
                    "frozen_recipe": _view_summary(
                        rows[method][split], recipe=recipes[method]
                    ),
                    "fixed_risk_plus_utility": _view_summary(
                        rows[method][split], recipe="risk_plus_utility"
                    ),
                }
                for split in ("val", "test")
            },
        }

    ensemble_rows = {
        method: {
            split: BASE._ensemble(rows[method][split])
            for split in ("val", "test")
        }
        for method in ("baseline", "variant")
    }
    result = {
        "scope": "candidate-level grouped utility rank stability diagnostic",
        "protocol": {
            "checkpoint_seeds": list(seeds),
            "decision_step": args.decision_step,
            "calibration_and_recipe": "frozen from prior validation selection",
            "test_labels": "diagnostic only; no test retuning",
            "fixed_recipe_control": "risk_plus_utility",
        },
        "provenance": {
            "data_root": str(args.data_root.resolve()),
            "baseline_model_root": str(args.baseline_model_root.resolve()),
            "variant_model_root": str(args.variant_model_root.resolve()),
            "baseline_downstream": str(args.baseline_downstream.resolve()),
            "variant_downstream": str(args.variant_downstream.resolve()),
        },
        "methods": methods,
        "ensemble_top1_comparison": {
            split: _selection_comparison(
                ensemble_rows["baseline"][split],
                ensemble_rows["variant"][split],
                baseline_recipe=recipes["baseline"],
                variant_recipe=recipes["variant"],
            )
            for split in ("val", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.candidate_output is not None:
        candidate_payload = {
            method: {
                split: {
                    str(seed): _with_joint_targets(seed_rows)
                    for seed, seed_rows in sorted(split_rows.items())
                }
                for split, split_rows in method_rows.items()
            }
            for method, method_rows in rows.items()
        }
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(candidate_payload, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
