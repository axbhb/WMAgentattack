"""Evaluate a validation-checkpointed configuration-level value head.

The value head predicts the continuous target ``ASR + BUP`` in [0, 2].  Its
score is used directly, so no test-time recipe or weight is selected.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = _load_script(
    "v2_downstream", ROOT / "scripts" / "70_evaluate_v2_downstream_selection.py"
)
RANK = _load_script(
    "v2_rank_stability",
    ROOT / "scripts" / "83_diagnose_v2_group_utility_rank_stability.py",
)


def _select_value(
    rows: list[dict[str, Any]], *, budget_per_task: int
) -> list[dict[str, Any]]:
    grouped = RANK._by_task(rows)
    selected = []
    for task, candidates in grouped.items():
        if len(candidates) < budget_per_task:
            raise ValueError(
                f"Task {task} has {len(candidates)} candidates, below budget "
                f"{budget_per_task}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["configuration_value_score"]),
                str(row["group_id"]),
            ),
        )
        for row in ranked[:budget_per_task]:
            selected.append(
                {
                    **row,
                    "decision_score": float(row["configuration_value_score"]),
                }
            )
    return selected


def _value_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.asarray(
        [float(row["target_asr"] + row["target_bup"]) for row in rows]
    )
    prediction = np.asarray(
        [float(row["configuration_value_score"]) for row in rows]
    )
    grouped = RANK._by_task(RANK._with_joint_targets(rows))
    per_task = {
        task: RANK._component_task_metrics(
            task_rows,
            score=lambda row: float(row["configuration_value_score"]),
            target_key="target_joint",
            observed_key="observed_joint",
        )
        for task, task_rows in grouped.items()
    }
    rank = {
        "aggregate": RANK._aggregate_task_metrics(per_task),
        "per_task": per_task,
    }
    return {
        "normalized_brier": float(np.mean(((prediction - target) / 2.0) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rank": rank,
    }


def _load_rows(
    *,
    data_root: Path,
    model_root: Path,
    seeds: tuple[int, ...],
    decision_step: str,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    split_steps = {
        split: BASE._steps(data_root / f"{split}_steps.jsonl")
        for split in ("val", "test")
    }
    rows: dict[str, dict[int, list[dict[str, Any]]]] = {"val": {}, "test": {}}
    for seed in seeds:
        model = BASE.FullSheepRLDreamerV3.load(
            model_root / f"seed{seed}" / "model"
        )
        for split, steps in split_steps.items():
            prediction = model.predict(steps)
            if "configuration_value_score" not in prediction:
                raise ValueError(f"Seed {seed} model has no configuration value head")
            rows[split][seed] = BASE._configuration_rows(
                steps,
                prediction,
                BASE.MonotonicAffineRiskCalibrator(),
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
    return rows


def _baseline_reference(
    payload: dict[str, Any], *, budget: int, seeds: tuple[int, ...]
) -> dict[str, Any]:
    calibrated = payload["test"][str(budget)]["calibrated"]
    if len(seeds) == 1:
        return calibrated["per_seed"][str(seeds[0])]
    return calibrated["ensemble"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--baseline-downstream", type=Path, required=True)
    parser.add_argument("--decision-step", choices=["first", "final"], default="first")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    rows = _load_rows(
        data_root=args.data_root,
        model_root=args.model_root,
        seeds=seeds,
        decision_step=args.decision_step,
    )
    ensemble = {
        split: BASE._ensemble(rows[split]) for split in ("val", "test")
    }
    baseline = json.loads(args.baseline_downstream.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "scope": "direct configuration-level ASR+BUP value selection",
        "protocol": {
            "checkpoint_seeds": list(seeds),
            "decision_step": args.decision_step,
            "score": "configuration_value_score in [0,2]",
            "selection": "direct descending value; no recipe or test retuning",
            "test_labels": "evaluation only",
        },
        "provenance": {
            "data_root": str(args.data_root.resolve()),
            "model_root": str(args.model_root.resolve()),
            "baseline_downstream": str(args.baseline_downstream.resolve()),
        },
        "quality": {
            split: {
                "per_seed": {
                    str(seed): _value_quality(seed_rows)
                    for seed, seed_rows in sorted(rows[split].items())
                },
                "ensemble": _value_quality(ensemble[split]),
                "cross_seed_stability": RANK._cross_seed_stability(
                    rows[split], recipe="risk_plus_utility"
                )["configuration_value"],
            }
            for split in ("val", "test")
        },
        "selection": {"val": {}, "test": {}},
    }
    for budget in args.budgets:
        key = str(budget)
        for split in ("val", "test"):
            per_seed_rows = {
                seed: _select_value(seed_rows, budget_per_task=budget)
                for seed, seed_rows in rows[split].items()
            }
            per_seed_metrics = {
                str(seed): BASE._metrics(selected)
                for seed, selected in per_seed_rows.items()
            }
            ensemble_selected = _select_value(
                ensemble[split], budget_per_task=budget
            )
            result["selection"][split][key] = {
                "per_seed": per_seed_metrics,
                "ensemble": BASE._metrics(ensemble_selected),
            }
            if split == "test":
                variant_metrics = result["selection"][split][key]["ensemble"]
                baseline_metrics = _baseline_reference(
                    baseline, budget=budget, seeds=seeds
                )
                result["selection"][split][key].update(
                    {
                        "ensemble_task_bootstrap": BASE._task_bootstrap(
                            ensemble_selected,
                            draws=args.bootstrap_draws,
                            seed=args.bootstrap_seed + budget,
                        ),
                        "random_expected": BASE._pool_expected(
                            ensemble[split], budget
                        ),
                        "oracle_diagnostic_only": BASE._oracle(
                            ensemble[split], budget
                        ),
                        "baseline_reference": baseline_metrics,
                        "variant_minus_baseline": {
                            metric: float(
                                variant_metrics[metric] - baseline_metrics[metric]
                            )
                            for metric in ("ASR", "BUP", "ASR_plus_BUP")
                        },
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
