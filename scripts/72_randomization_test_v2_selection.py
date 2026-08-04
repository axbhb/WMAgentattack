"""Task-balanced randomization test for frozen AgentDojo-v2 selections."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def _configuration_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    steps = [StepRecord.model_validate(row) for row in read_jsonl(path)]
    first: dict[str, StepRecord] = {}
    for step in steps:
        previous = first.get(step.trajectory_id)
        if previous is None or step.step_id < previous.step_id:
            first[step.trajectory_id] = step
    grouped: dict[str, list[StepRecord]] = defaultdict(list)
    for step in first.values():
        if (
            step.multiseed_group_id is not None
            and step.attack_probability_target is not None
        ):
            grouped[str(step.multiseed_group_id)].append(step)
    output = {}
    for group_id, records in grouped.items():
        task_keys = {f"{row.domain}|{row.task_id}" for row in records}
        if len(task_keys) != 1:
            raise ValueError(f"Configuration spans task keys: {group_id}")
        output[group_id] = {
            "task_key": next(iter(task_keys)),
            "ASR": float(np.mean([row.attack_success for row in records])),
            "BUP": float(np.mean([row.task_success for row in records])),
        }
        output[group_id]["ASR_plus_BUP"] = (
            output[group_id]["ASR"] + output[group_id]["BUP"]
        )
    return output


def _random_samples(
    outcomes: dict[str, dict[str, Any]],
    *,
    budget: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes.values():
        grouped[row["task_key"]].append(row)
    rng = np.random.default_rng(seed)
    metrics = ("ASR", "BUP", "ASR_plus_BUP")
    samples = {metric: np.zeros(draws, dtype=np.float64) for metric in metrics}
    for task_key in sorted(grouped):
        rows = grouped[task_key]
        if len(rows) < budget:
            raise ValueError(f"Budget exceeds pool for {task_key}")
        random_order = np.argpartition(
            rng.random((draws, len(rows))), budget - 1, axis=1
        )[:, :budget]
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            samples[metric] += values[random_order].mean(axis=1)
    for metric in metrics:
        samples[metric] /= len(grouped)
    return samples


def _selection_metrics(
    selected_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, float]:
    rows = [outcomes[group_id] for group_id in selected_ids]
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("ASR", "BUP", "ASR_plus_BUP")
    }


def _compare(observed: float, random: np.ndarray) -> dict[str, float]:
    return {
        "observed": observed,
        "random_mean": float(np.mean(random)),
        "random_std": float(np.std(random)),
        "random_ci95_low": float(np.quantile(random, 0.025)),
        "random_ci95_high": float(np.quantile(random, 0.975)),
        "random_percentile": float(np.mean(random <= observed)),
        "one_sided_p_random_at_least_observed": float(
            (1 + np.sum(random >= observed)) / (len(random) + 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--test-steps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error("--draws must be at least 1000")

    selection = json.loads(
        args.selection_result.read_text(encoding="utf-8")
    )
    outcomes = _configuration_outcomes(args.test_steps)
    tests = {}
    for budget_key, budget_result in selection["test"].items():
        budget = int(budget_key)
        random = _random_samples(
            outcomes,
            budget=budget,
            draws=args.draws,
            seed=args.seed + budget,
        )
        tests[budget_key] = {}
        for variant in ("raw", "calibrated"):
            selected_ids = budget_result[variant]["ensemble"][
                "selected_group_ids"
            ]
            observed = _selection_metrics(selected_ids, outcomes)
            tests[budget_key][variant] = {
                "selected_group_ids": selected_ids,
                "metrics": {
                    metric: _compare(observed[metric], random[metric])
                    for metric in observed
                },
            }
    result = {
        "scope": "task-balanced randomization test for frozen selection",
        "selection_result": str(args.selection_result.resolve()),
        "test_steps": str(args.test_steps.resolve()),
        "draws": args.draws,
        "random_seed": args.seed,
        "null": (
            "uniformly select the same number of configurations without "
            "replacement inside every held-out user task"
        ),
        "configuration_count": len(outcomes),
        "task_count": len({row["task_key"] for row in outcomes.values()}),
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
