"""Apply the frozen task-level gate to the v4 belief world-model pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import (
    exact_sign_test,
    paired_bootstrap,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    kind: str,
    horizon: int | None = None,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm
        and int(row["training_seed"]) == seed
        and row["kind"] == kind
        and (horizon is None or int(row["horizon"]) == horizon)
    ]
    if not selected:
        raise ValueError(f"empty prediction surface: {arm}/{seed}/{kind}/{horizon}")
    return selected


def _task_map(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_name"])].append(float(row[metric]))
    return {task: float(np.mean(values)) for task, values in sorted(grouped.items())}


def _paired_gain(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    higher_is_better: bool,
) -> dict[str, float]:
    left_ids = {
        (row["event_id"], row.get("horizon"), row["training_seed"])
        for row in left
    }
    right_ids = {
        (row["event_id"], row.get("horizon"), row["training_seed"])
        for row in right
    }
    if left_ids != right_ids:
        raise ValueError("paired prediction surfaces differ")
    left_tasks = _task_map(left, metric)
    right_tasks = _task_map(right, metric)
    if left_tasks.keys() != right_tasks.keys():
        raise ValueError("paired task surfaces differ")
    if higher_is_better:
        return {task: right_tasks[task] - left_tasks[task] for task in left_tasks}
    return {task: left_tasks[task] - right_tasks[task] for task in left_tasks}


def _effect_summary(
    by_seed: Mapping[int, Mapping[str, float]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    seeds = sorted(by_seed)
    tasks = set(by_seed[seeds[0]])
    if any(set(by_seed[seed]) != tasks for seed in seeds):
        raise ValueError("seed task surfaces differ")
    paired = {
        task: float(np.mean([by_seed[seed][task] for seed in seeds]))
        for task in sorted(tasks)
    }
    seed_means = {
        str(seed): float(np.mean(list(by_seed[seed].values()))) for seed in seeds
    }
    return {
        "mean_gain": float(np.mean(list(seed_means.values()))),
        "gain_by_seed": seed_means,
        "paired_task_gains": paired,
        "positive_task_fraction": sum(value > 0.0 for value in paired.values())
        / len(paired),
        "paired_bootstrap": paired_bootstrap(
            list(paired.values()),
            draws=int(protocol["uncertainty"]["paired_task_bootstrap_draws"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
        ),
        "paired_sign_test": exact_sign_test(list(paired.values())),
    }


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    kind: str,
    metric: str,
    seeds: Sequence[int],
    protocol: Mapping[str, Any],
    horizon: int | None = None,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    effects = {}
    for seed in seeds:
        left = _select(rows, arm=left_arm, seed=seed, kind=kind, horizon=horizon)
        right = _select(rows, arm=right_arm, seed=seed, kind=kind, horizon=horizon)
        effects[seed] = _paired_gain(
            left, right, metric=metric, higher_is_better=higher_is_better
        )
    return _effect_summary(effects, protocol)


def _markdown(summary: Mapping[str, Any]) -> str:
    effects = summary["effects"]
    lines = [
        "# Factorized belief world model v4 pilot",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| comparison | metric | task-macro gain | positive tasks |",
        "|---|---|---:|---:|",
    ]
    for name, value in effects.items():
        lines.append(
            f"| {name} | {value['metric']} | {value['summary']['mean_gain']:+.6f} | {value['summary']['positive_task_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate",
            "",
            *[
                f"- {name}: **{'PASS' if passed else 'FAIL'}**"
                for name, passed in summary["gate_checks"].items()
            ],
            "",
            "The typed one-step ablation, paired intervals, calibration diagnostics, and individual horizons are retained as counterevidence and were not used to change the gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.predictions)
    if not audit["passed"]:
        raise ValueError("dataset audit failed")
    if metrics["neural_training_runs"] != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("training budget incomplete")
    if metrics["predictions_sha256"] != file_sha256(args.predictions):
        raise ValueError("prediction hash mismatch")
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]

    h1_nll = _comparison(
        rows,
        left_arm="structured_mlp",
        right_arm="fns_bwm_multihorizon",
        kind="rollout",
        metric="action_nll",
        seeds=seeds,
        protocol=protocol,
        horizon=1,
    )
    h1_accuracy = _comparison(
        rows,
        left_arm="structured_mlp",
        right_arm="fns_bwm_multihorizon",
        kind="rollout",
        metric="action_correct",
        seeds=seeds,
        protocol=protocol,
        horizon=1,
        higher_is_better=True,
    )
    typed_h1_nll = _comparison(
        rows,
        left_arm="structured_mlp",
        right_arm="fns_bwm_onestep",
        kind="rollout",
        metric="action_nll",
        seeds=seeds,
        protocol=protocol,
        horizon=1,
    )
    multi_effects = {}
    for horizon in range(2, int(protocol["training"]["maximum_horizon"]) + 1):
        multi_effects[horizon] = _comparison(
            rows,
            left_arm="fns_bwm_onestep",
            right_arm="fns_bwm_multihorizon",
            kind="rollout",
            metric="action_nll",
            seeds=seeds,
            protocol=protocol,
            horizon=horizon,
        )
    multi_seed_tasks: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for horizon, effect in multi_effects.items():
        del horizon
        for seed, mean in effect["gain_by_seed"].items():
            multi_seed_tasks[int(seed)]["__seed_mean__"].append(float(mean))
        for task, gain in effect["paired_task_gains"].items():
            multi_seed_tasks[-1][task].append(float(gain))
    multi_seed_means = {
        str(seed): float(np.mean(values["__seed_mean__"]))
        for seed, values in multi_seed_tasks.items()
        if seed >= 0
    }
    multi_task_gains = {
        task: float(np.mean(values)) for task, values in multi_seed_tasks[-1].items()
    }
    multi_summary = {
        "mean_gain": float(np.mean(list(multi_seed_means.values()))),
        "gain_by_seed": multi_seed_means,
        "paired_task_gains": multi_task_gains,
        "positive_task_fraction": sum(value > 0.0 for value in multi_task_gains.values())
        / len(multi_task_gains),
        "paired_bootstrap": paired_bootstrap(
            list(multi_task_gains.values()),
            draws=int(protocol["uncertainty"]["paired_task_bootstrap_draws"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]) + 1,
        ),
        "paired_sign_test": exact_sign_test(list(multi_task_gains.values())),
    }
    outcome = _comparison(
        rows,
        left_arm="structured_mlp",
        right_arm="fns_bwm_multihorizon",
        kind="outcome",
        metric="outcome_bce",
        seeds=seeds,
        protocol=protocol,
    )
    execution_error = _comparison(
        rows,
        left_arm="structured_mlp",
        right_arm="fns_bwm_multihorizon",
        kind="outcome",
        metric="execution_error_bce",
        seeds=seeds,
        protocol=protocol,
    )
    gates = protocol["acceptance_gate"]
    positive_seed_count_h1 = sum(
        value >= float(gates["minimum_h1_nll_gain"])
        for value in h1_nll["gain_by_seed"].values()
    )
    positive_seed_count_multi = sum(
        value >= float(gates["minimum_h2_h5_nll_gain"])
        for value in multi_summary["gain_by_seed"].values()
    )
    legal = all(
        float(row["legal_prediction"]) == 1.0
        for row in rows
        if row["kind"] == "rollout"
    )
    checks = {
        "h1_task_macro_nll_gain": h1_nll["mean_gain"] >= float(gates["minimum_h1_nll_gain"]),
        "h1_task_macro_accuracy_gain": h1_accuracy["mean_gain"] >= float(gates["minimum_h1_accuracy_gain"]),
        "h1_seed_replication": positive_seed_count_h1 >= int(gates["minimum_positive_seeds"]),
        "h1_positive_task_fraction": h1_nll["positive_task_fraction"] >= float(gates["minimum_positive_task_fraction"]),
        "h2_h5_task_macro_nll_gain": multi_summary["mean_gain"] >= float(gates["minimum_h2_h5_nll_gain"]),
        "h2_h5_seed_replication": positive_seed_count_multi >= int(gates["minimum_positive_seeds"]),
        "h2_h5_positive_task_fraction": multi_summary["positive_task_fraction"] >= float(gates["minimum_positive_task_fraction"]),
        "outcome_bce_noninferiority": outcome["mean_gain"] >= -float(gates["maximum_outcome_bce_degradation"]),
        "execution_error_bce_noninferiority": execution_error["mean_gain"] >= -float(gates["maximum_execution_error_bce_degradation"]),
        "all_predictions_legal": legal,
        "complete_fixed_budget": metrics["neural_training_runs"] == int(protocol["fixed_budget"]["neural_training_runs"]),
    }
    passed = all(checks.values())
    effects = {
        "v4_multihorizon_vs_structured_h1_nll": {"metric": "action_nll", "summary": h1_nll},
        "v4_multihorizon_vs_structured_h1_accuracy": {"metric": "action_accuracy", "summary": h1_accuracy},
        "typed_onestep_vs_structured_h1_nll": {"metric": "action_nll", "summary": typed_h1_nll},
        "multihorizon_vs_onestep_h2_h5_nll": {"metric": "free_rollout_action_nll", "summary": multi_summary},
        "v4_multihorizon_vs_structured_outcome_bce": {"metric": "outcome_bce", "summary": outcome},
        "v4_multihorizon_vs_structured_execution_error_bce": {"metric": "execution_error_bce", "summary": execution_error},
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": (
            "GO_RETAIN_FNS_BWM_V4_FOR_NEXT_CLEAN_STAGE"
            if passed
            else "NO_GO_FNS_BWM_V4_DOES_NOT_CLEAR_FROZEN_GATE"
        ),
        "gate_passed": passed,
        "gate_checks": checks,
        "effects": effects,
        "individual_horizon_counterevidence": multi_effects,
        "run": {
            "neural_training_runs": metrics["neural_training_runs"],
            "prediction_rows": len(rows),
            "predictions_sha256": file_sha256(args.predictions),
            "run_metrics_sha256": file_sha256(args.run_metrics),
            "runtime_failures": 0,
            "new_llm_calls": 0,
            "new_tool_executions": 0,
            "real_external_endpoint_calls": 0,
            "new_attack_generation": 0,
            "dreamer_runs": 0,
        },
        "dataset": {
            "events": len(dataset["events"]),
            "trajectories": len({event["trajectory_id"] for event in dataset["events"]}),
            "adjacent_transitions": sum(event["next_target_candidate_id"] is not None for event in dataset["events"]),
            "tasks": len({event["task_name"] for event in dataset["events"]}),
            "sha256": file_sha256(args.dataset),
            "audit_sha256": file_sha256(args.audit),
        },
    }
    _write(args.output, summary)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
