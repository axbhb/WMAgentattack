"""Apply the frozen v13 M2 predicted-event-graph gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test, paired_bootstrap

KEYS = ("fold", "training_seed", "horizon", "event_id")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row):
    return tuple(row[name] for name in KEYS)


def _task_means(rows, metric):
    values = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            values[row["task_name"]].append(float(row[metric]))
    return {task: float(np.mean(entries)) for task, entries in values.items()}


def _effect(left, right, metric, *, higher=False, seed=81350):
    per_seed = {}
    for training_seed in (7, 17, 29):
        a = _task_means([row for row in left if int(row["training_seed"]) == training_seed], metric)
        b = _task_means([row for row in right if int(row["training_seed"]) == training_seed], metric)
        if set(a) != set(b):
            raise ValueError(f"task mismatch {metric} {training_seed}")
        per_seed[training_seed] = {
            task: (b[task] - a[task] if higher else a[task] - b[task]) for task in a
        }
    tasks = set(per_seed[7])
    paired = {task: float(np.mean([per_seed[value][task] for value in per_seed])) for task in tasks}
    seeds = {str(value): float(np.mean(list(entries.values()))) for value, entries in per_seed.items()}
    return {
        "mean": float(np.mean(list(seeds.values()))),
        "seeds": seeds,
        "tasks": paired,
        "positive_task_fraction": float(np.mean([value > 0 for value in paired.values()])),
        "paired_bootstrap": paired_bootstrap(list(paired.values()), draws=10000, seed=seed),
        "exact_sign_test": exact_sign_test(list(paired.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--composite-predictions", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    stage = protocol["stage_m2"]
    current = _read(args.predictions)
    metrics = json.loads(args.run_metrics.read_text())
    external_path = Path(protocol["frozen_sources"]["v6_predictions"])
    if file_sha256(external_path) != protocol["frozen_sources"]["v6_predictions_sha256"]:
        raise ValueError("v6 hash mismatch")
    external = [row for row in _read(external_path) if row["arm"] == "structured_residual_v6"]
    control = [row for row in current if row["arm"] == "unsupervised_graph_capacity_control_v13"]
    candidate = [row for row in current if row["arm"] == "predicted_event_graph_v13"]
    if not len(external) == len(control) == len(candidate):
        raise ValueError("paired row mismatch")
    v6 = {_key(row): row for row in external}
    if set(v6) != {_key(row) for row in control} or set(v6) != {_key(row) for row in candidate}:
        raise ValueError("paired key mismatch")

    composite = []
    for row in candidate:
        source = v6[_key(row)]
        output = dict(row)
        output["arm"] = "predicted_event_action_v6_outcome_v13"
        output["joint_trainable"] = source["joint_trainable"]
        output["joint_ce"] = source["joint_ce"]
        composite.append(output)
    args.composite_predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in composite))

    effects = {
        "h1_nll_vs_v6": _effect(
            [row for row in external if row["horizon"] == 1],
            [row for row in candidate if row["horizon"] == 1],
            "action_nll",
            seed=81351,
        ),
        "h1_accuracy_vs_v6": _effect(
            [row for row in external if row["horizon"] == 1],
            [row for row in candidate if row["horizon"] == 1],
            "action_correct",
            higher=True,
            seed=81352,
        ),
        "h2_h5_nll_vs_v6": _effect(
            [row for row in external if row["horizon"] >= 2],
            [row for row in candidate if row["horizon"] >= 2],
            "action_nll",
            seed=81353,
        ),
        "h2_h5_nll_vs_capacity": _effect(
            [row for row in control if row["horizon"] >= 2],
            [row for row in candidate if row["horizon"] >= 2],
            "action_nll",
            seed=81354,
        ),
        "graph_bce_vs_train_prior": _effect(
            [row for row in candidate if row["horizon"] == 1],
            [row for row in candidate if row["horizon"] == 1],
            "graph_bce",
            seed=81355,
        ),
        "future_joint_ce_vs_v6": _effect(
            [row for row in external if row["horizon"] >= 2 and row["joint_trainable"]],
            [row for row in composite if row["horizon"] >= 2 and row["joint_trainable"]],
            "joint_ce",
            seed=81356,
        ),
    }
    graph_rows = [row for row in candidate if row["horizon"] == 1]
    graph_by_task = defaultdict(list)
    graph_by_seed = defaultdict(list)
    for row in graph_rows:
        gain = float(row["graph_prior_bce"] - row["graph_bce"])
        graph_by_task[row["task_name"]].append(gain)
        graph_by_seed[str(row["training_seed"])].append(gain)
    graph_effect = {
        "mean": float(np.mean([value for values in graph_by_task.values() for value in values])),
        "tasks": {task: float(np.mean(values)) for task, values in graph_by_task.items()},
        "seeds": {seed: float(np.mean(values)) for seed, values in graph_by_seed.items()},
    }
    effects["graph_bce_vs_train_prior"] = graph_effect

    gate = stage["gate"]
    checks = {
        "complete_budget": metrics["training_units"] == 30 and metrics["teacher_fits"] == 15,
        "runtime_clean": metrics["runtime_failures"] == 0,
        "parameter_match": bool(metrics["parameter_match"]),
        "paired_rows_complete": len(candidate) == len(external),
        "graph_predictive_gain": graph_effect["mean"] >= gate["minimum_graph_bce_gain_vs_train_prior"],
        "h1_nll_noninferiority": effects["h1_nll_vs_v6"]["mean"] >= -gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority": effects["h1_accuracy_vs_v6"]["mean"] >= -gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain_vs_v6": effects["h2_h5_nll_vs_v6"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_gain_vs_capacity": effects["h2_h5_nll_vs_capacity"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_capacity_control"],
        "task_breadth": effects["h2_h5_nll_vs_v6"]["positive_task_fraction"] >= gate["minimum_positive_task_fraction"],
        "seed_replication": sum(value > 0 for value in effects["h2_h5_nll_vs_v6"]["seeds"].values()) >= gate["minimum_positive_seeds"],
        "future_joint_exact": abs(effects["future_joint_ce_vs_v6"]["mean"]) <= gate["maximum_absolute_future_joint_ce_difference_vs_v6"],
        "all_legal": all(row["legal_prediction"] == 1 for row in current),
    }
    decision = "GO_PREDICTED_EVENT_GRAPH_WORLD_MODEL_V13" if all(checks.values()) else "NO_GO_PREDICTED_EVENT_GRAPH_WORLD_MODEL_V13"
    failed = [name for name, passed in checks.items() if not passed]
    if decision.startswith("GO"):
        diagnosis = "A causal predicted event graph preserves the v6 outcome branch and improves task-disjoint multi-step action dynamics."
    elif not checks["graph_predictive_gain"]:
        diagnosis = "The event graph is useful as an oracle but is not sufficiently predictable from causal state and action under task-disjoint transfer."
    else:
        diagnosis = "The event graph is predictable, but predicted-graph errors or conditioning prevent transferable action-dynamics gains."
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "diagnosis": diagnosis,
        "failed_checks": failed,
        "gate_checks": checks,
        "effects": effects,
        "counts": {"v6": len(external), "control": len(control), "candidate": len(candidate), "composite": len(composite)},
        "predictions_sha256": file_sha256(args.predictions),
        "composite_predictions_sha256": file_sha256(args.composite_predictions),
        "run_metrics_sha256": file_sha256(args.run_metrics),
    }
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    lines = ["# Predicted event-graph world model v13", "", f"Decision: `{decision}`", "", diagnosis, "", "## Frozen gate", ""]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    lines.extend(["", "## Task-macro paired effects", ""])
    for name, value in effects.items():
        lines.append(f"- {name}: {value['mean']:.6f}")
    lines.extend(["", "The four-cell outcome probabilities are copied exactly from frozen v6; the predicted graph is used only by the action-dynamics branch.", ""])
    args.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
