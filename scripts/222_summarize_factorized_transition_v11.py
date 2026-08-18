"""Apply the frozen v11 factorized-transition gate and preserve counterevidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.factorized_transition_labels import FACTOR_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import exact_sign_test, paired_bootstrap


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_means(rows, metric):
    values = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            values[row["task_name"]].append(float(value))
    return {task: float(np.mean(task_values)) for task, task_values in values.items()}


def _paired_effect(left, right, metric, *, higher_is_better=False, bootstrap_seed=81800):
    seed_effects = {}
    for training_seed in (7, 17, 29):
        left_task = _task_means(
            [row for row in left if int(row["training_seed"]) == training_seed], metric
        )
        right_task = _task_means(
            [row for row in right if int(row["training_seed"]) == training_seed], metric
        )
        if set(left_task) != set(right_task):
            raise ValueError(f"task mismatch for {metric} seed {training_seed}")
        seed_effects[training_seed] = {
            task: (
                right_task[task] - left_task[task]
                if higher_is_better else left_task[task] - right_task[task]
            )
            for task in left_task
        }
    tasks = set(seed_effects[7])
    paired = {
        task: float(np.mean([seed_effects[seed][task] for seed in seed_effects]))
        for task in tasks
    }
    seeds = {
        str(seed): float(np.mean(list(task_values.values())))
        for seed, task_values in seed_effects.items()
    }
    return {
        "mean": float(np.mean(list(seeds.values()))),
        "seeds": seeds,
        "tasks": paired,
        "positive_task_fraction": float(np.mean([value > 0 for value in paired.values()])),
        "paired_bootstrap": paired_bootstrap(
            list(paired.values()), draws=10000, seed=bootstrap_seed
        ),
        "exact_sign_test": exact_sign_test(list(paired.values())),
    }


def _factor_effect(rows, name):
    # Retain identical task/seed aggregation for predictions and train-only priors.
    prior_rows = []
    for row in rows:
        copied = dict(row)
        copied[f"{name}_nll"] = row[f"{name}_prior_nll"]
        copied[f"{name}_correct"] = row[f"{name}_prior_correct"]
        prior_rows.append(copied)
    nll = _paired_effect(
        prior_rows, rows, f"{name}_nll", bootstrap_seed=81810 + list(FACTOR_CLASSES).index(name)
    )
    accuracy = _paired_effect(
        prior_rows, rows, f"{name}_correct", higher_is_better=True,
        bootstrap_seed=81820 + list(FACTOR_CLASSES).index(name),
    )
    return {"nll_gain_vs_train_prior": nll, "accuracy_gain_vs_train_prior": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    current = _read(args.predictions)
    run_metrics = json.loads(args.run_metrics.read_text())
    external_path = Path(protocol["external_v6_control"]["predictions"])
    if file_sha256(external_path) != protocol["external_v6_control"]["sha256"]:
        raise ValueError("external v6 prediction hash mismatch")
    external = _read(external_path)
    v6 = [row for row in external if row["arm"] == "structured_residual_v6"]
    predicted = [row for row in current if row["arm"] == "factorized_predicted_f11"]
    capacity = [row for row in current if row["arm"] == "capacity_control_f11"]
    oracle = [row for row in current if row["arm"] == "factorized_oracle_diagnostic_f11"]
    if not (len(predicted) == len(capacity) == len(oracle) == len(v6)):
        raise ValueError("paired prediction budget mismatch")

    effects = {
        "h1_nll_vs_v6": _paired_effect(
            [row for row in v6 if row["horizon"] == 1],
            [row for row in predicted if row["horizon"] == 1], "action_nll",
            bootstrap_seed=81831,
        ),
        "h1_accuracy_vs_v6": _paired_effect(
            [row for row in v6 if row["horizon"] == 1],
            [row for row in predicted if row["horizon"] == 1], "action_correct",
            higher_is_better=True, bootstrap_seed=81832,
        ),
        "h2_h5_nll_vs_v6": _paired_effect(
            [row for row in v6 if row["horizon"] >= 2],
            [row for row in predicted if row["horizon"] >= 2], "action_nll",
            bootstrap_seed=81833,
        ),
        "h2_h5_nll_vs_capacity": _paired_effect(
            [row for row in capacity if row["horizon"] >= 2],
            [row for row in predicted if row["horizon"] >= 2], "action_nll",
            bootstrap_seed=81834,
        ),
        "future_joint_ce_vs_v6": _paired_effect(
            [row for row in v6 if row["horizon"] >= 2 and row["joint_trainable"]],
            [row for row in predicted if row["horizon"] >= 2 and row["joint_trainable"]],
            "joint_ce", bootstrap_seed=81835,
        ),
        "oracle_h2_h5_nll_vs_predicted": _paired_effect(
            [row for row in predicted if row["horizon"] >= 2],
            [row for row in oracle if row["horizon"] >= 2], "action_nll",
            bootstrap_seed=81836,
        ),
        "oracle_h2_h5_nll_vs_v6": _paired_effect(
            [row for row in v6 if row["horizon"] >= 2],
            [row for row in oracle if row["horizon"] >= 2], "action_nll",
            bootstrap_seed=81837,
        ),
    }
    h1_predicted = [row for row in predicted if row["horizon"] == 1]
    factor_effects = {
        name: _factor_effect(h1_predicted, name) for name in FACTOR_CLASSES
    }
    gate = protocol["stage_f2_model"]["gate"]
    checks = {
        "fixed_budget_complete": run_metrics["training_units"] == 30,
        "runtime_clean": run_metrics["runtime_failures"] == 0,
        "parameter_matched": bool(run_metrics["parameter_match"]),
        "paired_rows_complete": len(predicted) == len(v6),
        "each_factor_nll_gain": all(
            value["nll_gain_vs_train_prior"]["mean"]
            >= gate["minimum_each_factor_nll_gain_vs_train_prior"]
            for value in factor_effects.values()
        ),
        "each_factor_accuracy_gain": all(
            value["accuracy_gain_vs_train_prior"]["mean"]
            >= gate["minimum_each_factor_accuracy_gain_vs_train_prior"]
            for value in factor_effects.values()
        ),
        "h1_nll_noninferiority": effects["h1_nll_vs_v6"]["mean"]
        >= -gate["maximum_h1_nll_degradation_vs_v6"],
        "h1_accuracy_noninferiority": effects["h1_accuracy_vs_v6"]["mean"]
        >= -gate["maximum_h1_accuracy_degradation_vs_v6"],
        "h2_h5_gain_vs_v6": effects["h2_h5_nll_vs_v6"]["mean"]
        >= gate["minimum_h2_h5_nll_gain_vs_v6"],
        "h2_h5_gain_vs_capacity": effects["h2_h5_nll_vs_capacity"]["mean"]
        >= gate["minimum_h2_h5_nll_gain_vs_capacity_control"],
        "h2_h5_task_breadth": effects["h2_h5_nll_vs_v6"]["positive_task_fraction"]
        >= gate["minimum_positive_task_fraction"],
        "h2_h5_seed_replication": sum(
            value > 0 for value in effects["h2_h5_nll_vs_v6"]["seeds"].values()
        ) >= gate["minimum_positive_seeds"],
        "future_joint_noninferiority": effects["future_joint_ce_vs_v6"]["mean"]
        >= -gate["maximum_future_joint_ce_degradation"],
        "all_predictions_legal": all(row["legal_prediction"] == 1 for row in current),
    }
    decision = (
        "GO_FACTORIZED_SEMANTIC_TRANSITION_V11"
        if all(checks.values()) else "NO_GO_FACTORIZED_SEMANTIC_TRANSITION_V11"
    )
    oracle_gain = effects["oracle_h2_h5_nll_vs_predicted"]["mean"]
    if decision.startswith("GO"):
        diagnosis = "Predicted semantic factors transfer and add multi-step value beyond capacity."
    elif oracle_gain >= gate["minimum_h2_h5_nll_gain_vs_capacity_control"]:
        diagnosis = "Factor information is useful when supplied correctly, but factor prediction is the bottleneck."
    else:
        diagnosis = "The proposed factor targets do not add sufficient action-dynamics information even under oracle conditioning."
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "diagnosis": diagnosis,
        "gate_checks": checks,
        "effects": effects,
        "factor_effects": factor_effects,
        "counts": {
            "v6_rows": len(v6), "predicted_rows": len(predicted),
            "capacity_rows": len(capacity), "oracle_rows": len(oracle),
        },
        "predictions_sha256": file_sha256(args.predictions),
        "run_metrics_sha256": file_sha256(args.run_metrics),
        "external_v6_sha256": file_sha256(external_path),
        "oracle_is_diagnostic_only": True,
    }
    _text = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    args.output.write_text(_text, encoding="utf-8")
    lines = [
        "# Factorized semantic-transition v11", "", f"Decision: `{decision}`", "",
        diagnosis, "", "## Frozen gate", "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )
    lines.extend(["", "## Main paired effects", ""])
    for name, value in effects.items():
        lines.append(f"- {name}: {value['mean']:.6f}")
    lines.extend(["", "## Factor prediction gains over train priors", ""])
    for name, value in factor_effects.items():
        lines.append(
            f"- {name}: NLL {value['nll_gain_vs_train_prior']['mean']:.6f}; "
            f"accuracy {value['accuracy_gain_vs_train_prior']['mean']:.6f}"
        )
    lines.extend(["", "Oracle results are diagnostic only and cannot authorize the model.", ""])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
