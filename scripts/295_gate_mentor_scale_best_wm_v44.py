"""Apply the frozen v44 task-level gates and select the retained architecture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values))


def grouped(rows, group_keys, value_key):
    values = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            values[tuple(row[key] for key in group_keys)].append(float(row[value_key]))
    return {key: float(np.mean(group)) for key, group in values.items()}


def paired_gain(candidate, baseline):
    keys = sorted(set(candidate) & set(baseline))
    gains = {key: baseline[key] - candidate[key] for key in keys}
    return gains


def positive_fraction(values):
    return float(np.mean([value > 0 for value in values])) if values else 0.0


def evaluate(protocol, metrics, joint_rows, action_rows):
    gate = protocol["acceptance_gate"]
    clauses = []

    baseline = [r for r in joint_rows if r["arm"] == "structured_baseline"]
    joint = [r for r in joint_rows if r["arm"] == "structured_joint_aux"]
    base_action = grouped(baseline, ("training_seed", "task_name"), "action_nll")
    joint_action = grouped(joint, ("training_seed", "task_name"), "action_nll")
    action_gains = paired_gain(joint_action, base_action)
    base_acc = grouped(baseline, ("training_seed", "task_name"), "action_correct")
    joint_acc = grouped(joint, ("training_seed", "task_name"), "action_correct")
    acc_gains = {key: joint_acc[key] - base_acc[key] for key in set(joint_acc) & set(base_acc)}
    joint_ce_gain = mean(joint, "joint_prior_cross_entropy") - mean(joint, "joint_cross_entropy")
    joint_brier_gain = mean(joint, "joint_prior_brier") - mean(joint, "joint_brier")
    outcome_degradation = mean(joint, "outcome_bce") - mean(baseline, "outcome_bce")
    seed_action = grouped(joint, ("training_seed",), "action_nll")
    seed_base_action = grouped(baseline, ("training_seed",), "action_nll")
    positive_action_seeds = sum(seed_action[s] < seed_base_action[s] for s in seed_action)
    task_action = grouped(joint, ("task_name",), "action_nll")
    task_base_action = grouped(baseline, ("task_name",), "action_nll")
    four = {
        "action_nll_gain": float(np.mean(list(action_gains.values()))),
        "action_accuracy_gain": float(np.mean(list(acc_gains.values()))),
        "joint_ce_gain_over_prior": joint_ce_gain,
        "joint_brier_gain_over_prior": joint_brier_gain,
        "outcome_bce_degradation": outcome_degradation,
        "positive_task_fraction": positive_fraction([
            task_base_action[t] - task_action[t] for t in task_action
        ]),
        "positive_seeds": int(positive_action_seeds),
    }
    fcfg = gate["four_cell"]
    four_checks = {
        "action_nll_gain": four["action_nll_gain"] >= fcfg["minimum_action_nll_gain"],
        "action_accuracy_gain": four["action_accuracy_gain"] >= fcfg["minimum_action_accuracy_gain"],
        "joint_ce_gain": four["joint_ce_gain_over_prior"] >= fcfg["minimum_joint_ce_gain_over_prior"],
        "joint_brier_gain": four["joint_brier_gain_over_prior"] >= fcfg["minimum_joint_brier_gain_over_prior"],
        "outcome_noninferiority": four["outcome_bce_degradation"] <= fcfg["maximum_outcome_bce_degradation"],
        "task_replication": four["positive_task_fraction"] >= fcfg["minimum_positive_task_fraction"],
        "seed_replication": four["positive_seeds"] >= fcfg["minimum_positive_seeds"],
    }
    four_pass = all(four_checks.values())

    repeated = [r for r in action_rows if r["control"] == "one_step_repeated"]
    residual = [r for r in action_rows if r["control"] == "free_latent_residual"]
    def action_map(rows, horizons, keys, metric):
        return grouped([r for r in rows if int(r["horizon"]) in horizons], keys, metric)
    h1_rep = action_map(repeated, {1}, ("training_seed", "task_name"), "action_nll")
    h1_res = action_map(residual, {1}, ("training_seed", "task_name"), "action_nll")
    h1_rep_acc = action_map(repeated, {1}, ("training_seed", "task_name"), "action_correct")
    h1_res_acc = action_map(residual, {1}, ("training_seed", "task_name"), "action_correct")
    multi_rep = action_map(repeated, {2, 3, 5}, ("training_seed", "task_name"), "action_nll")
    multi_res = action_map(residual, {2, 3, 5}, ("training_seed", "task_name"), "action_nll")
    multi_gains = paired_gain(multi_res, multi_rep)
    task_multi_rep = action_map(repeated, {2, 3, 5}, ("task_name",), "action_nll")
    task_multi_res = action_map(residual, {2, 3, 5}, ("task_name",), "action_nll")
    seed_multi_rep = action_map(repeated, {2, 3, 5}, ("training_seed",), "action_nll")
    seed_multi_res = action_map(residual, {2, 3, 5}, ("training_seed",), "action_nll")
    multi = {
        "h1_nll_degradation": float(np.mean([h1_res[k] - h1_rep[k] for k in h1_res])),
        "h1_accuracy_degradation": float(np.mean([h1_rep_acc[k] - h1_res_acc[k] for k in h1_res_acc])),
        "h2_h5_nll_gain_vs_repeated_teacher": float(np.mean(list(multi_gains.values()))),
        "positive_task_fraction": positive_fraction([
            task_multi_rep[t] - task_multi_res[t] for t in task_multi_res
        ]),
        "positive_seeds": int(sum(seed_multi_res[s] < seed_multi_rep[s] for s in seed_multi_res)),
        "legal_prediction_fraction": mean(residual, "legal_prediction"),
        "by_horizon": {
            str(h): {
                "repeated_nll": mean([r for r in repeated if int(r["horizon"]) == h], "action_nll"),
                "residual_nll": mean([r for r in residual if int(r["horizon"]) == h], "action_nll"),
                "repeated_accuracy": mean([r for r in repeated if int(r["horizon"]) == h], "action_correct"),
                "residual_accuracy": mean([r for r in residual if int(r["horizon"]) == h], "action_correct"),
            } for h in (1, 2, 3, 5, 10)
        },
    }
    mcfg = gate["multi_step"]
    multi_checks = {
        "h1_nll_noninferiority": multi["h1_nll_degradation"] <= mcfg["maximum_h1_nll_degradation"],
        "h1_accuracy_noninferiority": multi["h1_accuracy_degradation"] <= mcfg["maximum_h1_accuracy_degradation"],
        "h2_h5_gain": multi["h2_h5_nll_gain_vs_repeated_teacher"] >= mcfg["minimum_h2_h5_nll_gain_vs_repeated_teacher"],
        "task_replication": multi["positive_task_fraction"] >= mcfg["minimum_positive_task_fraction"],
        "seed_replication": multi["positive_seeds"] >= mcfg["minimum_positive_seeds"],
        "legality": (multi["legal_prediction_fraction"] == 1.0) if mcfg["require_all_predictions_legal"] else True,
    }
    multi_pass = all(multi_checks.values())

    effect_rows = metrics["effect_runs"]
    effect_base = [r for r in effect_rows if r["arm"] == "structured_residual_v6"]
    effect_new = [r for r in effect_rows if r["arm"] == "intervention_no_execution_experts_v21"]
    effect = {
        "hard_bce_gain": mean(effect_base, "hard_task_macro_bce") - mean(effect_new, "hard_task_macro_bce"),
        "positive_nll_gain": mean(effect_base, "hard_positive_task_macro_nll") - mean(effect_new, "hard_positive_task_macro_nll"),
        "positive_recall_gain": mean(effect_new, "hard_positive_task_macro_recall") - mean(effect_base, "hard_positive_task_macro_recall"),
        "rollout_bce_gain": mean(effect_base, "v19_rollout_hard_bce") - mean(effect_new, "v19_rollout_hard_bce"),
    }
    new_seed = grouped(effect_new, ("seed",), "hard_positive_task_macro_nll")
    base_seed = grouped(effect_base, ("seed",), "hard_positive_task_macro_nll")
    effect["positive_seeds"] = int(sum(new_seed[s] < base_seed[s] for s in new_seed))
    ecfg = gate["effect"]
    effect_checks = {
        "hard_bce_gain": effect["hard_bce_gain"] >= ecfg["minimum_hard_bce_gain"],
        "positive_nll_gain": effect["positive_nll_gain"] >= ecfg["minimum_positive_nll_gain"],
        "positive_recall_noninferiority": effect["positive_recall_gain"] >= -ecfg["positive_recall_noninferiority_margin"],
        "rollout_bce_gain": effect["rollout_bce_gain"] >= ecfg["minimum_rollout_bce_gain"],
        "seed_replication": effect["positive_seeds"] >= ecfg["minimum_positive_seeds"],
    }
    effect_pass = all(effect_checks.values())

    if four_pass and effect_pass:
        selected_action = "zero_init_residual" if multi_pass else "structured_joint_aux_teacher"
        decision = "GO_MENTOR_READY_CLOSED_VOCABULARY_V44"
    else:
        selected_action = "none"
        decision = "NO_GO_MENTOR_SCALE_V44"
    return {
        "decision": decision,
        "selected_action_model": selected_action,
        "four_cell": {"metrics": four, "checks": four_checks, "passed": four_pass},
        "multi_step": {"metrics": multi, "checks": multi_checks, "passed": multi_pass},
        "effect": {"metrics": effect, "checks": effect_checks, "passed": effect_pass},
        "completed_model_fits": metrics["completed_model_fits"],
        "runtime_failures": metrics["runtime_failures"],
        "claim_boundary": "Closed-vocabulary, existing-task, task-disjoint confirmation only; no open-vocabulary, new-task attack-selection, or causal attack-effect claim.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--joint-predictions", type=Path, required=True)
    parser.add_argument("--action-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics["completed_model_fits"] != protocol["fixed_budget"]["total_model_fits"]:
        raise ValueError("v44 fit budget mismatch")
    payload = evaluate(protocol, metrics, read_jsonl(args.joint_predictions), read_jsonl(args.action_predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "selected_action_model": payload["selected_action_model"]}, sort_keys=True))


if __name__ == "__main__":
    main()
