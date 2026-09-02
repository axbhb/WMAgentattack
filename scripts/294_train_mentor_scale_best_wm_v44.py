"""Run the frozen v44 large-replication experiment without content checksums."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v5 = load("v44_v5", ROOT / "scripts" / "201_train_structured_joint_outcome_v5.py")
v6 = load("v44_v6", ROOT / "scripts" / "203_train_structured_residual_v6.py")
v21 = load("v44_v21", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")
v22 = load("v44_v22", ROOT / "scripts" / "251_train_long_horizon_controls_v22.py")
v20 = v21.v20


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def augment_joint_prior_brier(rows, events, arrays, prior):
    index = {event["event_id"]: i for i, event in enumerate(events)}
    prior = np.asarray(prior, dtype=np.float32)
    for row in rows:
        if row["joint_trainable"]:
            target = arrays["joint"][index[row["event_id"]]]
            row["joint_prior_brier"] = float(((prior - target) ** 2).mean())
        else:
            row["joint_prior_brier"] = None
    return rows


def assert_dataset_contract(action_data: dict, effect_data: dict, protocol: dict) -> None:
    expected = protocol["data"]
    events = action_data.get("events", [])
    tasks = {str(row["task_name"]) for row in events}
    suites = {task.split("|")[0] for task in tasks}
    trajectories = {str(row["trajectory_id"]) for row in events}
    if len(events) != int(expected["expected_event_rows"]):
        raise ValueError("unexpected action event count")
    adjacent = sum(row.get("next_target_candidate_id") is not None for row in events)
    if adjacent != int(expected["expected_adjacent_transitions"]):
        raise ValueError("unexpected adjacent-transition count")
    if len(trajectories) != int(expected["expected_trajectories"]):
        raise ValueError("unexpected trajectory count")
    if len(tasks) != int(expected["expected_tasks"]) or len(suites) != int(expected["expected_suites"]):
        raise ValueError("unexpected task/suite count")
    if len(effect_data.get("rows", [])) != int(expected["expected_effect_rows"]):
        raise ValueError("unexpected effect row count")
    for fold in action_data["folds"]:
        if set(fold["train_tasks"]) & set(fold["test_tasks"]):
            raise ValueError("action task leakage")
    effect_task = {
        str(row["transition_ref"]): str(row["task_id"])
        for row in effect_data["rows"]
    }
    for split in effect_data["split_manifest"]["task_disjoint"].values():
        train_tasks = {effect_task[str(ref)] for ref in split["train_refs"]}
        test_tasks = {effect_task[str(ref)] for ref in split["test_refs"]}
        if train_tasks & test_tasks:
            raise ValueError("effect task leakage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--action-dataset", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--effect-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v44 protocol is not frozen")
    action_audit = json.loads(args.action_audit.read_text(encoding="utf-8"))
    if not action_audit.get("passed", False):
        raise ValueError("action dataset audit did not pass")
    action_data = json.loads(args.action_dataset.read_text(encoding="utf-8"))
    effect_data = json.loads(args.effect_dataset.read_text(encoding="utf-8"))
    assert_dataset_contract(action_data, effect_data, protocol)

    device = args.device
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("CUDA execution requires exactly one Slurm-isolated GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8 if device == "cpu" else 4)

    output = args.output_dir
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    joint_predictions = output / "joint_predictions.jsonl"
    action_predictions = output / "action_predictions.jsonl"
    joint_predictions.write_text("", encoding="utf-8")
    action_predictions.write_text("", encoding="utf-8")

    seeds = [int(value) for value in protocol["training"]["seeds"]]
    teacher_cfg = copy.deepcopy(protocol["training"]["teacher"])
    residual_cfg = copy.deepcopy(protocol["training"]["residual"])
    action_runs = []
    parameter_counts = {}

    for fold in range(int(protocol["fixed_budget"]["action_folds"])):
        events = v5._fold(action_data, fold)
        arrays = v5._arrays(events, action_data["candidate_catalog"], int(teacher_cfg["hash_dimension"]))
        surfaces = v6.horizons(events, arrays, max_h=10)
        for seed in seeds:
            baseline = v5._train(
                "structured_baseline", events, arrays, {"training": copy.deepcopy(teacher_cfg)},
                seed, device, return_model=True,
            )
            baseline_rows = v5._predictions(
                events, arrays, *baseline[:3], baseline[3], fold, "structured_baseline", seed
            )
            append_jsonl(joint_predictions, augment_joint_prior_brier(
                baseline_rows, events, arrays, baseline[3]["joint_prior"]
            ))

            joint = v5._train(
                "structured_joint_aux", events, arrays, {"training": copy.deepcopy(teacher_cfg)},
                seed, device, return_model=True,
            )
            joint_rows = v5._predictions(
                events, arrays, *joint[:3], joint[3], fold, "structured_joint_aux", seed
            )
            append_jsonl(joint_predictions, augment_joint_prior_brier(
                joint_rows, events, arrays, joint[3]["joint_prior"]
            ))

            teacher = joint[4]
            parameter_counts["structured_baseline"] = sum(p.numel() for p in baseline[4].parameters())
            parameter_counts["structured_joint_aux"] = sum(p.numel() for p in teacher.parameters())
            residual, context, logits, residual_history = v6.train_residual(
                teacher, events, arrays, surfaces,
                {"residual_training": copy.deepcopy(residual_cfg)}, seed, device,
            )
            parameter_counts["zero_init_residual"] = sum(p.numel() for p in residual.parameters())
            control_rows = v22.evaluate_controls(
                residual, teacher, context, logits, events, arrays, surfaces,
                fold, seed, device,
            )
            append_jsonl(action_predictions, control_rows)
            torch.save(
                {
                    "fold": fold,
                    "seed": seed,
                    "joint_teacher": cpu_state(teacher),
                    "residual": cpu_state(residual),
                },
                checkpoints / f"action_fold{fold}_seed{seed}.pt",
            )
            action_runs.append({
                "fold": fold,
                "seed": seed,
                "joint_teacher_history": joint[3]["history"],
                "baseline_history": baseline[3]["history"],
                "residual_history": residual_history,
                "joint_prediction_rows": len(joint_rows),
                "action_prediction_rows": len(control_rows),
            })
            del baseline, joint, teacher, residual, context, logits
            if device == "cuda":
                torch.cuda.empty_cache()

    effect_cfg = copy.deepcopy(protocol["training"]["effect"])
    effect_base = v20.arrays(
        effect_data, int(effect_cfg["state_hash_dimension"]), int(effect_cfg["action_hash_dimension"])
    )
    effect_runs = []
    effect_arms = ("structured_residual_v6", "intervention_no_execution_experts_v21")
    for fold, (split_name, split) in enumerate(sorted(effect_data["split_manifest"]["task_disjoint"].items())):
        data, train_indices, test_indices = v21.materialize_split(
            effect_base, split, fold, sequences_enabled=True
        )
        for arm in effect_arms:
            for seed in seeds:
                (model, history), runtime_arm = v21.train_alias(
                    arm, fold, seed, data, effect_cfg
                )
                parameter_counts[arm] = v21.trainable_parameter_count(model)
                row = v21.evaluate(
                    model, runtime_arm, arm, "task_disjoint", split_name, fold, seed,
                    data, train_indices, test_indices, effect_data["effect_token_vocabulary"],
                )
                row["history"] = history
                effect_runs.append(row)
                if arm == "intervention_no_execution_experts_v21":
                    torch.save(
                        {"fold": fold, "seed": seed, "model": cpu_state(model)},
                        checkpoints / f"effect_fold{fold}_seed{seed}.pt",
                    )

    completed = (
        len(action_runs) * 3
        + sum(1 for row in effect_runs if row["arm"] == "structured_residual_v6")
        + sum(1 for row in effect_runs if row["arm"] == "intervention_no_execution_experts_v21")
    )
    expected = int(protocol["fixed_budget"]["total_model_fits"])
    if completed != expected:
        raise RuntimeError(f"incomplete v44 budget: {completed} != {expected}")
    payload = {
        "schema_version": "wmagentattack.mentor_scale_best_wm.v44",
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "completed_model_fits": completed,
        "runtime_failures": 0,
        "parameter_counts": parameter_counts,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "action_runs": action_runs,
        "effect_runs": effect_runs,
    }
    write_json(output / "run_metrics.json", payload)
    print(json.dumps({
        "completed_model_fits": completed,
        "runtime_failures": 0,
        "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
