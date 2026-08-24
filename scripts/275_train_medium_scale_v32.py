"""Train the frozen v32 medium-scale modular diagnostic on one GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.intervention_modular_world_model import (
    InterventionSharedEffectTransition,
    assert_transition_only,
    trainable_parameter_count,
)
from wmagentattack.multisource_suitability import file_sha256


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v6 = load("v32_v6", ROOT / "scripts" / "203_train_structured_residual_v6.py")
v5 = v6.v5
v22 = load("v32_v22", ROOT / "scripts" / "251_train_long_horizon_controls_v22.py")
v21 = load("v32_v21", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")
v20 = v21.v20


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_effect(
    data: dict[str, Any], cfg: dict[str, Any], fold: int, seed: int, device: str
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    v20.set_seed(seed * 1009 + fold)
    states = torch.tensor(data["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    targets = torch.tensor(data["targets"], dtype=torch.float32, device=device)
    execution = torch.tensor(data["execution"], dtype=torch.float32, device=device)
    train_indices = np.asarray(
        [i for i, row in enumerate(data["rows"]) if int(row["confirmation_fold"]) != fold]
    )
    train = torch.tensor(train_indices, dtype=torch.long, device=device)
    task_counts: dict[str, int] = defaultdict(int)
    for index in train_indices:
        task_counts[str(data["rows"][index]["task_id"])] += 1
    row_weights = torch.tensor(
        [1.0 / task_counts[str(data["rows"][i]["task_id"])] for i in train_indices],
        dtype=torch.float32,
        device=device,
    )
    positives = targets[train].sum(0)
    negatives = len(train_indices) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 10.0)
    positive_execution = execution[train].sum()
    execution_pos_weight = torch.clamp(
        (len(train_indices) - positive_execution) / torch.clamp(positive_execution, min=1.0),
        1.0,
        10.0,
    )
    model = InterventionSharedEffectTransition(
        states.shape[1], actions.shape[1], int(cfg["hidden_size"]), targets.shape[1]
    ).to(device)
    assert_transition_only(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    train_sequences = [
        sequence
        for sequence in data["sequences"]
        if int(data["rows"][sequence[0]]["confirmation_fold"]) != fold
    ]
    train_pairs = [
        (left, right)
        for left, right in data["pairs"]
        if int(data["rows"][left]["confirmation_fold"]) != fold
        and int(data["rows"][right]["confirmation_fold"]) != fold
    ]
    local_index = {int(global_index): local for local, global_index in enumerate(train_indices)}
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        logits, execution_logits = model(states[train], actions[train])
        effect = v20.weighted_effect_loss(logits, targets[train], row_weights, pos_weight)
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits, execution[train], pos_weight=execution_pos_weight
        )
        recurrent = torch.stack(
            [v20.sequence_loss(model, "intervention_modular_v20", seq, states, actions, targets, pos_weight) for seq in train_sequences]
        ).mean()
        probabilities = torch.sigmoid(logits)
        paired = torch.stack(
            [
                F.mse_loss(
                    probabilities[local_index[left]] - probabilities[local_index[right]],
                    targets[left] - targets[right],
                )
                for left, right in train_pairs
            ]
        ).mean()
        total = (
            effect
            + float(cfg["execution_weight"]) * execution_loss
            + float(cfg["sequence_weight"]) * recurrent
            + float(cfg["pair_weight"]) * paired
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
        optimizer.step()
        if epoch in (0, int(cfg["epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total": float(total.detach()),
                    "effect": float(effect.detach()),
                    "execution": float(execution_loss.detach()),
                    "sequence": float(recurrent.detach()),
                    "pair": float(paired.detach()),
                }
            )
    return model.cpu(), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--action-dataset", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--effect-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v32 protocol is not frozen")
    for path, expected in (
        (args.action_dataset, protocol["data"]["action_dataset_sha256"]),
        (args.action_audit, protocol["data"]["action_audit_sha256"]),
        (args.effect_dataset, protocol["data"]["effect_dataset_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v32 requires exactly one Slurm-isolated CUDA device")
    device = "cuda"
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    action_predictions = args.output_dir / "action_predictions.jsonl"
    action_predictions.write_text("", encoding="utf-8")
    action_data = json.loads(args.action_dataset.read_text(encoding="utf-8"))
    action_cfg = protocol["training"]["action_branch"]
    action_runs = []
    parameter_counts: dict[str, int] = {}
    for fold in range(5):
        events = v5._fold(action_data, fold)
        arrays = v5._arrays(events, action_data["candidate_catalog"], int(action_cfg["hash_dimension"]))
        surfaces = v6.horizons(events, arrays, max_h=10)
        for seed in protocol["training"]["seeds"]:
            seed = int(seed)
            v6.seed(seed)
            teacher_protocol = {"training": copy.deepcopy(action_cfg["teacher"])}
            values = v5._train(
                "structured_joint_aux", events, arrays, teacher_protocol, seed, device, return_model=True
            )
            teacher = values[4]
            residual_protocol = {"residual_training": copy.deepcopy(action_cfg["residual"])}
            residual, context, logits, history = v6.train_residual(
                teacher, events, arrays, surfaces, residual_protocol, seed, device
            )
            parameter_counts["action_teacher"] = sum(p.numel() for p in teacher.parameters())
            parameter_counts["action_residual"] = sum(p.numel() for p in residual.parameters())
            rows = v22.evaluate_controls(
                residual, teacher, context, logits, events, arrays, surfaces, fold, seed, device
            )
            append(action_predictions, rows)
            checkpoint = checkpoints / f"action_fold{fold}_seed{seed}.pt"
            torch.save(
                {
                    "fold": fold,
                    "seed": seed,
                    "teacher": cpu_state_dict(teacher),
                    "residual": cpu_state_dict(residual),
                },
                checkpoint,
            )
            action_runs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "prediction_rows": len(rows),
                    "teacher_history": values[3]["history"],
                    "residual_history": history,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": file_sha256(checkpoint),
                }
            )
            del teacher, residual, context, logits
            torch.cuda.empty_cache()

    effect_data_raw = json.loads(args.effect_dataset.read_text(encoding="utf-8"))
    effect_cfg = protocol["training"]["effect_branch"]
    effect_base = v20.arrays(
        effect_data_raw, int(effect_cfg["state_hash_dimension"]), int(effect_cfg["action_hash_dimension"])
    )
    effect_runs = []
    for fold, (split_name, split) in enumerate(
        sorted(effect_data_raw["split_manifest"]["task_disjoint"].items())
    ):
        data, train_indices, test_indices = v21.materialize_split(
            effect_base, split, fold, sequences_enabled=True
        )
        for seed in protocol["training"]["seeds"]:
            seed = int(seed)
            model, history = train_effect(data, effect_cfg, fold, seed, device)
            parameter_counts["effect_transition"] = trainable_parameter_count(model)
            row = v21.evaluate(
                model,
                "intervention_modular_v20",
                protocol["training"]["effect_candidate_arm"],
                "task_disjoint",
                split_name,
                fold,
                seed,
                data,
                train_indices,
                test_indices,
                effect_data_raw["effect_token_vocabulary"],
            )
            row["history"] = history
            checkpoint = checkpoints / f"effect_fold{fold}_seed{seed}.pt"
            torch.save({"fold": fold, "seed": seed, "model": model.state_dict()}, checkpoint)
            row["checkpoint"] = checkpoint.name
            row["checkpoint_sha256"] = file_sha256(checkpoint)
            effect_runs.append(row)

    expected_action_units = int(protocol["fixed_budget"]["action_paired_units"])
    expected_effect_fits = int(protocol["fixed_budget"]["effect_fits"])
    if len(action_runs) != expected_action_units or len(effect_runs) != expected_effect_fits:
        raise RuntimeError("v32 fixed fit budget incomplete")
    payload = {
        "schema_version": "wmagentattack.medium_scale_training.v32",
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "action_dataset_sha256": file_sha256(args.action_dataset),
        "effect_dataset_sha256": file_sha256(args.effect_dataset),
        "completed_action_paired_units": len(action_runs),
        "completed_effect_fits": len(effect_runs),
        "completed_model_fits": len(action_runs) * 2 + len(effect_runs),
        "runtime_failures": 0,
        "parameter_counts": parameter_counts,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "action_predictions_sha256": file_sha256(action_predictions),
        "action_runs": action_runs,
        "effect_runs": effect_runs,
    }
    write(args.output_dir / "run_metrics.json", payload)
    print(json.dumps({key: payload[key] for key in ("completed_model_fits", "runtime_failures", "parameter_counts", "peak_cuda_memory_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
