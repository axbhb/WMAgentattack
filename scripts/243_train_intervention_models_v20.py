"""Run the frozen 3-arm, task-disjoint v20 transition-model comparison."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_evidence_probe import hashed_text
from wmagentattack.hybrid_semantic_world_model import semantic_state_v3_feature_vector
from wmagentattack.intervention_modular_world_model import (
    DirectStructuredTransition,
    InterventionModularTransition,
    RecurrentResidualTransition,
    assert_transition_only,
    trainable_parameter_count,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def arrays(dataset: dict[str, Any], state_hash: int, action_hash: int) -> dict[str, Any]:
    rows = dataset["transitions"]
    vocabulary = dataset["effect_token_vocabulary"]
    token_index = {token: index for index, token in enumerate(vocabulary)}
    states = np.stack([
        semantic_state_v3_feature_vector(
            row["model_input"]["current_semantic_state"], hash_dimension=state_hash
        )
        for row in rows
    ]).astype(np.float32)
    actions = np.stack([
        hashed_text(row["model_input"]["normalized_action"], action_hash, "v20-normalized-action")
        for row in rows
    ]).astype(np.float32)
    targets = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for token in row["model_target"]["effect_tokens"]:
            targets[row_index, token_index[token]] = 1.0
    execution = np.asarray(
        [row["model_target"]["execution_error"] for row in rows], dtype=np.float32
    )
    sequence_groups: dict[str, dict[int, int]] = defaultdict(dict)
    pair_groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row_index, row in enumerate(rows):
        for membership in row["group_memberships"]:
            sequence_ref = membership["sequence_ref"]
            step_index = int(membership["step_index"])
            if sequence_ref is not None:
                existing = sequence_groups[str(sequence_ref)].get(step_index)
                if existing is not None and existing != row_index:
                    raise ValueError("one sequence step maps to multiple canonical rows")
                sequence_groups[str(sequence_ref)][step_index] = row_index
            pair_groups[(str(membership["pair_ref"]), step_index)].add(row_index)
    sequences = [
        [mapping[index] for index in sorted(mapping)]
        for mapping in sequence_groups.values()
    ]
    if any(len(sequence) != 3 for sequence in sequences):
        raise ValueError("all v19 sequences must retain exactly three steps")
    pairs = []
    for indices in pair_groups.values():
        for left, right in itertools.combinations(sorted(indices), 2):
            if not np.array_equal(targets[left], targets[right]):
                pairs.append((left, right))
    return {
        "rows": rows,
        "states": states,
        "actions": actions,
        "targets": targets,
        "execution": execution,
        "sequences": sequences,
        "pairs": sorted(set(pairs)),
    }


def make_model(arm: str, state_size: int, action_size: int, targets: int, cfg: dict[str, Any]):
    hidden = int(cfg["hidden_sizes"][arm])
    if arm == "structured_markov_v3":
        model = DirectStructuredTransition(state_size, action_size, hidden, targets)
    elif arm == "structured_residual_v6":
        model = RecurrentResidualTransition(state_size, action_size, hidden, targets)
    elif arm == "intervention_modular_v20":
        model = InterventionModularTransition(state_size, action_size, hidden, targets)
    else:
        raise ValueError(arm)
    assert_transition_only(model)
    return model


def weighted_effect_loss(logits: Tensor, target: Tensor, weights: Tensor, pos_weight: Tensor) -> Tensor:
    per_token = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    per_row = per_token.mean(dim=1)
    return (per_row * weights).sum() / weights.sum()


def sequence_loss(model, arm: str, indices: list[int], states: Tensor, actions: Tensor, targets: Tensor, pos_weight: Tensor) -> Tensor:
    hidden = model.initial_hidden(states[indices[0] : indices[0] + 1])
    losses = []
    for index in indices:
        if arm == "structured_residual_v6":
            hidden = model.advance(hidden, actions[index : index + 1])
            logits, _ = model.predict_hidden(hidden)
        else:
            hidden, execution_logit = model.advance_with_execution(
                hidden, actions[index : index + 1]
            )
            logits = model.predict_hidden(hidden, execution_logit)
        losses.append(F.binary_cross_entropy_with_logits(
            logits, targets[index : index + 1], pos_weight=pos_weight
        ))
    return torch.stack(losses).mean()


def train_one(
    arm: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    set_seed(seed * 1009 + fold)
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    targets = torch.tensor(data["targets"])
    execution = torch.tensor(data["execution"])
    train_indices = np.asarray([
        index for index, row in enumerate(data["rows"])
        if int(row["confirmation_fold"]) != fold
    ])
    train = torch.tensor(train_indices, dtype=torch.long)
    task_counts = defaultdict(int)
    for index in train_indices:
        task_counts[data["rows"][index]["task_id"]] += 1
    row_weights = torch.tensor(
        [1.0 / task_counts[data["rows"][index]["task_id"]] for index in train_indices],
        dtype=torch.float32,
    )
    positives = targets[train].sum(dim=0)
    negatives = len(train_indices) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 10.0)
    positive_execution = execution[train].sum()
    execution_pos_weight = torch.clamp(
        (len(train_indices) - positive_execution) / torch.clamp(positive_execution, min=1.0),
        1.0,
        10.0,
    )
    model = make_model(
        arm, states.shape[1], actions.shape[1], targets.shape[1], cfg
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_sequences = [
        sequence for sequence in data["sequences"]
        if int(data["rows"][sequence[0]]["confirmation_fold"]) != fold
    ]
    train_pairs = [
        (left, right) for left, right in data["pairs"]
        if int(data["rows"][left]["confirmation_fold"]) != fold
        and int(data["rows"][right]["confirmation_fold"]) != fold
    ]
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        logits, execution_logits = model(states[train], actions[train])
        effect = weighted_effect_loss(
            logits, targets[train], row_weights, pos_weight
        )
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits,
            execution[train],
            pos_weight=execution_pos_weight,
        )
        total = effect + float(cfg["execution_weight"]) * execution_loss
        recurrent = torch.zeros(())
        if arm in {"structured_residual_v6", "intervention_modular_v20"}:
            recurrent = torch.stack([
                sequence_loss(model, arm, sequence, states, actions, targets, pos_weight)
                for sequence in train_sequences
            ]).mean()
            total = total + float(cfg["sequence_weight"]) * recurrent
        paired = torch.zeros(())
        if arm == "intervention_modular_v20" and train_pairs:
            probabilities = torch.sigmoid(logits)
            paired = torch.stack([
                F.mse_loss(
                    probabilities[int(np.flatnonzero(train_indices == left)[0])]
                    - probabilities[int(np.flatnonzero(train_indices == right)[0])],
                    targets[left] - targets[right],
                )
                for left, right in train_pairs
            ]).mean()
            total = total + float(cfg["pair_weight"]) * paired
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
        optimizer.step()
        if epoch in {0, int(cfg["epochs"]) - 1}:
            history.append({
                "epoch": epoch,
                "total": float(total.detach()),
                "effect": float(effect.detach()),
                "execution": float(execution_loss.detach()),
                "sequence": float(recurrent.detach()),
                "pair": float(paired.detach()),
            })
    return model, history


def binary_cross_entropy(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))


def rollout_probabilities(model, arm: str, sequence: list[int], data: dict[str, Any]) -> np.ndarray:
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    predictions = []
    with torch.no_grad():
        if arm == "structured_markov_v3":
            initial = states[sequence[0] : sequence[0] + 1]
            for index in sequence:
                logits, _ = model(initial, actions[index : index + 1])
                predictions.append(torch.sigmoid(logits)[0].numpy())
        else:
            hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
            for index in sequence:
                if arm == "structured_residual_v6":
                    hidden = model.advance(hidden, actions[index : index + 1])
                    logits, _ = model.predict_hidden(hidden)
                else:
                    hidden, execution_logit = model.advance_with_execution(
                        hidden, actions[index : index + 1]
                    )
                    logits = model.predict_hidden(hidden, execution_logit)
                predictions.append(torch.sigmoid(logits)[0].numpy())
    return np.stack(predictions)


def evaluate(model, arm: str, fold: int, seed: int, data: dict[str, Any]) -> dict[str, Any]:
    test_indices = np.asarray([
        index for index, row in enumerate(data["rows"])
        if int(row["confirmation_fold"]) == fold
    ])
    states = torch.tensor(data["states"][test_indices])
    actions = torch.tensor(data["actions"][test_indices])
    model.eval()
    with torch.no_grad():
        logits, execution_logits = model(states, actions)
        probabilities = torch.sigmoid(logits).numpy()
        execution_probabilities = torch.sigmoid(execution_logits).numpy()
    target = data["targets"][test_indices]
    execution = data["execution"][test_indices]
    row_bce = binary_cross_entropy(probabilities, target).mean(axis=1)
    task_values = defaultdict(list)
    for local_index, global_index in enumerate(test_indices):
        task_values[data["rows"][global_index]["task_id"]].append(float(row_bce[local_index]))
    prediction_binary = probabilities >= 0.5
    true_positive = float(np.logical_and(prediction_binary, target == 1).sum())
    false_positive = float(np.logical_and(prediction_binary, target == 0).sum())
    false_negative = float(np.logical_and(~prediction_binary, target == 1).sum())
    f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1.0)
    index_lookup = {global_index: local for local, global_index in enumerate(test_indices)}
    pair_scores = []
    for left, right in data["pairs"]:
        if left not in index_lookup or right not in index_lookup:
            continue
        lp, rp = probabilities[index_lookup[left]], probabilities[index_lookup[right]]
        ly, ry = data["targets"][left], data["targets"][right]
        own = binary_cross_entropy(lp, ly).mean() + binary_cross_entropy(rp, ry).mean()
        swapped = binary_cross_entropy(lp, ry).mean() + binary_cross_entropy(rp, ly).mean()
        pair_scores.append(1.0 if own < swapped else 0.5 if own == swapped else 0.0)
    rollout_losses = []
    rollout_by_task = defaultdict(list)
    for sequence in data["sequences"]:
        if int(data["rows"][sequence[0]]["confirmation_fold"]) != fold:
            continue
        rollout = rollout_probabilities(model, arm, sequence, data)
        value = float(binary_cross_entropy(rollout, data["targets"][sequence]).mean())
        rollout_losses.append(value)
        rollout_by_task[data["rows"][sequence[0]]["task_id"]].append(value)
    return {
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "confirmation_rows": len(test_indices),
        "confirmation_tasks": len(task_values),
        "one_step_task_macro_bce": float(np.mean([np.mean(v) for v in task_values.values()])),
        "one_step_micro_f1": float(f1),
        "execution_brier": float(np.mean((execution_probabilities - execution) ** 2)),
        "execution_accuracy": float(np.mean((execution_probabilities >= 0.5) == execution)),
        "pair_assignment_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "pair_comparisons": len(pair_scores),
        "v19_rollout_task_macro_bce": float(np.mean([
            np.mean(values) for values in rollout_by_task.values()
        ])),
        "v19_rollout_sequences": len(rollout_losses),
        "parameter_count": trainable_parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "model_preregistered_before_results":
        raise ValueError("v20 model protocol is not frozen before results")
    if sha256(args.dataset) != protocol["union"]["sha256"]:
        raise ValueError("v20 union hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    data = arrays(dataset, int(cfg["state_hash_dimension"]), int(cfg["action_hash_dimension"]))
    torch.set_num_threads(int(cfg["torch_threads"]))
    runs = []
    for arm in cfg["arms"]:
        for fold in range(int(protocol["split_contract"]["folds"])):
            for seed in cfg["training_seeds"]:
                model, history = train_one(arm, fold, int(seed), data, cfg)
                metrics = evaluate(model, arm, fold, int(seed), data)
                metrics["history"] = history
                runs.append(metrics)
    expected = (
        len(cfg["arms"])
        * int(protocol["split_contract"]["folds"])
        * len(cfg["training_seeds"])
    )
    if len(runs) != expected:
        raise RuntimeError("incomplete fixed model budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "run_metrics.json", {
        "schema_version": "wmagentattack.intervention_models.v20",
        "dataset_sha256": sha256(args.dataset),
        "expected_runs": expected,
        "completed_runs": len(runs),
        "runtime_failures": 0,
        "pair_groups": len(data["pairs"]),
        "sequences": len(data["sequences"]),
        "runs": runs,
    })
    print(json.dumps({"completed_runs": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
