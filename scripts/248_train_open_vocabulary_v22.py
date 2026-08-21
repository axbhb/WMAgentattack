"""Frozen v22 open-vocabulary candidate-conditioned effect-token experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.compositional_effect_world_model import (
    CompositionalEffectTransition,
    IndependentLabelEffectTransition,
    effect_vocabulary_features,
    normalized_action_feature_vector,
)
from wmagentattack.intervention_modular_world_model import (
    InterventionSharedEffectTransition,
    assert_transition_only,
    trainable_parameter_count,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v20 = _load("v20_train", ROOT / "scripts" / "243_train_intervention_models_v20.py")
v21 = _load("v21_train", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")

ARMS = (
    "fixed_v21",
    "independent_candidate_control_v22",
    "compositional_candidate_v22",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int, fold: int) -> None:
    value = seed * 1009 + fold
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def augment_arrays(base: dict[str, Any], vocabulary: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    data = dict(base)
    data["action_semantics"] = np.stack([
        normalized_action_feature_vector(
            row["model_input"]["normalized_action"],
            int(cfg["action_semantic_dimension"]),
        )
        for row in data["rows"]
    ]).astype(np.float32)
    data["label_features"] = effect_vocabulary_features(
        vocabulary, int(cfg["label_hash_dimension"])
    )
    return data


def make_candidate(arm: str, data: dict[str, Any], cfg: dict[str, Any]):
    values = dict(
        state_size=data["states"].shape[1],
        action_size=data["actions"].shape[1],
        action_semantic_size=data["action_semantics"].shape[1],
        hidden_size=int(cfg["hidden_size"]),
    )
    if arm == "independent_candidate_control_v22":
        model = IndependentLabelEffectTransition(
            **values, targets=data["targets"].shape[1]
        )
    elif arm == "compositional_candidate_v22":
        model = CompositionalEffectTransition(
            **values, label_feature_size=data["label_features"].shape[1]
        )
    else:
        raise ValueError(arm)
    assert_transition_only(model)
    return model


def _effect_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    row_weights: torch.Tensor,
    pos_weight: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(
        logits[:, label_mask], target[:, label_mask], reduction="none",
        pos_weight=pos_weight[label_mask],
    ).mean(dim=1)
    return (per * row_weights).sum() / row_weights.sum()


def train_candidate(
    arm: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    cfg: dict[str, Any],
):
    set_seed(seed, fold)
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    action_semantics = torch.tensor(data["action_semantics"])
    labels = torch.tensor(data["label_features"])
    targets = torch.tensor(data["targets"])
    execution = torch.tensor(data["execution"])
    train_indices = np.asarray([
        index for index, row in enumerate(data["rows"])
        if int(row["confirmation_fold"]) != fold
    ])
    train = torch.tensor(train_indices, dtype=torch.long)
    observed = targets[train].sum(dim=0) > 0
    if not observed.any():
        raise ValueError("training fold has no observed positive labels")
    task_counts = defaultdict(int)
    for index in train_indices:
        task_counts[data["rows"][index]["task_id"]] += 1
    row_weights = torch.tensor([
        1.0 / task_counts[data["rows"][index]["task_id"]] for index in train_indices
    ])
    positives = targets[train].sum(dim=0)
    negatives = len(train_indices) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 10.0)
    positive_execution = execution[train].sum()
    execution_pos_weight = torch.clamp(
        (len(train_indices) - positive_execution) / torch.clamp(positive_execution, min=1.0),
        1.0, 10.0,
    )
    model = make_candidate(arm, data, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
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
    lookup = {value: local for local, value in enumerate(train_indices)}
    for epoch in range(int(cfg["epochs"])):
        model.train()
        logits, execution_logits = model(
            states[train], actions[train], action_semantics[train], labels
        )
        effect = _effect_loss(
            logits, targets[train], row_weights, pos_weight, observed
        )
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits, execution[train], pos_weight=execution_pos_weight
        )
        total = effect + float(cfg["execution_weight"]) * execution_loss
        recurrent = torch.zeros(())
        if train_sequences:
            losses = []
            for sequence in train_sequences:
                hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
                for index in sequence:
                    hidden, _ = model.advance_with_execution(
                        hidden, actions[index : index + 1], action_semantics[index : index + 1]
                    )
                    prediction = model.predict_hidden(hidden, labels)
                    losses.append(F.binary_cross_entropy_with_logits(
                        prediction[:, observed], targets[index : index + 1, observed],
                        pos_weight=pos_weight[observed],
                    ))
            recurrent = torch.stack(losses).mean()
            total = total + float(cfg["sequence_weight"]) * recurrent
        paired = torch.zeros(())
        if train_pairs:
            probabilities = torch.sigmoid(logits[:, observed])
            paired = torch.stack([
                F.mse_loss(
                    probabilities[lookup[left]] - probabilities[lookup[right]],
                    targets[left, observed] - targets[right, observed],
                ) for left, right in train_pairs
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
                "observed_training_labels": int(observed.sum()),
            })
    return model, history, observed.numpy()


def candidate_probabilities(model, indices: np.ndarray, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, execution = model(
            torch.tensor(data["states"][indices]),
            torch.tensor(data["actions"][indices]),
            torch.tensor(data["action_semantics"][indices]),
            torch.tensor(data["label_features"]),
        )
    return torch.sigmoid(logits).numpy(), torch.sigmoid(execution).numpy()


def candidate_rollout(model, sequence: list[int], data: dict[str, Any]) -> np.ndarray:
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    semantics = torch.tensor(data["action_semantics"])
    labels = torch.tensor(data["label_features"])
    predictions = []
    with torch.no_grad():
        hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
        for index in sequence:
            hidden, _ = model.advance_with_execution(
                hidden, actions[index : index + 1], semantics[index : index + 1]
            )
            predictions.append(torch.sigmoid(model.predict_hidden(hidden, labels))[0].numpy())
    return np.stack(predictions)


def _bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))


def _macro(values: np.ndarray, indices: np.ndarray, rows: list[dict]) -> float:
    grouped = defaultdict(list)
    for local, global_index in enumerate(indices):
        grouped[rows[global_index]["task_id"]].append(float(values[local]))
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def evaluate(
    model,
    arm: str,
    split_suite: str,
    split_name: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    vocabulary: list[str],
    observed: np.ndarray,
) -> dict[str, Any]:
    if arm == "fixed_v21":
        with torch.no_grad():
            logits, execution_logits = model(
                torch.tensor(data["states"][test_indices]),
                torch.tensor(data["actions"][test_indices]),
            )
        probabilities = torch.sigmoid(logits).numpy()
        execution_probability = torch.sigmoid(execution_logits).numpy()
    else:
        probabilities, execution_probability = candidate_probabilities(model, test_indices, data)
    target = data["targets"][test_indices]
    positives = target == 1
    unseen = np.logical_and(positives, ~observed[None, :])
    seen = np.logical_and(positives, observed[None, :])
    per_row_bce = _bce(probabilities, target).mean(axis=1)
    positive_nll = np.asarray([
        float((-np.log(np.clip(probabilities[i][positives[i]], 1e-7, 1.0))).mean())
        for i in range(len(test_indices))
    ])
    positive_recall = np.asarray([
        float((probabilities[i][positives[i]] >= 0.5).mean())
        for i in range(len(test_indices))
    ])
    test_lookup = {global_index: local for local, global_index in enumerate(test_indices)}
    pair_scores = []
    for left, right in data["pairs"]:
        if left not in test_lookup or right not in test_lookup:
            continue
        lp, rp = probabilities[test_lookup[left]], probabilities[test_lookup[right]]
        ly, ry = data["targets"][left], data["targets"][right]
        own = _bce(lp, ly).mean() + _bce(rp, ry).mean()
        swapped = _bce(lp, ry).mean() + _bce(rp, ly).mean()
        pair_scores.append(1.0 if own < swapped else 0.5 if own == swapped else 0.0)
    rollout_values = []
    rollout_positive = []
    if split_suite == "task_disjoint":
        for sequence in data["sequences"]:
            if not all(index in test_lookup for index in sequence):
                continue
            if arm == "fixed_v21":
                rollout = v20.rollout_probabilities(
                    model, "intervention_modular_v20", sequence, data
                )
            else:
                rollout = candidate_rollout(model, sequence, data)
            sequence_target = data["targets"][sequence]
            rollout_values.append(float(_bce(rollout, sequence_target).mean()))
            rollout_positive.extend(
                (-np.log(np.clip(rollout[sequence_target == 1], 1e-7, 1.0))).tolist()
            )
    tool_families = [
        __import__("wmagentattack.hard_label_confirmation", fromlist=["tool_family"]).tool_family(
            data["rows"][index]["model_input"]["normalized_action"]["tool_id"]
        ) for index in test_indices
    ]
    query = np.asarray([family == "query_read" for family in tool_families])
    return {
        "arm": arm,
        "split_suite": split_suite,
        "split_name": split_name,
        "fold_marker": fold,
        "seed": seed,
        "training_rows": len(train_indices),
        "confirmation_rows": len(test_indices),
        "observed_training_labels": int(observed.sum()),
        "unseen_labels": int((~observed).sum()),
        "task_macro_bce": _macro(per_row_bce, test_indices, data["rows"]),
        "positive_task_macro_nll": _macro(positive_nll, test_indices, data["rows"]),
        "positive_task_macro_recall": _macro(positive_recall, test_indices, data["rows"]),
        "seen_positive_nll": float((-np.log(np.clip(probabilities[seen], 1e-7, 1.0))).mean()) if seen.any() else None,
        "seen_positive_recall": float((probabilities[seen] >= 0.5).mean()) if seen.any() else None,
        "unseen_positive_nll": float((-np.log(np.clip(probabilities[unseen], 1e-7, 1.0))).mean()) if unseen.any() else None,
        "unseen_positive_recall": float((probabilities[unseen] >= 0.5).mean()) if unseen.any() else None,
        "unseen_positive_occurrences": int(unseen.sum()),
        "query_read_positive_recall": float((probabilities[query][target[query] == 1] >= 0.5).mean()) if query.any() else None,
        "execution_brier": float(np.mean((execution_probability - data["execution"][test_indices]) ** 2)),
        "pair_assignment_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "pair_comparisons": len(pair_scores),
        "rollout_bce": float(np.mean(rollout_values)) if rollout_values else None,
        "rollout_positive_nll": float(np.mean(rollout_positive)) if rollout_positive else None,
        "parameter_count": trainable_parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v22 protocol is not frozen before results")
    if sha256(args.dataset) != protocol["data"]["sha256"]:
        raise ValueError("v22 hard-view hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cfg = protocol["open_vocabulary_gate"]["model_comparison"]
    base = augment_arrays(
        v20.arrays(dataset, int(cfg["state_hash_dimension"]), int(cfg["action_hash_dimension"])),
        dataset["effect_token_vocabulary"], cfg,
    )
    torch.set_num_threads(int(cfg["torch_threads"]))
    suites = (
        ("task_disjoint", dataset["split_manifest"]["task_disjoint"], cfg["task_training_seeds"], True),
        ("tool_family_heldout", dataset["split_manifest"]["tool_family_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
        ("source_heldout", dataset["split_manifest"]["source_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
    )
    runs = []
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = v21.materialize_split(
                base, split, fold, sequences_enabled=sequences_enabled
            )
            for arm in ARMS:
                for seed in seeds:
                    run_cfg = copy.deepcopy(cfg)
                    if not sequences_enabled:
                        run_cfg["sequence_weight"] = 0.0
                    if arm == "fixed_v21":
                        (model, history), _ = v21.train_alias(
                            "intervention_no_execution_experts_v21", fold, int(seed), data,
                            {**run_cfg, "hidden_sizes": {"intervention_modular_v20": int(cfg["hidden_size"])}},
                        )
                        observed = data["targets"][train_indices].sum(axis=0) > 0
                    else:
                        model, history, observed = train_candidate(
                            arm, fold, int(seed), data, run_cfg
                        )
                    metrics = evaluate(
                        model, arm, split_suite, split_name, fold, int(seed), data,
                        train_indices, test_indices, dataset["effect_token_vocabulary"], observed,
                    )
                    metrics["history"] = history
                    runs.append(metrics)
    expected = int(protocol["open_vocabulary_gate"]["fixed_budget"]["model_fits"])
    if len(runs) != expected:
        raise RuntimeError(f"incomplete v22 budget: {len(runs)} != {expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "run_metrics.json", {
        "schema_version": "wmagentattack.open_vocabulary_effect.v22",
        "dataset_sha256": sha256(args.dataset),
        "expected_runs": expected,
        "completed_runs": len(runs),
        "runtime_failures": 0,
        "arms": list(ARMS),
        "runs": runs,
    })
    print(json.dumps({"completed_runs": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
