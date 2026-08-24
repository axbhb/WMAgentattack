"""Train the frozen v24 relation-factorized semantic effect model."""

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

from wmagentattack.hard_label_confirmation import tool_family
from wmagentattack.intervention_modular_world_model import (
    assert_transition_only,
    trainable_parameter_count,
)
from wmagentattack.pretrained_semantic_effect_world_model import (
    PretrainedSemanticEffectTransition,
    semantic_hard_negative_loss,
)
from wmagentattack.relation_factorized_semantic_world_model import (
    calibration_label_mask,
    select_support_set_rule,
    similarity_distribution_loss,
    support_fused_probabilities,
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v20 = load("v20_train_v24", ROOT / "scripts" / "243_train_intervention_models_v20.py")
v21 = load("v21_train_v24", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")
v22 = load("v22_train_v24", ROOT / "scripts" / "248_train_open_vocabulary_v22.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int, fold: int, offset: int = 0) -> None:
    value = seed * 1009 + fold + offset
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def augment(base: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    cache = np.load(cache_path)
    data = dict(base)
    data["action_semantics"] = cache["action_features"].astype(np.float32)
    data["label_features"] = cache["label_features"].astype(np.float32)
    data["relation_kernel"] = cache["relation_kernel"].astype(np.float32)
    if len(data["rows"]) != len(data["action_semantics"]):
        raise ValueError("v24 action cache is not row-aligned")
    if data["targets"].shape[1] != len(data["label_features"]):
        raise ValueError("v24 label cache is not vocabulary-aligned")
    if data["relation_kernel"].shape != (
        data["targets"].shape[1], data["targets"].shape[1]
    ):
        raise ValueError("v24 relation kernel is not vocabulary-square")
    return data


def train_relation(
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    fit_labels: np.ndarray,
    cfg: dict[str, Any],
    epochs: int,
    offset: int,
):
    set_seed(seed, fold, offset)
    model = PretrainedSemanticEffectTransition(
        state_size=data["states"].shape[1],
        action_size=data["actions"].shape[1],
        semantic_size=data["label_features"].shape[1],
        hidden_size=int(cfg["hidden_size"]),
    )
    assert_transition_only(model)
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    semantics = torch.tensor(data["action_semantics"])
    labels = torch.tensor(data["label_features"])
    relation = torch.tensor(data["relation_kernel"])
    targets = torch.tensor(data["targets"])
    execution = torch.tensor(data["execution"])
    train = torch.tensor(train_indices, dtype=torch.long)
    fit = torch.tensor(fit_labels, dtype=torch.bool)
    task_counts = defaultdict(int)
    for index in train_indices:
        task_counts[data["rows"][index]["task_id"]] += 1
    row_weights = torch.tensor([
        1.0 / task_counts[data["rows"][index]["task_id"]] for index in train_indices
    ])
    positives = targets[train].sum(0)
    negatives = len(train_indices) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 10.0)
    positive_execution = execution[train].sum()
    execution_pos_weight = torch.clamp(
        (len(train_indices) - positive_execution)
        / torch.clamp(positive_execution, min=1.0),
        1.0,
        10.0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_set = set(train_indices.tolist())
    train_sequences = [
        sequence for sequence in data["sequences"]
        if all(index in train_set for index in sequence)
    ]
    train_pairs = [
        (left, right) for left, right in data["pairs"]
        if left in train_set and right in train_set
    ]
    lookup = {value: local for local, value in enumerate(train_indices)}
    history = []
    for epoch in range(epochs):
        model.train()
        logits, execution_logits = model(
            states[train], actions[train], semantics[train], labels
        )
        per = F.binary_cross_entropy_with_logits(
            logits[:, fit],
            targets[train][:, fit],
            reduction="none",
            pos_weight=pos_weight[fit],
        ).mean(1)
        effect = (per * row_weights).sum() / row_weights.sum()
        hard = semantic_hard_negative_loss(
            logits,
            targets[train],
            fit,
            labels,
            int(cfg["hard_negatives_per_positive"]),
            float(cfg["ranking_margin"]),
        )
        distribution = similarity_distribution_loss(
            logits,
            targets[train],
            fit,
            relation,
            float(cfg["distribution_temperature"]),
        )
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits, execution[train], pos_weight=execution_pos_weight
        )
        recurrent = torch.zeros(())
        if train_sequences:
            values = []
            for sequence in train_sequences:
                hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
                for index in sequence:
                    hidden, _ = model.advance_with_execution(
                        hidden,
                        actions[index : index + 1],
                        semantics[index : index + 1],
                    )
                    values.append(F.binary_cross_entropy_with_logits(
                        model.predict_hidden(hidden, labels)[:, fit],
                        targets[index : index + 1, fit],
                        pos_weight=pos_weight[fit],
                    ))
            recurrent = torch.stack(values).mean()
        paired = torch.zeros(())
        if train_pairs:
            probabilities = torch.sigmoid(logits[:, fit])
            paired = torch.stack([
                F.mse_loss(
                    probabilities[lookup[left]] - probabilities[lookup[right]],
                    targets[left, fit] - targets[right, fit],
                )
                for left, right in train_pairs
            ]).mean()
        total = (
            effect
            + float(cfg["ranking_weight"]) * hard
            + float(cfg["distribution_weight"]) * distribution
            + float(cfg["execution_weight"]) * execution_loss
            + float(cfg["sequence_weight"]) * recurrent
            + float(cfg["pair_weight"]) * paired
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["gradient_clip"])
        )
        optimizer.step()
        if epoch in {0, epochs - 1}:
            history.append({
                "epoch": epoch,
                "total": float(total.detach()),
                "effect": float(effect.detach()),
                "ranking": float(hard.detach()),
                "distribution": float(distribution.detach()),
                "execution": float(execution_loss.detach()),
                "sequence": float(recurrent.detach()),
                "pair": float(paired.detach()),
                "fit_labels": int(fit.sum()),
            })
    return model, history


def semantic_logits(model, indices: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            torch.tensor(data["states"][indices]),
            torch.tensor(data["actions"][indices]),
            torch.tensor(data["action_semantics"][indices]),
            torch.tensor(data["label_features"]),
        )
    return logits.numpy()


def semantic_rollout_logits(
    model, sequence: list[int], data: dict[str, Any]
) -> np.ndarray:
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    semantics = torch.tensor(data["action_semantics"])
    labels = torch.tensor(data["label_features"])
    rows = []
    with torch.no_grad():
        hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
        for index in sequence:
            hidden, _ = model.advance_with_execution(
                hidden,
                actions[index : index + 1],
                semantics[index : index + 1],
            )
            rows.append(model.predict_hidden(hidden, labels)[0].numpy())
    return np.stack(rows)


def evaluate_hybrid(
    fixed_model,
    relation_model,
    arm: str,
    split_suite: str,
    split_name: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    observed: np.ndarray,
    support_weight: float,
    decision_threshold: float,
    support_top_k: int,
    selection: dict[str, Any],
) -> dict[str, Any]:
    fixed_model.eval()
    with torch.no_grad():
        fixed_logits, execution_logits = fixed_model(
            torch.tensor(data["states"][test_indices]),
            torch.tensor(data["actions"][test_indices]),
        )
    fixed_probability = torch.sigmoid(fixed_logits).numpy()
    relation_logits = semantic_logits(relation_model, test_indices, data)
    relation_probability = support_fused_probabilities(
        relation_logits,
        observed,
        ~observed,
        data["relation_kernel"],
        support_weight,
        support_top_k,
    )
    probabilities = np.where(
        observed[None, :], fixed_probability, relation_probability
    )
    threshold_vector = np.where(observed, 0.5, decision_threshold)
    predicted = probabilities >= threshold_vector[None, :]
    execution_probability = torch.sigmoid(execution_logits).numpy()
    target = data["targets"][test_indices]
    positives = target == 1
    unseen_mask = ~observed[None, :]
    unseen = positives & unseen_mask
    unseen_negative = (target == 0) & unseen_mask
    seen = positives & observed[None, :]
    per_row_bce = v22._bce(probabilities, target).mean(1)
    positive_nll = np.asarray([
        float(-np.log(np.clip(probabilities[i][positives[i]], 1e-7, 1.0)).mean())
        for i in range(len(test_indices))
    ])
    positive_recall = np.asarray([
        float(predicted[i][positives[i]].mean()) for i in range(len(test_indices))
    ])
    lookup = {value: local for local, value in enumerate(test_indices)}
    pair_scores = []
    for left, right in data["pairs"]:
        if left not in lookup or right not in lookup:
            continue
        lp, rp = probabilities[lookup[left]], probabilities[lookup[right]]
        ly, ry = data["targets"][left], data["targets"][right]
        own = v22._bce(lp, ly).mean() + v22._bce(rp, ry).mean()
        swapped = v22._bce(lp, ry).mean() + v22._bce(rp, ly).mean()
        pair_scores.append(1.0 if own < swapped else 0.5 if own == swapped else 0.0)
    rollout_bce, rollout_positive = [], []
    if split_suite == "task_disjoint":
        for sequence in data["sequences"]:
            if not all(index in lookup for index in sequence):
                continue
            fixed_rollout = v20.rollout_probabilities(
                fixed_model, "intervention_modular_v20", sequence, data
            )
            relation_rollout = support_fused_probabilities(
                semantic_rollout_logits(relation_model, sequence, data),
                observed,
                ~observed,
                data["relation_kernel"],
                support_weight,
                support_top_k,
            )
            rollout = np.where(
                observed[None, :], fixed_rollout, relation_rollout
            )
            sequence_target = data["targets"][sequence]
            rollout_bce.append(float(v22._bce(rollout, sequence_target).mean()))
            rollout_positive.extend(
                (-np.log(np.clip(
                    rollout[sequence_target == 1], 1e-7, 1.0
                ))).tolist()
            )
    families = [
        tool_family(
            data["rows"][index]["model_input"]["normalized_action"]["tool_id"]
        )
        for index in test_indices
    ]
    query = np.asarray([family == "query_read" for family in families])
    unseen_predicted = predicted & unseen_mask
    unseen_tp = int((unseen_predicted & positives).sum())
    unseen_fp = int((unseen_predicted & ~positives).sum())
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
        "task_macro_bce": v22._macro(
            per_row_bce, test_indices, data["rows"]
        ),
        "positive_task_macro_nll": v22._macro(
            positive_nll, test_indices, data["rows"]
        ),
        "positive_task_macro_recall": v22._macro(
            positive_recall, test_indices, data["rows"]
        ),
        "seen_positive_nll": (
            float(-np.log(np.clip(probabilities[seen], 1e-7, 1.0)).mean())
            if seen.any() else None
        ),
        "seen_positive_recall": (
            float(predicted[seen].mean()) if seen.any() else None
        ),
        "unseen_positive_nll": (
            float(-np.log(np.clip(probabilities[unseen], 1e-7, 1.0)).mean())
            if unseen.any() else None
        ),
        "unseen_positive_recall": (
            float(predicted[unseen].mean()) if unseen.any() else None
        ),
        "unseen_positive_occurrences": int(unseen.sum()),
        "unseen_false_positive_rate": (
            float(predicted[unseen_negative].mean())
            if unseen_negative.any() else 0.0
        ),
        "unseen_precision": float(unseen_tp / max(unseen_tp + unseen_fp, 1)),
        "mean_predicted_unseen_set_size": float(
            unseen_predicted.sum(1).mean()
        ),
        "mean_true_unseen_set_size": float(unseen.sum(1).mean()),
        "decision_threshold": float(decision_threshold),
        "support_weight": float(support_weight),
        "query_read_positive_recall": (
            float(predicted[query][target[query] == 1].mean())
            if query.any() else None
        ),
        "execution_brier": float(np.mean(
            (execution_probability - data["execution"][test_indices]) ** 2
        )),
        "pair_assignment_accuracy": (
            float(np.mean(pair_scores)) if pair_scores else None
        ),
        "pair_comparisons": len(pair_scores),
        "rollout_bce": (
            float(np.mean(rollout_bce)) if rollout_bce else None
        ),
        "rollout_positive_nll": (
            float(np.mean(rollout_positive)) if rollout_positive else None
        ),
        "parameter_count": (
            trainable_parameter_count(fixed_model)
            + trainable_parameter_count(relation_model)
        ),
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v24 protocol is not frozen")
    if sha256(args.dataset) != protocol["data"]["sha256"]:
        raise ValueError("v24 dataset hash mismatch")
    cache_expected = protocol["semantic_cache"].get("cache_sha256")
    if cache_expected and sha256(args.semantic_cache) != cache_expected:
        raise ValueError("v24 semantic cache hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    base = augment(
        v20.arrays(
            dataset,
            int(cfg["state_hash_dimension"]),
            int(cfg["action_hash_dimension"]),
        ),
        args.semantic_cache,
    )
    torch.set_num_threads(int(cfg["torch_threads"]))
    suites = (
        (
            "task_disjoint",
            dataset["split_manifest"]["task_disjoint"],
            cfg["task_training_seeds"],
            True,
        ),
        (
            "tool_family_heldout",
            dataset["split_manifest"]["tool_family_heldout_diagnostic"],
            cfg["diagnostic_training_seeds"],
            False,
        ),
        (
            "source_heldout",
            dataset["split_manifest"]["source_heldout_diagnostic"],
            cfg["diagnostic_training_seeds"],
            False,
        ),
    )
    calibration_partition = calibration_label_mask(
        dataset["effect_token_vocabulary"],
        int(cfg["calibration_label_modulus"]),
    )
    runs, fit_count = [], 0
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = v21.materialize_split(
                base, split, fold, sequences_enabled=sequences_enabled
            )
            run_cfg = copy.deepcopy(cfg)
            if not sequences_enabled:
                run_cfg["sequence_weight"] = 0.0
            observed = data["targets"][train_indices].sum(0) > 0
            heldout = observed & calibration_partition
            fitted = observed & ~calibration_partition
            if heldout.sum() < int(cfg["minimum_calibration_labels"]):
                raise ValueError(
                    f"insufficient v24 calibration labels for {split_suite}/{split_name}"
                )
            for seed in seeds:
                inner, inner_history = train_relation(
                    fold,
                    int(seed),
                    data,
                    train_indices,
                    fitted,
                    run_cfg,
                    int(cfg["inner_epochs"]),
                    300000,
                )
                fit_count += 1
                inner_logits = semantic_logits(inner, train_indices, data)
                selection = select_support_set_rule(
                    inner_logits,
                    data["targets"][train_indices],
                    fitted,
                    heldout,
                    data["relation_kernel"],
                    cfg["support_weights"],
                    cfg["decision_thresholds"],
                    int(cfg["support_top_k"]),
                    float(cfg["selection_maximum_false_positive_rate"]),
                    float(cfg["selection_maximum_set_size_multiplier"]),
                    float(cfg["selection_set_size_offset"]),
                )
                relation, relation_history = train_relation(
                    fold,
                    int(seed),
                    data,
                    train_indices,
                    observed,
                    run_cfg,
                    int(cfg["outer_epochs"]),
                    400000,
                )
                fit_count += 1
                (fixed, fixed_history), _ = v21.train_alias(
                    "intervention_no_execution_experts_v21",
                    fold,
                    int(seed),
                    data,
                    {
                        **run_cfg,
                        "epochs": int(cfg["fixed_v21_epochs"]),
                        "hidden_sizes": {
                            "intervention_modular_v20": int(
                                cfg["fixed_v21_hidden_size"]
                            )
                        },
                    },
                )
                fit_count += 1
                fixed_row = v22.evaluate(
                    fixed,
                    "fixed_v21",
                    split_suite,
                    split_name,
                    fold,
                    int(seed),
                    data,
                    train_indices,
                    test_indices,
                    dataset["effect_token_vocabulary"],
                    observed,
                )
                fixed_row["history"] = fixed_history
                runs.append(fixed_row)
                raw = evaluate_hybrid(
                    fixed,
                    relation,
                    "relation_e5_raw_v24",
                    split_suite,
                    split_name,
                    fold,
                    int(seed),
                    data,
                    train_indices,
                    test_indices,
                    observed,
                    0.0,
                    0.5,
                    int(cfg["support_top_k"]),
                    {"mode": "identity"},
                )
                raw["history"] = relation_history
                runs.append(raw)
                candidate = evaluate_hybrid(
                    fixed,
                    relation,
                    "relation_support_set_v24",
                    split_suite,
                    split_name,
                    fold,
                    int(seed),
                    data,
                    train_indices,
                    test_indices,
                    observed,
                    float(selection["support_weight"]),
                    float(selection["threshold"]),
                    int(cfg["support_top_k"]),
                    selection,
                )
                candidate["history"] = relation_history
                candidate["inner_history"] = inner_history
                runs.append(candidate)
    expected_fits = int(protocol["fixed_budget"]["model_fits"])
    expected_rows = int(protocol["fixed_budget"]["metric_rows"])
    if fit_count != expected_fits or len(runs) != expected_rows:
        raise RuntimeError(
            f"incomplete v24 budget: fits={fit_count}, rows={len(runs)}"
        )
    payload = {
        "schema_version": "wmagentattack.relation_factorized_distribution.v24",
        "dataset_sha256": sha256(args.dataset),
        "semantic_cache_sha256": sha256(args.semantic_cache),
        "completed_model_fits": fit_count,
        "completed_metric_rows": len(runs),
        "runtime_failures": 0,
        "arms": [
            "fixed_v21",
            "relation_e5_raw_v24",
            "relation_support_set_v24",
        ],
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model_fits": fit_count,
        "metric_rows": len(runs),
        "runtime_failures": 0,
    }))


if __name__ == "__main__":
    main()
