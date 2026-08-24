"""Run the frozen v26 support-conditioned compositional effect comparison."""

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

from wmagentattack.clean_evidence_probe import hashed_text
from wmagentattack.hard_label_confirmation import tool_family
from wmagentattack.hybrid_semantic_world_model import semantic_state_v3_feature_vector
from wmagentattack.intervention_modular_world_model import assert_transition_only, trainable_parameter_count
from wmagentattack.support_conditioned_effect_world_model import (
    SupportConditionedEffectTransition,
    atom_target_matrix,
    atom_vocabulary,
    compose_effect_probabilities,
    matched_count_targets,
    ordinal_nll,
    support_atom_target_matrix,
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v20 = load("v20_train_v26", ROOT / "scripts" / "243_train_intervention_models_v20.py")
v21 = load("v21_train_v26", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")
v22 = load("v22_train_v26", ROOT / "scripts" / "248_train_open_vocabulary_v22.py")


ARMS = ("fixed_v21", "factorized_ordinal_no_support_v26", "factorized_ordinal_support_v26")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int, fold: int, offset: int) -> None:
    value = seed * 1009 + fold + offset
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def prepare(
    hard_dataset: dict[str, Any], support_dataset: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    base = v20.arrays(
        hard_dataset, int(cfg["state_hash_dimension"]), int(cfg["action_hash_dimension"])
    )
    vocabulary = hard_dataset["effect_token_vocabulary"]
    atoms = atom_vocabulary(vocabulary, support_dataset["atom_vocabulary"])
    effects = [row["model_target"]["effect_tokens"] for row in hard_dataset["transitions"]]
    base["atom_vocabulary"] = atoms
    base["atom_targets"] = atom_target_matrix(effects, atoms)
    base["count_targets"] = matched_count_targets(effects)
    support_rows = support_dataset["support_rows"]
    base["support_states"] = np.stack([
        semantic_state_v3_feature_vector(
            row["model_input"]["current_semantic_state"],
            hash_dimension=int(cfg["state_hash_dimension"]),
        )
        for row in support_rows
    ]).astype(np.float32)
    base["support_actions"] = np.stack([
        hashed_text(
            row["model_input"]["normalized_action"],
            int(cfg["action_hash_dimension"]),
            "v20-normalized-action",
        )
        for row in support_rows
    ]).astype(np.float32)
    base["support_atom_targets"] = support_atom_target_matrix(support_rows, atoms)
    base["support_execution"] = np.asarray(
        [row["model_target"]["execution_error"] for row in support_rows], dtype=np.float32
    )
    base["support_rows"] = support_rows
    return base


def _weighted_bce(logits, targets, weights, pos_weight):
    per = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    ).mean(1)
    return (per * weights).sum() / weights.sum()


def train_atom_model(
    arm: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    cfg: dict[str, Any],
):
    use_support = arm == "factorized_ordinal_support_v26"
    set_seed(seed, fold, 300000 if use_support else 250000)
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    atoms = torch.tensor(data["atom_targets"])
    counts = torch.tensor(data["count_targets"], dtype=torch.long)
    execution = torch.tensor(data["execution"])
    train = torch.tensor(train_indices, dtype=torch.long)
    support_states = torch.tensor(data["support_states"])
    support_actions = torch.tensor(data["support_actions"])
    support_atoms = torch.tensor(data["support_atom_targets"])
    support_execution = torch.tensor(data["support_execution"])

    task_counts = defaultdict(int)
    for index in train_indices:
        task_counts[data["rows"][index]["task_id"]] += 1
    row_weights = torch.tensor([
        1.0 / task_counts[data["rows"][index]["task_id"]] for index in train_indices
    ])
    combined = torch.cat((atoms[train], support_atoms), dim=0) if use_support else atoms[train]
    positives = combined.sum(0)
    negatives = len(combined) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 10.0)
    positive_execution = execution[train].sum()
    execution_pos_weight = torch.clamp(
        (len(train_indices) - positive_execution) / torch.clamp(positive_execution, min=1.0),
        1.0,
        10.0,
    )
    model = SupportConditionedEffectTransition(
        states.shape[1], actions.shape[1], int(cfg["hidden_size"]), atoms.shape[1]
    )
    assert_transition_only(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    train_set = set(train_indices.tolist())
    sequences = [sequence for sequence in data["sequences"] if all(index in train_set for index in sequence)]
    pairs = [(left, right) for left, right in data["pairs"] if left in train_set and right in train_set]
    lookup = {value: local for local, value in enumerate(train_indices)}
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        atom_logits, count_probabilities, execution_logits = model(states[train], actions[train])
        hard_atom = _weighted_bce(atom_logits, atoms[train], row_weights, pos_weight)
        count_loss = ordinal_nll(count_probabilities, counts[train])
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits, execution[train], pos_weight=execution_pos_weight
        )
        support_loss = torch.zeros(())
        if use_support:
            support_logits, _, support_execution_logits = model(support_states, support_actions)
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits, support_atoms, pos_weight=pos_weight
            ) + float(cfg["execution_weight"]) * F.binary_cross_entropy_with_logits(
                support_execution_logits, support_execution
            )
        recurrent = torch.zeros(())
        if sequences:
            values = []
            for sequence in sequences:
                hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
                for index in sequence:
                    hidden, _ = model.advance_with_execution(hidden, actions[index : index + 1])
                    recurrent_atoms, recurrent_counts = model.predict_hidden(hidden)
                    values.append(
                        F.binary_cross_entropy_with_logits(
                            recurrent_atoms, atoms[index : index + 1], pos_weight=pos_weight
                        )
                        + float(cfg["ordinal_weight"]) * ordinal_nll(
                            recurrent_counts, counts[index : index + 1]
                        )
                    )
            recurrent = torch.stack(values).mean()
        paired = torch.zeros(())
        if pairs:
            probabilities = torch.sigmoid(atom_logits)
            paired = torch.stack([
                F.mse_loss(
                    probabilities[lookup[left]] - probabilities[lookup[right]],
                    atoms[left] - atoms[right],
                )
                for left, right in pairs
            ]).mean()
        total = (
            hard_atom
            + float(cfg["ordinal_weight"]) * count_loss
            + float(cfg["execution_weight"]) * execution_loss
            + float(cfg["support_weight"]) * support_loss
            + float(cfg["sequence_weight"]) * recurrent
            + float(cfg["pair_weight"]) * paired
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
        optimizer.step()
        if epoch in {0, int(cfg["epochs"]) - 1}:
            history.append({
                "epoch": epoch,
                "total": float(total.detach()),
                "hard_atom": float(hard_atom.detach()),
                "ordinal": float(count_loss.detach()),
                "execution": float(execution_loss.detach()),
                "support": float(support_loss.detach()),
                "sequence": float(recurrent.detach()),
                "pair": float(paired.detach()),
                "uses_support": use_support,
            })
    return model, history


def candidate_probability(model, indices: np.ndarray, data: dict[str, Any], vocabulary: list[str]):
    model.eval()
    with torch.no_grad():
        atom_logits, count_probabilities, _ = model(
            torch.tensor(data["states"][indices]), torch.tensor(data["actions"][indices])
        )
    return compose_effect_probabilities(
        torch.sigmoid(atom_logits).numpy(), count_probabilities.numpy(), vocabulary, data["atom_vocabulary"]
    )


def candidate_rollout(model, sequence: list[int], data: dict[str, Any], vocabulary: list[str]):
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    atom_rows, count_rows = [], []
    with torch.no_grad():
        hidden = model.initial_hidden(states[sequence[0] : sequence[0] + 1])
        for index in sequence:
            hidden, _ = model.advance_with_execution(hidden, actions[index : index + 1])
            atom_logits, count_probabilities = model.predict_hidden(hidden)
            atom_rows.append(torch.sigmoid(atom_logits)[0].numpy())
            count_rows.append(count_probabilities[0].numpy())
    return compose_effect_probabilities(
        np.stack(atom_rows), np.stack(count_rows), vocabulary, data["atom_vocabulary"]
    )


def evaluate(
    fixed_model,
    candidate_model,
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
    fixed_model.eval()
    with torch.no_grad():
        fixed_logits, execution_logits = fixed_model(
            torch.tensor(data["states"][test_indices]), torch.tensor(data["actions"][test_indices])
        )
    fixed_probability = torch.sigmoid(fixed_logits).numpy()
    if candidate_model is None:
        probabilities = fixed_probability
    else:
        open_probability = candidate_probability(candidate_model, test_indices, data, vocabulary)
        probabilities = np.where(observed[None, :], fixed_probability, open_probability)
    execution_probability = torch.sigmoid(execution_logits).numpy()
    target = data["targets"][test_indices]
    positives = target == 1
    unseen = positives & ~observed[None, :]
    seen = positives & observed[None, :]
    unseen_negative = (target == 0) & ~observed[None, :]
    unseen_prediction = (probabilities >= 0.5) & ~observed[None, :]
    unseen_tp = int((unseen_prediction & unseen).sum())
    unseen_fp = int((unseen_prediction & unseen_negative).sum())
    per_row_bce = v22._bce(probabilities, target).mean(1)
    positive_nll = np.asarray([
        float(-np.log(np.clip(probabilities[i][positives[i]], 1e-7, 1.0)).mean())
        for i in range(len(test_indices))
    ])
    positive_recall = np.asarray([
        float((probabilities[i][positives[i]] >= 0.5).mean()) for i in range(len(test_indices))
    ])
    token_count3 = np.asarray([token == "matched_count=3" for token in vocabulary])
    count3 = unseen & token_count3[None, :]
    focused = unseen & ~np.asarray([token.startswith("matched_count=") for token in vocabulary])[None, :]
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
            if candidate_model is None:
                rollout = fixed_rollout
            else:
                open_rollout = candidate_rollout(candidate_model, sequence, data, vocabulary)
                rollout = np.where(observed[None, :], fixed_rollout, open_rollout)
            sequence_target = data["targets"][sequence]
            rollout_bce.append(float(v22._bce(rollout, sequence_target).mean()))
            rollout_positive.extend(
                (-np.log(np.clip(rollout[sequence_target == 1], 1e-7, 1.0))).tolist()
            )
    query = np.asarray([
        tool_family(data["rows"][index]["model_input"]["normalized_action"]["tool_id"]) == "query_read"
        for index in test_indices
    ])
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
        "task_macro_bce": v22._macro(per_row_bce, test_indices, data["rows"]),
        "positive_task_macro_nll": v22._macro(positive_nll, test_indices, data["rows"]),
        "positive_task_macro_recall": v22._macro(positive_recall, test_indices, data["rows"]),
        "seen_positive_nll": float(-np.log(np.clip(probabilities[seen], 1e-7, 1.0)).mean()) if seen.any() else None,
        "seen_positive_recall": float((probabilities[seen] >= 0.5).mean()) if seen.any() else None,
        "unseen_positive_nll": float(-np.log(np.clip(probabilities[unseen], 1e-7, 1.0)).mean()) if unseen.any() else None,
        "unseen_positive_recall": float((probabilities[unseen] >= 0.5).mean()) if unseen.any() else None,
        "unseen_positive_occurrences": int(unseen.sum()),
        "unseen_false_positive_rate": unseen_fp / max(1, int(unseen_negative.sum())),
        "unseen_precision": unseen_tp / max(1, unseen_tp + unseen_fp),
        "unseen_false_positives": unseen_fp,
        "matched_count3_recall": float((probabilities[count3] >= 0.5).mean()) if count3.any() else None,
        "matched_count3_occurrences": int(count3.sum()),
        "focused_unseen_recall": float((probabilities[focused] >= 0.5).mean()) if focused.any() else None,
        "focused_unseen_occurrences": int(focused.sum()),
        "query_read_positive_recall": float((probabilities[query][target[query] == 1] >= 0.5).mean()) if query.any() else None,
        "execution_brier": float(np.mean((execution_probability - data["execution"][test_indices]) ** 2)),
        "pair_assignment_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "pair_comparisons": len(pair_scores),
        "rollout_bce": float(np.mean(rollout_bce)) if rollout_bce else None,
        "rollout_positive_nll": float(np.mean(rollout_positive)) if rollout_positive else None,
        "parameter_count": trainable_parameter_count(fixed_model)
        + (trainable_parameter_count(candidate_model) if candidate_model is not None else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--support-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v26 protocol is not frozen")
    if sha256(args.dataset) != protocol["data"]["hard_dataset_sha256"]:
        raise ValueError("v26 hard dataset hash mismatch")
    if sha256(args.support_dataset) != protocol["data"]["support_dataset_sha256"]:
        raise ValueError("v26 support dataset hash mismatch")
    hard_dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    support_dataset = json.loads(args.support_dataset.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    base = prepare(hard_dataset, support_dataset, cfg)
    torch.set_num_threads(int(cfg["torch_threads"]))
    suites = (
        ("task_disjoint", hard_dataset["split_manifest"]["task_disjoint"], cfg["task_training_seeds"], True),
        ("tool_family_heldout", hard_dataset["split_manifest"]["tool_family_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
        ("source_heldout", hard_dataset["split_manifest"]["source_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
    )
    runs, fits = [], 0
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = v21.materialize_split(
                base, split, fold, sequences_enabled=sequences_enabled
            )
            run_cfg = copy.deepcopy(cfg)
            if not sequences_enabled:
                run_cfg["sequence_weight"] = 0.0
            observed = data["targets"][train_indices].sum(0) > 0
            for seed in seeds:
                (fixed, fixed_history), _ = v21.train_alias(
                    "intervention_no_execution_experts_v21",
                    fold,
                    int(seed),
                    data,
                    {
                        **run_cfg,
                        "epochs": int(cfg["fixed_v21_epochs"]),
                        "hidden_sizes": {"intervention_modular_v20": int(cfg["fixed_v21_hidden_size"])},
                    },
                )
                fits += 1
                fixed_row = evaluate(
                    fixed, None, "fixed_v21", split_suite, split_name, fold, int(seed), data,
                    train_indices, test_indices, hard_dataset["effect_token_vocabulary"], observed,
                )
                fixed_row["history"] = fixed_history
                runs.append(fixed_row)
                for arm in ARMS[1:]:
                    candidate, history = train_atom_model(
                        arm, fold, int(seed), data, train_indices, run_cfg
                    )
                    fits += 1
                    row = evaluate(
                        fixed, candidate, arm, split_suite, split_name, fold, int(seed), data,
                        train_indices, test_indices, hard_dataset["effect_token_vocabulary"], observed,
                    )
                    row["history"] = history
                    runs.append(row)
    if fits != int(protocol["fixed_budget"]["model_fits"]):
        raise RuntimeError(f"incomplete v26 fit budget: {fits}")
    if len(runs) != int(protocol["fixed_budget"]["metric_rows"]):
        raise RuntimeError(f"incomplete v26 metric budget: {len(runs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wmagentattack.support_conditioned_compositional.v26",
        "hard_dataset_sha256": sha256(args.dataset),
        "support_dataset_sha256": sha256(args.support_dataset),
        "completed_model_fits": fits,
        "completed_metric_rows": len(runs),
        "runtime_failures": 0,
        "arms": list(ARMS),
        "atom_vocabulary_size": len(base["atom_vocabulary"]),
        "runs": runs,
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"model_fits": fits, "metric_rows": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
