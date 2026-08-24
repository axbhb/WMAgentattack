"""Train the frozen v28 bound successor-record comparison."""

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

from wmagentattack.bound_successor_world_model import (
    BoundSuccessorRecordTransition,
    candidate_features,
    conflict_signature,
    conflict_vocabulary,
    goal_term_features,
    record_signature,
    record_vocabulary,
    render_effect_probabilities,
)
from wmagentattack.clean_evidence_probe import hashed_text
from wmagentattack.hard_label_confirmation import tool_family
from wmagentattack.hybrid_semantic_world_model import semantic_state_v3_feature_vector
from wmagentattack.intervention_modular_world_model import (
    assert_transition_only,
    trainable_parameter_count,
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v20 = load("v20_train_v28", ROOT / "scripts" / "243_train_intervention_models_v20.py")
v21 = load("v21_train_v28", ROOT / "scripts" / "246_train_hard_label_confirmation_v21.py")
v22 = load("v22_train_v28", ROOT / "scripts" / "248_train_open_vocabulary_v22.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int, fold: int, offset: int) -> None:
    value = seed * 1009 + fold + offset
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def target(row: dict[str, Any]) -> dict[str, Any]:
    return row["model_target"]["structured_successor_delta"]


def prepare(
    hard: dict[str, Any], structured: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    base = v20.arrays(hard, int(cfg["state_hash_dimension"]), int(cfg["action_hash_dimension"]))
    by_ref = {row["transition_ref"]: row for row in structured["confirmation_rows"]}
    if set(by_ref) != {row["transition_ref"] for row in base["rows"]}:
        raise ValueError("v28 structured/hard transition mismatch")
    base["structured_rows"] = [by_ref[row["transition_ref"]] for row in base["rows"]]
    support = structured["support_rows"]
    base["support_rows"] = support
    base["support_states"] = np.stack([
        semantic_state_v3_feature_vector(
            row["model_input"]["current_semantic_state"],
            hash_dimension=int(cfg["state_hash_dimension"]),
        )
        for row in support
    ]).astype(np.float32)
    base["support_actions"] = np.stack([
        hashed_text(
            row["model_input"]["normalized_action"],
            int(cfg["action_hash_dimension"]),
            "v20-normalized-action",
        )
        for row in support
    ]).astype(np.float32)
    return base


def binary_targets(
    rows: list[dict[str, Any]], candidates: list[str], kind: str, *, strict: bool = True
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(candidates)}
    values = np.zeros((len(rows), len(candidates)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        current = target(row)
        items = (
            [record_signature(value) for value in current["added_evidence_records"]]
            if kind == "record"
            else [conflict_signature(value) for value in current["added_conflicts"]]
        )
        for value in items:
            if value not in lookup and strict:
                raise ValueError(f"target absent from frozen {kind} candidates")
            if value in lookup:
                values[row_index, lookup[value]] = 1.0
    return values


def delta_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([target(row)["delta_bits"] for row in rows], dtype=np.float32)


def execution_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([target(row)["execution_status"] == "error" for row in rows], dtype=np.float32)


def pointer_targets(row: dict[str, Any]) -> np.ndarray:
    size = len(row["model_input"]["current_semantic_state"].get("goal", {}).get("fact_terms", ()))
    values = np.zeros(size, dtype=np.float32)
    values[target(row)["newly_matched_goal_term_indices"]] = 1.0
    return values


def positive_weight(values: Tensor, maximum: float) -> Tensor:
    positives = values.sum(0)
    negatives = len(values) - positives
    return torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, maximum)


def row_pointer_loss(
    model: BoundSuccessorRecordTransition,
    hidden: torch.Tensor,
    rows: list[dict[str, Any]],
    term_features: list[torch.Tensor],
    pointer_pos_weight: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for index, row in enumerate(rows):
        logits = model.pointer_logits(hidden[index : index + 1], term_features[index])
        labels = torch.tensor(pointer_targets(row))
        losses.append(F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pointer_pos_weight))
    return torch.stack(losses).mean() if losses else hidden.new_zeros(())


def train_bound(
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[BoundSuccessorRecordTransition, dict[str, Any], list[dict[str, float]]]:
    set_seed(seed, fold, 420000)
    train_rows = [data["structured_rows"][index] for index in train_indices]
    support_rows = data["support_rows"]
    candidate_rows = train_rows + support_rows
    records = record_vocabulary(candidate_rows)
    conflicts = conflict_vocabulary(candidate_rows)
    heldout_rows = [
        row for index, row in enumerate(data["structured_rows"]) if index not in set(train_indices.tolist())
    ]
    heldout_records = {
        record_signature(record)
        for row in heldout_rows
        for record in target(row)["added_evidence_records"]
    }
    coverage = len(heldout_records & set(records)) / max(1, len(heldout_records))

    record_feature = torch.tensor(candidate_features(records, int(cfg["record_feature_dimension"]), "v28-bound-record"))
    conflict_feature = torch.tensor(candidate_features(conflicts, int(cfg["conflict_feature_dimension"]), "v28-conflict"))
    train_term_features = [torch.tensor(goal_term_features(row, int(cfg["pointer_feature_dimension"]))) for row in train_rows]
    support_term_features = [torch.tensor(goal_term_features(row, int(cfg["pointer_feature_dimension"]))) for row in support_rows]

    states = torch.tensor(data["states"][train_indices])
    actions = torch.tensor(data["actions"][train_indices])
    support_states = torch.tensor(data["support_states"])
    support_actions = torch.tensor(data["support_actions"])
    train_record = torch.tensor(binary_targets(train_rows, records, "record"))
    support_record = torch.tensor(binary_targets(support_rows, records, "record"))
    train_conflict = torch.tensor(binary_targets(train_rows, conflicts, "conflict"))
    support_conflict = torch.tensor(binary_targets(support_rows, conflicts, "conflict"))
    train_delta = torch.tensor(delta_targets(train_rows))
    support_delta = torch.tensor(delta_targets(support_rows))
    train_execution = torch.tensor(execution_targets(train_rows))
    support_execution = torch.tensor(execution_targets(support_rows))
    record_pos_weight = positive_weight(torch.cat((train_record, support_record)), 12.0)
    conflict_pos_weight = positive_weight(
        torch.cat((train_conflict, support_conflict)), 12.0
    ) if conflicts else torch.zeros(0)
    execution_positive = torch.cat((train_execution, support_execution)).sum()
    execution_pos_weight = torch.clamp(
        (len(train_execution) + len(support_execution) - execution_positive)
        / torch.clamp(execution_positive, min=1.0),
        1.0,
        12.0,
    )
    pointer_values = np.concatenate([
        pointer_targets(row) for row in candidate_rows
    ])
    pointer_positive = float(pointer_values.sum())
    pointer_pos_weight = torch.tensor(
        min(max((len(pointer_values) - pointer_positive) / max(pointer_positive, 1.0), 1.0), 20.0)
    )

    model = BoundSuccessorRecordTransition(
        states.shape[1], actions.shape[1], int(cfg["hidden_size"]),
        int(cfg["record_feature_dimension"]), int(cfg["pointer_feature_dimension"]),
        int(cfg["conflict_feature_dimension"]),
    )
    assert_transition_only(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    global_to_local = {value: index for index, value in enumerate(train_indices)}
    train_sequences = [
        sequence for sequence in data["sequences"] if all(index in global_to_local for index in sequence)
    ]
    train_pairs = [
        (left, right) for left, right in data["pairs"]
        if left in global_to_local and right in global_to_local
    ]
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        record_logits, delta_logits, conflict_logits, execution_logits = model(
            states, actions, record_feature, conflict_feature
        )
        record_loss = F.binary_cross_entropy_with_logits(
            record_logits, train_record, pos_weight=record_pos_weight
        )
        delta_loss = F.binary_cross_entropy_with_logits(delta_logits, train_delta)
        conflict_loss = (
            F.binary_cross_entropy_with_logits(
                conflict_logits, train_conflict, pos_weight=conflict_pos_weight
            ) if conflicts else torch.zeros(())
        )
        execution_loss = F.binary_cross_entropy_with_logits(
            execution_logits, train_execution, pos_weight=execution_pos_weight
        )
        hidden, _ = model.advance_with_execution(model.initial_hidden(states), actions)
        pointer_loss = row_pointer_loss(
            model, hidden, train_rows, train_term_features, pointer_pos_weight
        )

        support_record_logits, support_delta_logits, support_conflict_logits, support_execution_logits = model(
            support_states, support_actions, record_feature, conflict_feature
        )
        support_hidden, _ = model.advance_with_execution(
            model.initial_hidden(support_states), support_actions
        )
        support_loss = (
            F.binary_cross_entropy_with_logits(
                support_record_logits, support_record, pos_weight=record_pos_weight
            )
            + float(cfg["delta_weight"]) * F.binary_cross_entropy_with_logits(
                support_delta_logits, support_delta
            )
            + float(cfg["execution_weight"]) * F.binary_cross_entropy_with_logits(
                support_execution_logits, support_execution
            )
            + float(cfg["pointer_weight"]) * row_pointer_loss(
                model, support_hidden, support_rows, support_term_features, pointer_pos_weight
            )
        )
        if conflicts:
            support_loss = support_loss + float(cfg["conflict_weight"]) * F.binary_cross_entropy_with_logits(
                support_conflict_logits, support_conflict, pos_weight=conflict_pos_weight
            )

        recurrent_losses = []
        for sequence in train_sequences:
            first = global_to_local[sequence[0]]
            recurrent_hidden = model.initial_hidden(states[first : first + 1])
            for global_index in sequence:
                local = global_to_local[global_index]
                recurrent_hidden, _ = model.advance_with_execution(
                    recurrent_hidden, actions[local : local + 1]
                )
                recurrent_record, recurrent_delta, _ = model.predict_hidden(
                    recurrent_hidden, record_feature, conflict_feature
                )
                recurrent_pointer = model.pointer_logits(
                    recurrent_hidden, train_term_features[local]
                )
                recurrent_losses.append(
                    F.binary_cross_entropy_with_logits(
                        recurrent_record, train_record[local : local + 1], pos_weight=record_pos_weight
                    )
                    + float(cfg["delta_weight"]) * F.binary_cross_entropy_with_logits(
                        recurrent_delta, train_delta[local : local + 1]
                    )
                    + float(cfg["pointer_weight"]) * F.binary_cross_entropy_with_logits(
                        recurrent_pointer, torch.tensor(pointer_targets(train_rows[local])),
                        pos_weight=pointer_pos_weight,
                    )
                )
        recurrent = torch.stack(recurrent_losses).mean() if recurrent_losses else torch.zeros(())
        paired_losses = []
        record_probability = torch.sigmoid(record_logits)
        delta_probability = torch.sigmoid(delta_logits)
        for left, right in train_pairs:
            li, ri = global_to_local[left], global_to_local[right]
            paired_losses.append(
                F.mse_loss(
                    record_probability[li] - record_probability[ri],
                    train_record[li] - train_record[ri],
                )
                + F.mse_loss(
                    delta_probability[li] - delta_probability[ri],
                    train_delta[li] - train_delta[ri],
                )
            )
        paired = torch.stack(paired_losses).mean() if paired_losses else torch.zeros(())
        total = (
            record_loss
            + float(cfg["delta_weight"]) * delta_loss
            + float(cfg["conflict_weight"]) * conflict_loss
            + float(cfg["execution_weight"]) * execution_loss
            + float(cfg["pointer_weight"]) * pointer_loss
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
                "epoch": epoch, "total": float(total.detach()),
                "record": float(record_loss.detach()), "delta": float(delta_loss.detach()),
                "conflict": float(conflict_loss.detach()), "execution": float(execution_loss.detach()),
                "pointer": float(pointer_loss.detach()), "support": float(support_loss.detach()),
                "sequence": float(recurrent.detach()), "pair": float(paired.detach()),
            })
    metadata = {
        "record_candidates": records,
        "conflict_candidates": conflicts,
        "record_features": record_feature,
        "conflict_features": conflict_feature,
        "heldout_exact_record_candidate_coverage": coverage,
    }
    return model, metadata, history


def predict_structured(
    model: BoundSuccessorRecordTransition,
    metadata: dict[str, Any],
    data: dict[str, Any],
    indices: list[int] | np.ndarray,
    cfg: dict[str, Any],
    *,
    recurrent_hidden: torch.Tensor | None = None,
) -> dict[str, Any]:
    indices = list(indices)
    record_rows, delta_rows, conflict_rows, execution_rows, pointer_rows = [], [], [], [], []
    hidden = recurrent_hidden
    with torch.no_grad():
        for index in indices:
            state = torch.tensor(data["states"][index : index + 1])
            action = torch.tensor(data["actions"][index : index + 1])
            if hidden is None:
                current = model.initial_hidden(state)
            else:
                current = hidden
            hidden, execution = model.advance_with_execution(current, action)
            records, delta, conflicts = model.predict_hidden(
                hidden, metadata["record_features"], metadata["conflict_features"]
            )
            terms = torch.tensor(goal_term_features(
                data["structured_rows"][index], int(cfg["pointer_feature_dimension"])
            ))
            pointers = model.pointer_logits(hidden, terms)
            record_rows.append(torch.sigmoid(records)[0].numpy())
            delta_rows.append(torch.sigmoid(delta)[0].numpy())
            conflict_rows.append(torch.sigmoid(conflicts)[0].numpy())
            execution_rows.append(float(torch.sigmoid(execution)[0]))
            pointer_rows.append(torch.sigmoid(pointers).numpy())
            if recurrent_hidden is None:
                hidden = None
    return {
        "records": np.stack(record_rows),
        "deltas": np.stack(delta_rows),
        "conflicts": np.stack(conflict_rows) if metadata["conflict_candidates"] else np.zeros((len(indices), 0)),
        "execution": np.asarray(execution_rows),
        "pointers": pointer_rows,
    }


def canonical_probabilities(
    prediction: dict[str, Any], metadata: dict[str, Any], vocabulary: list[str]
) -> np.ndarray:
    return render_effect_probabilities(
        prediction["records"], prediction["pointers"], prediction["deltas"],
        prediction["execution"], prediction["conflicts"], metadata["record_candidates"],
        metadata["conflict_candidates"], vocabulary,
    )


def set_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    target_values = truth == 1
    tp = int(np.logical_and(prediction, target_values).sum())
    fp = int(np.logical_and(prediction, ~target_values).sum())
    fn = int(np.logical_and(~prediction, target_values).sum())
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "exact_set_accuracy": float(np.mean(np.all(prediction == target_values, axis=1))),
    }


def evaluate(
    fixed_model,
    bound_model,
    metadata: dict[str, Any] | None,
    arm: str,
    split_suite: str,
    split_name: str,
    fold: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    vocabulary: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    fixed_model.eval()
    with torch.no_grad():
        fixed_logits, fixed_execution = fixed_model(
            torch.tensor(data["states"][test_indices]), torch.tensor(data["actions"][test_indices])
        )
    fixed_probability = torch.sigmoid(fixed_logits).numpy()
    observed = data["targets"][train_indices].sum(0) > 0
    structured_prediction = None
    if bound_model is None:
        probabilities = fixed_probability
    else:
        structured_prediction = predict_structured(bound_model, metadata, data, test_indices, cfg)
        open_probability = canonical_probabilities(structured_prediction, metadata, vocabulary)
        probabilities = np.where(observed[None, :], fixed_probability, open_probability)
    target_values = data["targets"][test_indices]
    positives = target_values == 1
    unseen = positives & ~observed[None, :]
    seen = positives & observed[None, :]
    unseen_negative = (target_values == 0) & ~observed[None, :]
    unseen_prediction = (probabilities >= 0.5) & ~observed[None, :]
    unseen_tp = int((unseen_prediction & unseen).sum())
    unseen_fp = int((unseen_prediction & unseen_negative).sum())
    per_row_bce = v22._bce(probabilities, target_values).mean(1)
    positive_nll = np.asarray([
        float(-np.log(np.clip(probabilities[i][positives[i]], 1e-7, 1.0)).mean())
        for i in range(len(test_indices))
    ])
    positive_recall = np.asarray([
        float((probabilities[i][positives[i]] >= 0.5).mean())
        for i in range(len(test_indices))
    ])
    count3_token = np.asarray([token == "matched_count=3" for token in vocabulary])
    count3 = unseen & count3_token[None, :]
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
            if bound_model is None:
                rollout = fixed_rollout
            else:
                prediction = predict_structured(
                    bound_model, metadata, data, sequence, cfg,
                    recurrent_hidden=bound_model.initial_hidden(torch.tensor(data["states"][sequence[0]:sequence[0]+1])),
                )
                open_rollout = canonical_probabilities(prediction, metadata, vocabulary)
                rollout = np.where(observed[None, :], fixed_rollout, open_rollout)
            sequence_target = data["targets"][sequence]
            rollout_bce.append(float(v22._bce(rollout, sequence_target).mean()))
            rollout_positive.extend((-np.log(np.clip(rollout[sequence_target == 1], 1e-7, 1.0))).tolist())
    query = np.asarray([
        tool_family(data["rows"][index]["model_input"]["normalized_action"]["tool_id"]) == "query_read"
        for index in test_indices
    ])
    row = {
        "arm": arm, "split_suite": split_suite, "split_name": split_name,
        "fold_marker": fold, "seed": seed, "training_rows": len(train_indices),
        "confirmation_rows": len(test_indices), "observed_training_labels": int(observed.sum()),
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
        "query_read_positive_recall": float((probabilities[query][target_values[query] == 1] >= 0.5).mean()) if query.any() else None,
        "execution_brier": float(np.mean((torch.sigmoid(fixed_execution).numpy() - data["execution"][test_indices]) ** 2)),
        "pair_assignment_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "pair_comparisons": len(pair_scores),
        "rollout_bce": float(np.mean(rollout_bce)) if rollout_bce else None,
        "rollout_positive_nll": float(np.mean(rollout_positive)) if rollout_positive else None,
        "parameter_count": trainable_parameter_count(fixed_model) + (trainable_parameter_count(bound_model) if bound_model else 0),
    }
    if structured_prediction is not None:
        test_rows = [data["structured_rows"][index] for index in test_indices]
        record_truth = binary_targets(
            test_rows, metadata["record_candidates"], "record", strict=False
        )
        record_metric = set_metrics(structured_prediction["records"], record_truth)
        pointer_prob = np.concatenate(structured_prediction["pointers"])
        pointer_truth = np.concatenate([pointer_targets(row) for row in test_rows])
        pointer_prediction = pointer_prob >= 0.5
        pointer_tp = int(np.logical_and(pointer_prediction, pointer_truth == 1).sum())
        pointer_fp = int(np.logical_and(pointer_prediction, pointer_truth == 0).sum())
        pointer_fn = int(np.logical_and(~pointer_prediction, pointer_truth == 1).sum())
        row.update({
            "bound_record_precision": record_metric["precision"],
            "bound_record_recall": record_metric["recall"],
            "bound_record_f1": record_metric["f1"],
            "bound_record_exact_set_accuracy": record_metric["exact_set_accuracy"],
            "goal_pointer_f1": 2 * pointer_tp / max(1, 2 * pointer_tp + pointer_fp + pointer_fn),
            "goal_pointer_precision": pointer_tp / max(1, pointer_tp + pointer_fp),
            "goal_pointer_recall": pointer_tp / max(1, pointer_tp + pointer_fn),
            "heldout_exact_record_candidate_coverage": metadata["heldout_exact_record_candidate_coverage"],
            "record_candidate_count": len(metadata["record_candidates"]),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hard-dataset", type=Path, required=True)
    parser.add_argument("--structured-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v28 protocol is not frozen")
    if sha256(args.hard_dataset) != protocol["data"]["hard_dataset_sha256"]:
        raise ValueError("v28 hard dataset hash mismatch")
    if sha256(args.structured_dataset) != protocol["data"]["structured_dataset_sha256"]:
        raise ValueError("v28 structured dataset hash mismatch")
    hard = json.loads(args.hard_dataset.read_text(encoding="utf-8"))
    structured = json.loads(args.structured_dataset.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    base = prepare(hard, structured, cfg)
    torch.set_num_threads(int(cfg["torch_threads"]))
    suites = (
        ("task_disjoint", hard["split_manifest"]["task_disjoint"], cfg["task_training_seeds"], True),
        ("tool_family_heldout", hard["split_manifest"]["tool_family_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
        ("source_heldout", hard["split_manifest"]["source_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
    )
    runs, fits = [], 0
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = v21.materialize_split(
                base, split, fold, sequences_enabled=sequences_enabled
            )
            data["structured_rows"] = base["structured_rows"]
            data["support_rows"] = base["support_rows"]
            data["support_states"] = base["support_states"]
            data["support_actions"] = base["support_actions"]
            run_cfg = copy.deepcopy(cfg)
            if not sequences_enabled:
                run_cfg["sequence_weight"] = 0.0
            for seed in seeds:
                (fixed, fixed_history), _ = v21.train_alias(
                    "intervention_no_execution_experts_v21", fold, int(seed), data,
                    {
                        **run_cfg,
                        "epochs": int(cfg["fixed_v21_epochs"]),
                        "hidden_sizes": {"intervention_modular_v20": int(cfg["fixed_v21_hidden_size"])},
                    },
                )
                fits += 1
                fixed_row = evaluate(
                    fixed, None, None, "fixed_v21", split_suite, split_name, fold, int(seed),
                    data, train_indices, test_indices, hard["effect_token_vocabulary"], run_cfg,
                )
                fixed_row["history"] = fixed_history
                runs.append(fixed_row)
                bound, metadata, history = train_bound(
                    fold, int(seed), data, train_indices, run_cfg
                )
                fits += 1
                bound_row = evaluate(
                    fixed, bound, metadata, "bound_successor_records_v28", split_suite, split_name,
                    fold, int(seed), data, train_indices, test_indices,
                    hard["effect_token_vocabulary"], run_cfg,
                )
                bound_row["history"] = history
                runs.append(bound_row)
    if fits != int(protocol["fixed_budget"]["model_fits"]):
        raise RuntimeError(f"incomplete v28 fit budget: {fits}")
    if len(runs) != int(protocol["fixed_budget"]["metric_rows"]):
        raise RuntimeError(f"incomplete v28 metric budget: {len(runs)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wmagentattack.bound_successor_records.v28",
        "hard_dataset_sha256": sha256(args.hard_dataset),
        "structured_dataset_sha256": sha256(args.structured_dataset),
        "completed_model_fits": fits, "completed_metric_rows": len(runs),
        "runtime_failures": 0, "arms": ["fixed_v21", "bound_successor_records_v28"],
        "runs": runs,
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"model_fits": fits, "metric_rows": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
