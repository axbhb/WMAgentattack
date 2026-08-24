"""Train the frozen v30 joint record--goal relational successor comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.bound_successor_world_model import (
    candidate_features,
    conflict_signature,
    conflict_vocabulary,
    goal_term_features,
    record_signature,
    render_effect_probabilities,
)
from wmagentattack.intervention_modular_world_model import assert_transition_only, trainable_parameter_count
from wmagentattack.joint_relational_world_model import (
    JointRelationalSuccessorTransition,
    global_pointer_probabilities,
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v28 = load("v28_train_v30", ROOT / "scripts" / "267_train_bound_successor_records_v28.py")
v20, v21, v22 = v28.v20, v28.v21, v28.v22


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int, fold: int) -> None:
    value = seed * 1009 + fold + 430000
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def typed_target(row: dict[str, Any]) -> dict[str, Any]:
    return row["model_target"]["structured_successor_delta"]


def relation_target(row: dict[str, Any]) -> dict[str, Any]:
    return row["model_target"]["relational_successor_delta"]


def prepare(hard: dict[str, Any], structured: dict[str, Any], relational: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    base = v28.prepare(hard, structured, cfg)
    by_ref = {row["transition_ref"]: row for row in relational["confirmation_rows"]}
    if set(by_ref) != {row["transition_ref"] for row in base["rows"]}:
        raise ValueError("v30 relational/hard transition mismatch")
    base["relation_rows"] = [by_ref[row["transition_ref"]] for row in base["rows"]]
    support_by_ref = {row["support_ref"]: row for row in relational["support_rows"]}
    base["relation_support_rows"] = [support_by_ref[row["support_ref"]] for row in base["support_rows"]]
    base["static_record_candidates"] = relational["static_record_candidates"]
    base["static_candidates_by_tool"] = relational["static_candidates_by_tool"]
    return base


def bare_tool(row: dict[str, Any]) -> str:
    return str(row["model_input"]["normalized_action"]["tool_id"]).rsplit("::", 1)[-1]


def allowed_indices(row: dict[str, Any], candidates: list[str], by_tool: dict[str, list[str]]) -> list[int]:
    lookup = {value: index for index, value in enumerate(candidates)}
    values = by_tool[bare_tool(row)]
    if any(value not in lookup for value in values):
        raise ValueError("tool candidate absent from global static vocabulary")
    return [lookup[value] for value in values]


def row_targets(row: dict[str, Any], candidates: list[str], allowed: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local = {candidates[index]: position for position, index in enumerate(allowed)}
    goal_count = len(row["model_input"]["current_semantic_state"].get("goal", {}).get("fact_terms", ()))
    records = torch.zeros(len(allowed), dtype=torch.float32)
    relations = torch.zeros((len(allowed), goal_count), dtype=torch.float32)
    pointers = torch.zeros(goal_count, dtype=torch.float32)
    target = relation_target(row)
    pointers[target["newly_matched_goal_term_indices"]] = 1.0
    for record in target["added_evidence_records"]:
        signature = record_signature(record)
        if signature not in local:
            raise ValueError(f"target record absent from action-conditioned candidates: {bare_tool(row)}")
        index = local[signature]
        records[index] = 1.0
        relations[index, record["newly_matched_goal_term_indices"]] = 1.0
    return records, relations, pointers


def scalar_pos_weight(labels: list[torch.Tensor], maximum: float) -> torch.Tensor:
    flat = torch.cat([value.reshape(-1) for value in labels])
    positive = flat.sum()
    negative = len(flat) - positive
    return torch.clamp(negative / torch.clamp(positive, min=1.0), 1.0, maximum)


def conflict_targets(rows: list[dict[str, Any]], vocabulary: list[str]) -> torch.Tensor:
    lookup = {value: index for index, value in enumerate(vocabulary)}
    values = torch.zeros((len(rows), len(vocabulary)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for conflict in typed_target(row)["added_conflicts"]:
            signature = conflict_signature(conflict)
            if signature in lookup:
                values[row_index, lookup[signature]] = 1.0
    return values


def delta_targets(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([typed_target(row)["delta_bits"] for row in rows], dtype=torch.float32)


def execution_targets(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([typed_target(row)["execution_status"] == "error" for row in rows], dtype=torch.float32)


def relational_losses(
    model: JointRelationalSuccessorTransition,
    hidden: torch.Tensor,
    record_logits: torch.Tensor,
    rows: list[dict[str, Any]],
    candidates: list[str],
    by_tool: dict[str, list[str]],
    record_features: torch.Tensor,
    goal_dimension: int,
    record_pos_weight: torch.Tensor,
    relation_pos_weight: torch.Tensor,
    pointer_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    record_losses, relation_losses, pointer_losses = [], [], []
    for row_index, row in enumerate(rows):
        allowed = allowed_indices(row, candidates, by_tool)
        index = torch.tensor(allowed, dtype=torch.long)
        record_y, relation_y, pointer_y = row_targets(row, candidates, allowed)
        selected_record_logits = record_logits[row_index, index]
        terms = torch.tensor(goal_term_features(row, goal_dimension), dtype=torch.float32)
        relation_logits = model.relation_logits(hidden[row_index : row_index + 1], record_features[index], terms)
        pointer_probability = global_pointer_probabilities(selected_record_logits, relation_logits)
        record_losses.append(F.binary_cross_entropy_with_logits(selected_record_logits, record_y, pos_weight=record_pos_weight))
        relation_losses.append(F.binary_cross_entropy_with_logits(relation_logits, relation_y, pos_weight=relation_pos_weight))
        pointer_losses.append(F.binary_cross_entropy(pointer_probability.clamp(1e-7, 1 - 1e-7), pointer_y, weight=torch.where(pointer_y > 0, pointer_pos_weight, 1.0)))
    zero = hidden.new_zeros(())
    return (
        torch.stack(record_losses).mean() if record_losses else zero,
        torch.stack(relation_losses).mean() if relation_losses else zero,
        torch.stack(pointer_losses).mean() if pointer_losses else zero,
    )


def train_joint(
    fold: int, seed: int, data: dict[str, Any], train_indices: np.ndarray, cfg: dict[str, Any]
) -> tuple[JointRelationalSuccessorTransition, dict[str, Any], list[dict[str, float]]]:
    set_seed(seed, fold)
    train_typed = [data["structured_rows"][index] for index in train_indices]
    train_relation = [data["relation_rows"][index] for index in train_indices]
    support_typed = data["support_rows"]
    support_relation = data["relation_support_rows"]
    candidates = data["static_record_candidates"]
    by_tool = data["static_candidates_by_tool"]
    conflicts = conflict_vocabulary(train_typed + support_typed)
    record_features = torch.tensor(candidate_features(candidates, int(cfg["record_feature_dimension"]), "v30-static-record"))
    conflict_features = torch.tensor(candidate_features(conflicts, int(cfg["conflict_feature_dimension"]), "v30-conflict"))

    all_relation = train_relation + support_relation
    record_labels, relation_labels, pointer_labels = [], [], []
    for row in all_relation:
        allowed = allowed_indices(row, candidates, by_tool)
        r, e, p = row_targets(row, candidates, allowed)
        record_labels.append(r); relation_labels.append(e); pointer_labels.append(p)
    record_pos = scalar_pos_weight(record_labels, float(cfg["maximum_record_pos_weight"]))
    relation_pos = scalar_pos_weight(relation_labels, float(cfg["maximum_relation_pos_weight"]))
    pointer_pos = scalar_pos_weight(pointer_labels, float(cfg["maximum_pointer_pos_weight"]))

    states = torch.tensor(data["states"][train_indices])
    actions = torch.tensor(data["actions"][train_indices])
    support_states = torch.tensor(data["support_states"])
    support_actions = torch.tensor(data["support_actions"])
    train_delta, support_delta = delta_targets(train_typed), delta_targets(support_typed)
    train_execution, support_execution = execution_targets(train_typed), execution_targets(support_typed)
    train_conflict, support_conflict = conflict_targets(train_typed, conflicts), conflict_targets(support_typed, conflicts)

    model = JointRelationalSuccessorTransition(
        states.shape[1], actions.shape[1], int(cfg["hidden_size"]),
        int(cfg["record_feature_dimension"]), int(cfg["goal_feature_dimension"]),
        int(cfg["conflict_feature_dimension"]),
    )
    assert_transition_only(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    global_to_local = {value: index for index, value in enumerate(train_indices)}
    sequences = [value for value in data["sequences"] if all(index in global_to_local for index in value)]
    pairs = [(left, right) for left, right in data["pairs"] if left in global_to_local and right in global_to_local]
    history = []

    for epoch in range(int(cfg["epochs"])):
        hidden, record_logits, delta_logits, conflict_logits, execution_logits = model(states, actions, record_features, conflict_features)
        rec, rel, ptr = relational_losses(
            model, hidden, record_logits, train_relation, candidates, by_tool, record_features,
            int(cfg["goal_feature_dimension"]), record_pos, relation_pos, pointer_pos,
        )
        delta = F.binary_cross_entropy_with_logits(delta_logits, train_delta)
        execution = F.binary_cross_entropy_with_logits(execution_logits, train_execution)
        conflict = F.binary_cross_entropy_with_logits(conflict_logits, train_conflict) if conflicts else hidden.new_zeros(())

        sh, sr, sd, sc, se = model(support_states, support_actions, record_features, conflict_features)
        srec, srel, sptr = relational_losses(
            model, sh, sr, support_relation, candidates, by_tool, record_features,
            int(cfg["goal_feature_dimension"]), record_pos, relation_pos, pointer_pos,
        )
        support = srec + float(cfg["relation_weight"]) * srel + float(cfg["pointer_weight"]) * sptr
        support = support + float(cfg["delta_weight"]) * F.binary_cross_entropy_with_logits(sd, support_delta)
        support = support + float(cfg["execution_weight"]) * F.binary_cross_entropy_with_logits(se, support_execution)
        if conflicts:
            support = support + float(cfg["conflict_weight"]) * F.binary_cross_entropy_with_logits(sc, support_conflict)

        recurrent_parts = []
        for sequence in sequences:
            first = global_to_local[sequence[0]]
            recurrent_hidden = model.initial_hidden(states[first : first + 1])
            for global_index in sequence:
                local = global_to_local[global_index]
                recurrent_hidden, _ = model.advance_with_execution(recurrent_hidden, actions[local : local + 1])
                rr, rd, _ = model.predict_hidden(recurrent_hidden, record_features, conflict_features)
                rrec, rrel, rptr = relational_losses(
                    model, recurrent_hidden, rr, [train_relation[local]], candidates, by_tool,
                    record_features, int(cfg["goal_feature_dimension"]), record_pos, relation_pos, pointer_pos,
                )
                recurrent_parts.append(rrec + float(cfg["relation_weight"]) * rrel + float(cfg["pointer_weight"]) * rptr + float(cfg["delta_weight"]) * F.binary_cross_entropy_with_logits(rd, train_delta[local : local + 1]))
        recurrent = torch.stack(recurrent_parts).mean() if recurrent_parts else hidden.new_zeros(())

        pair_parts = []
        probability = torch.sigmoid(record_logits)
        delta_probability = torch.sigmoid(delta_logits)
        for left, right in pairs:
            li, ri = global_to_local[left], global_to_local[right]
            left_allowed = allowed_indices(train_relation[li], candidates, by_tool)
            right_allowed = allowed_indices(train_relation[ri], candidates, by_tool)
            common = sorted(set(left_allowed) & set(right_allowed))
            if common:
                index = torch.tensor(common, dtype=torch.long)
                left_y = torch.zeros(len(common)); right_y = torch.zeros(len(common))
                for position, candidate_index in enumerate(common):
                    signature = candidates[candidate_index]
                    left_y[position] = any(record_signature(value) == signature for value in relation_target(train_relation[li])["added_evidence_records"])
                    right_y[position] = any(record_signature(value) == signature for value in relation_target(train_relation[ri])["added_evidence_records"])
                pair_parts.append(F.mse_loss(probability[li, index] - probability[ri, index], left_y - right_y))
            pair_parts.append(F.mse_loss(delta_probability[li] - delta_probability[ri], train_delta[li] - train_delta[ri]))
        pair = torch.stack(pair_parts).mean() if pair_parts else hidden.new_zeros(())

        total = rec + float(cfg["relation_weight"]) * rel + float(cfg["pointer_weight"]) * ptr
        total = total + float(cfg["delta_weight"]) * delta + float(cfg["execution_weight"]) * execution + float(cfg["conflict_weight"]) * conflict
        total = total + float(cfg["support_weight"]) * support + float(cfg["sequence_weight"]) * recurrent + float(cfg["pair_weight"]) * pair
        optimizer.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
        optimizer.step()
        if epoch in {0, int(cfg["epochs"]) - 1}:
            history.append({
                "epoch": epoch, "total": float(total.detach()), "record": float(rec.detach()),
                "relation": float(rel.detach()), "pointer": float(ptr.detach()), "delta": float(delta.detach()),
                "execution": float(execution.detach()), "support": float(support.detach()),
                "recurrent": float(recurrent.detach()), "pair": float(pair.detach()),
            })
    return model, {
        "record_candidates": candidates, "record_features": record_features,
        "candidates_by_tool": by_tool, "conflict_candidates": conflicts,
        "conflict_features": conflict_features,
    }, history


def predict_joint(
    model: JointRelationalSuccessorTransition, metadata: dict[str, Any], data: dict[str, Any],
    indices: list[int] | np.ndarray, cfg: dict[str, Any], *, recurrent: bool = False,
) -> dict[str, Any]:
    indices = list(indices)
    record_rows, delta_rows, conflict_rows, execution_rows, pointer_rows = [], [], [], [], []
    relation_probability_rows, relation_truth_rows, record_truth_rows = [], [], []
    hidden_state = None
    with torch.no_grad():
        for index_value in indices:
            state = torch.tensor(data["states"][index_value : index_value + 1])
            action = torch.tensor(data["actions"][index_value : index_value + 1])
            current = model.initial_hidden(state) if hidden_state is None else hidden_state
            hidden_state, execution = model.advance_with_execution(current, action)
            records, delta, conflicts = model.predict_hidden(hidden_state, metadata["record_features"], metadata["conflict_features"])
            row = data["relation_rows"][index_value]
            allowed = allowed_indices(row, metadata["record_candidates"], metadata["candidates_by_tool"])
            allowed_tensor = torch.tensor(allowed, dtype=torch.long)
            terms = torch.tensor(goal_term_features(row, int(cfg["goal_feature_dimension"])))
            relation_logits = model.relation_logits(hidden_state, metadata["record_features"][allowed_tensor], terms)
            selected_record_logits = records[0, allowed_tensor]
            pointer = global_pointer_probabilities(selected_record_logits, relation_logits)
            full_records = np.zeros(len(metadata["record_candidates"]), dtype=np.float32)
            full_records[allowed] = torch.sigmoid(selected_record_logits).numpy()
            record_y, relation_y, _ = row_targets(row, metadata["record_candidates"], allowed)
            record_rows.append(full_records); delta_rows.append(torch.sigmoid(delta)[0].numpy())
            conflict_rows.append(torch.sigmoid(conflicts)[0].numpy() if len(metadata["conflict_candidates"]) else np.zeros(0))
            execution_rows.append(float(torch.sigmoid(execution)[0])); pointer_rows.append(pointer.numpy())
            relation_probability_rows.append(torch.sigmoid(relation_logits).numpy())
            relation_truth_rows.append(relation_y.numpy()); record_truth_rows.append(record_y.numpy())
            if not recurrent:
                hidden_state = None
    return {
        "records": np.stack(record_rows), "deltas": np.stack(delta_rows),
        "conflicts": np.stack(conflict_rows) if metadata["conflict_candidates"] else np.zeros((len(indices), 0)),
        "execution": np.asarray(execution_rows), "pointers": pointer_rows,
        "relation_probabilities": relation_probability_rows, "relation_truth": relation_truth_rows,
        "record_truth": record_truth_rows,
    }


def binary_metrics(probabilities: list[np.ndarray], truths: list[np.ndarray]) -> dict[str, float]:
    p = np.concatenate([value.reshape(-1) for value in probabilities])
    y = np.concatenate([value.reshape(-1) for value in truths]).astype(bool)
    prediction = p >= 0.5
    tp = int(np.logical_and(prediction, y).sum()); fp = int(np.logical_and(prediction, ~y).sum()); fn = int(np.logical_and(~prediction, y).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def evaluate(
    fixed_model, joint_model, metadata: dict[str, Any] | None, arm: str, split_suite: str,
    split_name: str, fold: int, seed: int, data: dict[str, Any], train_indices: np.ndarray,
    test_indices: np.ndarray, vocabulary: list[str], cfg: dict[str, Any],
) -> dict[str, Any]:
    fixed_model.eval()
    with torch.no_grad():
        fixed_logits, fixed_execution = fixed_model(torch.tensor(data["states"][test_indices]), torch.tensor(data["actions"][test_indices]))
    fixed_probability = torch.sigmoid(fixed_logits).numpy()
    observed = data["targets"][train_indices].sum(0) > 0
    prediction = None
    if joint_model is None:
        probabilities = fixed_probability
    else:
        joint_model.eval(); prediction = predict_joint(joint_model, metadata, data, test_indices, cfg)
        open_probability = render_effect_probabilities(
            prediction["records"], prediction["pointers"], prediction["deltas"], prediction["execution"],
            prediction["conflicts"], metadata["record_candidates"], metadata["conflict_candidates"], vocabulary,
        )
        probabilities = np.where(observed[None, :], fixed_probability, open_probability)
    targets = data["targets"][test_indices]
    positives = targets == 1; unseen = positives & ~observed[None, :]; seen = positives & observed[None, :]
    unseen_negative = (targets == 0) & ~observed[None, :]; unseen_prediction = (probabilities >= 0.5) & ~observed[None, :]
    unseen_tp = int((unseen_prediction & unseen).sum()); unseen_fp = int((unseen_prediction & unseen_negative).sum())
    row_bce = v22._bce(probabilities, targets).mean(1)
    positive_nll = np.asarray([float(-np.log(np.clip(probabilities[i][positives[i]], 1e-7, 1.0)).mean()) for i in range(len(test_indices))])
    positive_recall = np.asarray([float((probabilities[i][positives[i]] >= 0.5).mean()) for i in range(len(test_indices))])
    count3_mask = np.asarray([token == "matched_count=3" for token in vocabulary])[None, :]
    count3 = unseen & count3_mask; focused = unseen & ~np.asarray([token.startswith("matched_count=") for token in vocabulary])[None, :]
    lookup = {value: local for local, value in enumerate(test_indices)}
    pair_scores = []
    for left, right in data["pairs"]:
        if left not in lookup or right not in lookup: continue
        lp, rp = probabilities[lookup[left]], probabilities[lookup[right]]; ly, ry = data["targets"][left], data["targets"][right]
        own = v22._bce(lp, ly).mean() + v22._bce(rp, ry).mean(); swapped = v22._bce(lp, ry).mean() + v22._bce(rp, ly).mean()
        pair_scores.append(1.0 if own < swapped else 0.5 if own == swapped else 0.0)
    rollout_bce, rollout_positive = [], []
    if split_suite == "task_disjoint":
        for sequence in data["sequences"]:
            if not all(index in lookup for index in sequence): continue
            fixed_rollout = v20.rollout_probabilities(fixed_model, "intervention_modular_v20", sequence, data)
            if joint_model is None:
                rollout = fixed_rollout
            else:
                joint_rollout = predict_joint(joint_model, metadata, data, sequence, cfg, recurrent=True)
                open_rollout = render_effect_probabilities(
                    joint_rollout["records"], joint_rollout["pointers"], joint_rollout["deltas"], joint_rollout["execution"],
                    joint_rollout["conflicts"], metadata["record_candidates"], metadata["conflict_candidates"], vocabulary,
                )
                rollout = np.where(observed[None, :], fixed_rollout, open_rollout)
            sequence_target = data["targets"][sequence]
            rollout_bce.append(float(v22._bce(rollout, sequence_target).mean()))
            rollout_positive.extend((-np.log(np.clip(rollout[sequence_target == 1], 1e-7, 1.0))).tolist())
    query = np.asarray([v28.tool_family(data["rows"][index]["model_input"]["normalized_action"]["tool_id"]) == "query_read" for index in test_indices])
    row = {
        "arm": arm, "split_suite": split_suite, "split_name": split_name, "fold_marker": fold, "seed": seed,
        "training_rows": len(train_indices), "confirmation_rows": len(test_indices),
        "observed_training_labels": int(observed.sum()), "unseen_labels": int((~observed).sum()),
        "task_macro_bce": v22._macro(row_bce, test_indices, data["rows"]),
        "positive_task_macro_nll": v22._macro(positive_nll, test_indices, data["rows"]),
        "positive_task_macro_recall": v22._macro(positive_recall, test_indices, data["rows"]),
        "seen_positive_nll": float(-np.log(np.clip(probabilities[seen], 1e-7, 1.0)).mean()) if seen.any() else None,
        "seen_positive_recall": float((probabilities[seen] >= 0.5).mean()) if seen.any() else None,
        "unseen_positive_nll": float(-np.log(np.clip(probabilities[unseen], 1e-7, 1.0)).mean()) if unseen.any() else None,
        "unseen_positive_recall": float((probabilities[unseen] >= 0.5).mean()) if unseen.any() else None,
        "unseen_positive_occurrences": int(unseen.sum()), "unseen_false_positive_rate": unseen_fp / max(1, int(unseen_negative.sum())),
        "unseen_precision": unseen_tp / max(1, unseen_tp + unseen_fp), "unseen_false_positives": unseen_fp,
        "matched_count3_recall": float((probabilities[count3] >= 0.5).mean()) if count3.any() else None,
        "matched_count3_occurrences": int(count3.sum()),
        "focused_unseen_recall": float((probabilities[focused] >= 0.5).mean()) if focused.any() else None,
        "focused_unseen_occurrences": int(focused.sum()),
        "query_read_positive_recall": float((probabilities[query][targets[query] == 1] >= 0.5).mean()) if query.any() else None,
        "execution_brier": float(np.mean((torch.sigmoid(fixed_execution).numpy() - data["execution"][test_indices]) ** 2)),
        "pair_assignment_accuracy": float(np.mean(pair_scores)) if pair_scores else None, "pair_comparisons": len(pair_scores),
        "rollout_bce": float(np.mean(rollout_bce)) if rollout_bce else None,
        "rollout_positive_nll": float(np.mean(rollout_positive)) if rollout_positive else None,
        "parameter_count": trainable_parameter_count(fixed_model) + (trainable_parameter_count(joint_model) if joint_model else 0),
    }
    if prediction is not None:
        record_metric = binary_metrics([value for value in [prediction["records"][i, allowed_indices(data["relation_rows"][index], metadata["record_candidates"], metadata["candidates_by_tool"])] for i, index in enumerate(test_indices)]], prediction["record_truth"])
        relation_metric = binary_metrics(prediction["relation_probabilities"], prediction["relation_truth"])
        pointer_p = np.concatenate(prediction["pointers"]); pointer_y = np.concatenate([row_targets(data["relation_rows"][index], metadata["record_candidates"], allowed_indices(data["relation_rows"][index], metadata["record_candidates"], metadata["candidates_by_tool"]))[2].numpy() for index in test_indices])
        pointer_metric = binary_metrics([pointer_p], [pointer_y])
        exact = [float(np.array_equal(prob >= 0.5, truth == 1)) for prob, truth in zip([prediction["records"][i, allowed_indices(data["relation_rows"][index], metadata["record_candidates"], metadata["candidates_by_tool"])] for i, index in enumerate(test_indices)], prediction["record_truth"])]
        row.update({
            "bound_record_precision": record_metric["precision"], "bound_record_recall": record_metric["recall"],
            "bound_record_f1": record_metric["f1"], "bound_record_exact_set_accuracy": float(np.mean(exact)),
            "record_goal_relation_precision": relation_metric["precision"], "record_goal_relation_recall": relation_metric["recall"],
            "record_goal_relation_f1": relation_metric["f1"], "goal_pointer_f1": pointer_metric["f1"],
            "goal_pointer_precision": pointer_metric["precision"], "goal_pointer_recall": pointer_metric["recall"],
            "heldout_exact_record_candidate_coverage": 1.0, "record_candidate_count": len(metadata["record_candidates"]),
        })
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--hard-dataset", type=Path, required=True)
    p.add_argument("--structured-dataset", type=Path, required=True); p.add_argument("--relational-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results": raise ValueError("v30 protocol is not frozen")
    for name, path in (("hard_dataset", a.hard_dataset), ("structured_dataset", a.structured_dataset), ("relational_dataset", a.relational_dataset)):
        if sha256(path) != protocol["data"][name + "_sha256"]: raise ValueError(f"v30 {name} hash mismatch")
    hard = json.loads(a.hard_dataset.read_text()); structured = json.loads(a.structured_dataset.read_text()); relational = json.loads(a.relational_dataset.read_text())
    cfg = protocol["model_comparison"]; base = prepare(hard, structured, relational, cfg); torch.set_num_threads(int(cfg["torch_threads"]))
    suites = (
        ("task_disjoint", hard["split_manifest"]["task_disjoint"], cfg["task_training_seeds"], True),
        ("tool_family_heldout", hard["split_manifest"]["tool_family_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
        ("source_heldout", hard["split_manifest"]["source_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
    )
    runs, fits = [], 0
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = v21.materialize_split(base, split, fold, sequences_enabled=sequences_enabled)
            for key in ("structured_rows", "relation_rows", "support_rows", "relation_support_rows", "support_states", "support_actions", "static_record_candidates", "static_candidates_by_tool"):
                data[key] = base[key]
            run_cfg = copy.deepcopy(cfg)
            if not sequences_enabled: run_cfg["sequence_weight"] = 0.0
            for seed in seeds:
                (fixed, fixed_history), _ = v21.train_alias("intervention_no_execution_experts_v21", fold, int(seed), data, {**run_cfg, "epochs": int(cfg["fixed_v21_epochs"]), "hidden_sizes": {"intervention_modular_v20": int(cfg["fixed_v21_hidden_size"])}})
                fits += 1
                fixed_row = evaluate(fixed, None, None, "fixed_v21", split_suite, split_name, fold, int(seed), data, train_indices, test_indices, hard["effect_token_vocabulary"], run_cfg); fixed_row["history"] = fixed_history; runs.append(fixed_row)
                model, metadata, history = train_joint(fold, int(seed), data, train_indices, run_cfg); fits += 1
                joint_row = evaluate(fixed, model, metadata, "joint_relational_successor_v30", split_suite, split_name, fold, int(seed), data, train_indices, test_indices, hard["effect_token_vocabulary"], run_cfg); joint_row["history"] = history; runs.append(joint_row)
    if fits != int(protocol["fixed_budget"]["model_fits"]): raise RuntimeError(f"incomplete v30 fit budget: {fits}")
    if len(runs) != int(protocol["fixed_budget"]["metric_rows"]): raise RuntimeError(f"incomplete v30 metric budget: {len(runs)}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "wmagentattack.joint_relational_successor.v30", "hard_dataset_sha256": sha256(a.hard_dataset), "structured_dataset_sha256": sha256(a.structured_dataset), "relational_dataset_sha256": sha256(a.relational_dataset), "completed_model_fits": fits, "completed_metric_rows": len(runs), "runtime_failures": 0, "arms": ["fixed_v21", "joint_relational_successor_v30"], "runs": runs}
    (a.output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_fits": fits, "metric_rows": len(runs), "runtime_failures": 0}))


if __name__ == "__main__": main()
