"""Train the large v45 state/action/residual model on frozen full AgentDojo data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import OBSERVED_OUTCOME_TARGETS
from wmagentattack.joint_outcome_auxiliary import (
    JOINT_OUTCOME_CLASSES,
    normalized_joint_event_weights,
)
from wmagentattack.large_hybrid_world_model import (
    LargeHybridWorldModel,
    LargeWorldModelConfig,
    parameter_breakdown,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(indices)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def task_weights(events: list[dict], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(events[index]["task_name"]) for index in indices)
    values = np.asarray([1.0 / counts[str(events[index]["task_name"])] for index in indices], dtype=np.float32)
    return values / values.mean()


def build_arrays(dataset: dict, cache: MappingLike) -> dict[str, np.ndarray | list[str]]:
    events = dataset["events"]
    event_ids = [str(row["event_id"]) for row in events]
    cached_event_ids = list(map(str, cache["event_ids"].tolist()))
    if event_ids != cached_event_ids:
        raise ValueError("semantic cache event order mismatch")
    candidate_ids = sorted(dataset["candidate_catalog"])
    if candidate_ids != list(map(str, cache["candidate_ids"].tolist())):
        raise ValueError("semantic cache candidate order mismatch")
    candidate_index = {name: index for index, name in enumerate(candidate_ids)}
    legal = np.zeros((len(events), len(candidate_ids)), dtype=np.bool_)
    target = np.full(len(events), -1, dtype=np.int64)
    selected = np.asarray([candidate_index[row["current_action_candidate_id"]] for row in events], dtype=np.int64)
    outcomes = np.zeros((len(events), len(OBSERVED_OUTCOME_TARGETS)), dtype=np.float32)
    joint = np.full((len(events), len(JOINT_OUTCOME_CLASSES)), np.nan, dtype=np.float32)
    for index, event in enumerate(events):
        for candidate in event["next_legal_candidate_ids"]:
            legal[index, candidate_index[candidate]] = True
        if event["next_target_candidate_id"] is not None:
            target[index] = candidate_index[event["next_target_candidate_id"]]
        outcomes[index] = [float(event["observed_outcome"][name]) for name in OBSERVED_OUTCOME_TARGETS]
        if event["joint_outcome_trainable"]:
            joint[index] = [float(event["joint_outcome_target"][name]) for name in JOINT_OUTCOME_CLASSES]
    return {
        "field_embeddings": cache["field_embeddings"].astype(np.float32),
        "field_mask": cache["field_mask"].astype(np.bool_),
        "candidate_embeddings": cache["candidate_embeddings"].astype(np.float32),
        "candidate_ids": candidate_ids,
        "legal": legal,
        "target": target,
        "selected": selected,
        "outcomes": outcomes,
        "joint": joint,
    }


class MappingLike:
    def __getitem__(self, key): ...


def trajectory_windows(events: list[dict], arrays: dict, train_tasks: set[str], max_horizon: int = 5):
    grouped = defaultdict(list)
    for index, event in enumerate(events):
        if str(event["task_name"]) in train_tasks:
            grouped[str(event["trajectory_id"])].append(index)
    windows = []
    for sequence in grouped.values():
        sequence.sort(key=lambda index: int(events[index]["step_id"]))
        for position, start in enumerate(sequence):
            for horizon in range(1, max_horizon + 1):
                final_position = position + horizon - 1
                if final_position >= len(sequence):
                    continue
                final = sequence[final_position]
                target = int(arrays["target"][final])
                if target < 0:
                    continue
                path = sequence[position + 1 : final_position + 1]
                windows.append({
                    "start": start,
                    "future_rows": path,
                    "final": final,
                    "horizon": horizon,
                    "target": target,
                })
    return windows


def model_config(protocol: dict) -> LargeWorldModelConfig:
    architecture = protocol["architecture"]
    state = architecture["structured_state_encoder"]
    action = architecture["victim_action_dynamics"]
    residual = architecture["multi_step_residual_dynamics"]
    if len({state["hidden_size"], action["hidden_size"], residual["hidden_size"]}) != 1:
        raise ValueError("all v45 components must share one hidden size")
    if len({state["attention_heads"], action["attention_heads"], residual["attention_heads"]}) != 1:
        raise ValueError("all v45 components must share one attention-head count")
    if len({state["feedforward_size"], action["feedforward_size"], residual["feedforward_size"]}) != 1:
        raise ValueError("all v45 components must share one feed-forward size")
    return LargeWorldModelConfig(
        semantic_size=int(protocol["semantic_backbone"]["embedding_size"]),
        hidden_size=int(state["hidden_size"]),
        state_layers=int(state["layers"]),
        action_layers=int(action["layers"]),
        residual_layers=int(residual["layers"]),
        attention_heads=int(state["attention_heads"]),
        feedforward_size=int(state["feedforward_size"]),
        dropout=float(state["dropout"]),
        memory_tokens=int(residual["memory_tokens"]),
        outcome_size=len(OBSERVED_OUTCOME_TARGETS),
    )


def tensor_rows(array: np.ndarray, indices: np.ndarray, device: str, dtype=None):
    tensor = torch.from_numpy(array[indices])
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(device, non_blocking=True)


def teacher_batch(model, arrays, indices, candidate_embeddings, device):
    fields = tensor_rows(arrays["field_embeddings"], indices, device, torch.float32)
    mask = tensor_rows(arrays["field_mask"], indices, device, torch.bool)
    return model.teacher(fields, mask, candidate_embeddings)


def train_teacher(model, events, arrays, train_indices, protocol, device, seed):
    cfg = protocol["training"]["teacher_stage"]
    loss_weights = cfg["losses"]
    for parameter in model.residual_dynamics.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.state_encoder.parameters()) + list(model.victim_dynamics.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    candidate_embeddings = torch.from_numpy(arrays["candidate_embeddings"]).float().to(device)
    weights = task_weights(events, train_indices)
    weight_lookup = {int(index): float(weight) for index, weight in zip(train_indices, weights)}
    joint_indices = np.asarray(
        [index for index in train_indices if bool(events[int(index)]["joint_outcome_trainable"])],
        dtype=np.int64,
    )
    joint_weights = normalized_joint_event_weights(events, joint_indices)
    joint_weight_lookup = {
        int(index): float(weight) for index, weight in zip(joint_indices, joint_weights)
    }
    outcome_pos = arrays["outcomes"][train_indices].sum(0)
    outcome_neg = len(train_indices) - outcome_pos
    pos_weight = torch.from_numpy(
        np.minimum(outcome_neg / np.maximum(outcome_pos, 1.0), cfg["outcome_positive_weight_cap"]).astype(np.float32)
    ).to(device)
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        rng = np.random.default_rng(seed * 1009 + epoch)
        losses = []
        for batch in batches(train_indices, cfg["batch_size"], rng):
            output = teacher_batch(model, arrays, batch, candidate_embeddings, device)
            batch_targets = tensor_rows(arrays["target"], batch, device, torch.long)
            batch_legal = tensor_rows(arrays["legal"], batch, device, torch.bool)
            has_target = batch_targets >= 0
            action_loss = output["action_logits"].sum() * 0.0
            if has_target.any():
                logits = output["action_logits"][has_target].masked_fill(
                    ~batch_legal[has_target], torch.finfo(output["action_logits"].dtype).min
                )
                per_action = F.cross_entropy(logits, batch_targets[has_target], reduction="none")
                row_weight = torch.tensor(
                    [weight_lookup[int(index)] for index in batch[has_target.cpu().numpy()]],
                    device=device,
                )
                action_loss = (per_action * row_weight).sum() / row_weight.sum()
            outcome_target = tensor_rows(arrays["outcomes"], batch, device, torch.float32)
            per_outcome = F.binary_cross_entropy_with_logits(
                output["outcome_logits"], outcome_target, pos_weight=pos_weight, reduction="none"
            ).mean(1)
            all_row_weight = torch.tensor(
                [weight_lookup[int(index)] for index in batch], device=device
            )
            outcome_loss = (per_outcome * all_row_weight).sum() / all_row_weight.sum()
            joint_target = tensor_rows(np.nan_to_num(arrays["joint"]), batch, device, torch.float32)
            joint_mask = torch.from_numpy(np.isfinite(arrays["joint"][batch]).all(1)).to(device)
            joint_loss = output["joint_logits"].sum() * 0.0
            if joint_mask.any():
                per_joint = -(
                    joint_target[joint_mask]
                    * F.log_softmax(output["joint_logits"][joint_mask], dim=1)
                ).sum(1)
                selected_indices = batch[joint_mask.cpu().numpy()]
                selected_weights = torch.tensor(
                    [joint_weight_lookup[int(index)] for index in selected_indices], device=device
                )
                joint_loss = (per_joint * selected_weights).sum() / selected_weights.sum()
            loss = (
                loss_weights["action_cross_entropy"] * action_loss
                + loss_weights["five_outcome_binary_cross_entropy"] * outcome_loss
                + loss_weights["four_cell_joint_cross_entropy"] * joint_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
            optimizer.step()
            losses.append((float(loss.detach()), float(action_loss.detach()), float(outcome_loss.detach()), float(joint_loss.detach())))
        means = np.mean(losses, axis=0)
        history.append({"epoch": epoch, "total": means[0], "action": means[1], "outcome": means[2], "joint": means[3]})
    return history


def train_residual(model, events, arrays, train_tasks, protocol, device, seed):
    cfg = protocol["training"]["residual_stage"]
    model.freeze_teacher()
    for parameter in model.residual_dynamics.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.residual_dynamics.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    candidates = torch.from_numpy(arrays["candidate_embeddings"]).float().to(device)
    with torch.no_grad():
        candidate_hidden = model.victim_dynamics.encode_candidates(candidates)
    windows = trajectory_windows(events, arrays, train_tasks, cfg["max_horizon"])
    history = []
    for epoch in range(cfg["epochs"]):
        model.residual_dynamics.train()
        rng = np.random.default_rng(seed * 2027 + epoch)
        order = rng.permutation(len(windows))
        losses = []
        for offset in range(0, len(order), cfg["batch_size"]):
            rows = [windows[int(i)] for i in order[offset : offset + cfg["batch_size"]]]
            starts = np.asarray([row["start"] for row in rows], dtype=np.int64)
            finals = np.asarray([row["final"] for row in rows], dtype=np.int64)
            targets = torch.tensor([row["target"] for row in rows], dtype=torch.long, device=device)
            horizons = torch.tensor([row["horizon"] for row in rows], dtype=torch.long, device=device)
            with torch.no_grad():
                teacher = teacher_batch(model, arrays, starts, candidates, device)
            hidden = teacher["state"]
            predicted = []
            for local, row in enumerate(rows):
                value = hidden[local : local + 1]
                if row["horizon"] == 1:
                    logits = teacher["action_logits"][local : local + 1] + model.residual_dynamics.one_step_delta_logits(
                        value, candidate_hidden
                    )
                else:
                    for step, future_index in enumerate(row["future_rows"], start=1):
                        action_index = int(arrays["selected"][future_index])
                        value = model.residual_dynamics.advance(
                            value, candidate_hidden[action_index : action_index + 1], step
                        )
                    logits = model.residual_dynamics.rollout_logits(value, candidate_hidden)
                predicted.append(logits)
            logits = torch.cat(predicted, dim=0)
            legal = tensor_rows(arrays["legal"], finals, device, torch.bool)
            logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
            per = F.cross_entropy(logits, targets, reduction="none")
            weights = torch.tensor(
                [cfg["horizon_weights"][str(int(h))] for h in horizons.tolist()],
                dtype=per.dtype, device=device,
            )
            rollout_loss = (per * weights).sum() / weights.sum()
            h1 = horizons == 1
            kl = logits.sum() * 0.0
            if h1.any():
                teacher_logits = teacher["action_logits"][h1].masked_fill(
                    ~legal[h1], torch.finfo(logits.dtype).min
                )
                kl = F.kl_div(
                    F.log_softmax(logits[h1], dim=1),
                    F.softmax(teacher_logits, dim=1), reduction="batchmean",
                )
            loss = rollout_loss + cfg["h1_teacher_kl_weight"] * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual_dynamics.parameters(), cfg["gradient_clip"])
            optimizer.step()
            losses.append((float(loss.detach()), float(rollout_loss.detach()), float(kl.detach())))
        means = np.mean(losses, axis=0)
        history.append({"epoch": epoch, "total": means[0], "rollout": means[1], "h1_kl": means[2], "windows": len(windows)})
    return history, len(windows)


def evaluate_one_step(model, events, arrays, test_tasks, device, fold, seed):
    model.eval()
    candidates = torch.from_numpy(arrays["candidate_embeddings"]).float().to(device)
    indices = np.asarray([i for i, row in enumerate(events) if str(row["task_name"]) in test_tasks], dtype=np.int64)
    predictions = []
    with torch.no_grad():
        candidate_hidden = model.victim_dynamics.encode_candidates(candidates)
        for batch_start in range(0, len(indices), 64):
            batch = indices[batch_start : batch_start + 64]
            output = teacher_batch(model, arrays, batch, candidates, device)
            legal = tensor_rows(arrays["legal"], batch, device, torch.bool)
            target = arrays["target"][batch]
            logits = output["action_logits"].masked_fill(~legal, torch.finfo(output["action_logits"].dtype).min)
            probability = F.softmax(logits, dim=1).cpu().numpy()
            outcome_probability = torch.sigmoid(output["outcome_logits"]).cpu().numpy()
            joint_probability = F.softmax(output["joint_logits"], dim=1).cpu().numpy()
            residual_logits = logits + model.residual_dynamics.one_step_delta_logits(output["state"], candidate_hidden).masked_fill(
                ~legal, 0.0
            )
            residual_probability = F.softmax(residual_logits, dim=1).cpu().numpy()
            for local, index in enumerate(batch):
                row = events[int(index)]
                y = int(target[local])
                item = {
                    "record_type": "one_step",
                    "fold": fold, "seed": seed, "event_id": row["event_id"],
                    "task_name": row["task_name"], "trajectory_id": row["trajectory_id"],
                    "joint_group_id": row["joint_outcome_group_id"],
                    "joint_trainable": bool(row["joint_outcome_trainable"]),
                    "step_id": row["step_id"], "has_next_action": y >= 0,
                    "teacher_action_nll": -math.log(max(float(probability[local, y]), 1e-12)) if y >= 0 else None,
                    "teacher_action_correct": float(probability[local].argmax() == y) if y >= 0 else None,
                    "residual_h1_nll": -math.log(max(float(residual_probability[local, y]), 1e-12)) if y >= 0 else None,
                    "residual_h1_correct": float(residual_probability[local].argmax() == y) if y >= 0 else None,
                    "outcome_bce": float(-(arrays["outcomes"][index] * np.log(np.clip(outcome_probability[local], 1e-7, 1)) + (1-arrays["outcomes"][index]) * np.log(np.clip(1-outcome_probability[local], 1e-7, 1))).mean()),
                    "legal_teacher_prediction": float(arrays["legal"][index, probability[local].argmax()]),
                    "legal_residual_prediction": float(arrays["legal"][index, residual_probability[local].argmax()]),
                }
                if np.isfinite(arrays["joint"][index]).all():
                    target_joint = arrays["joint"][index]
                    item["joint_cross_entropy"] = float(-(target_joint * np.log(np.clip(joint_probability[local], 1e-7, 1))).sum())
                    item["joint_brier"] = float(((joint_probability[local] - target_joint) ** 2).mean())
                    item["joint_p11"] = float(joint_probability[local, 3])
                else:
                    item.update({"joint_cross_entropy": None, "joint_brier": None, "joint_p11": None})
                predictions.append(item)
    return predictions


def evaluate_rollouts(model, events, arrays, test_tasks, protocol, device, fold, seed):
    """Evaluate every frozen H1-H5 task-disjoint trajectory window."""
    model.eval()
    max_horizon = int(protocol["training"]["residual_stage"]["max_horizon"])
    windows = trajectory_windows(events, arrays, test_tasks, max_horizon)
    candidates = torch.from_numpy(arrays["candidate_embeddings"]).float().to(device)
    predictions = []
    with torch.no_grad():
        candidate_hidden = model.victim_dynamics.encode_candidates(candidates)
        for row in windows:
            start = np.asarray([row["start"]], dtype=np.int64)
            teacher = teacher_batch(model, arrays, start, candidates, device)
            hidden = teacher["state"]
            horizon = int(row["horizon"])
            if horizon == 1:
                logits = teacher["action_logits"] + model.residual_dynamics.one_step_delta_logits(
                    hidden, candidate_hidden
                )
            else:
                for step, future_index in enumerate(row["future_rows"], start=1):
                    action_index = int(arrays["selected"][future_index])
                    hidden = model.residual_dynamics.advance(
                        hidden, candidate_hidden[action_index : action_index + 1], step
                    )
                logits = model.residual_dynamics.rollout_logits(hidden, candidate_hidden)
            final = int(row["final"])
            legal = torch.from_numpy(arrays["legal"][final : final + 1]).to(device)
            logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
            probability = F.softmax(logits, dim=1)[0].cpu().numpy()
            target = int(row["target"])
            event = events[final]
            predictions.append({
                "record_type": "rollout",
                "fold": fold,
                "seed": seed,
                "event_id": event["event_id"],
                "task_name": event["task_name"],
                "trajectory_id": event["trajectory_id"],
                "step_id": event["step_id"],
                "start_event_id": events[int(row["start"])]["event_id"],
                "horizon": horizon,
                "target_candidate_index": target,
                "action_nll": -math.log(max(float(probability[target]), 1e-12)),
                "action_correct": float(probability.argmax() == target),
                "legal_prediction": float(arrays["legal"][final, probability.argmax()]),
            })
    return predictions


def validate_formal_authorization(protocol):
    """Require the exact frozen status and explicit cache/training authorization."""
    authorization = protocol.get("authorization", {})
    if protocol.get("status") != "authorized_for_friend_v100_formal_run":
        raise ValueError("v45 protocol is not authorized for the friend V100 formal run")
    if authorization.get("semantic_cache_build") is not True:
        raise ValueError("v45 formal semantic cache build is not authorized")
    if authorization.get("formal_training_submission") is not True:
        raise ValueError("v45 formal training submission is not authorized")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_formal_authorization(protocol)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    with np.load(args.semantic_cache, allow_pickle=False) as cache:
        arrays = build_arrays(dataset, cache)
    if len(dataset["events"]) != protocol["scope"]["event_count"]:
        raise ValueError("full dataset event count mismatch")
    if args.fold not in range(protocol["scope"]["fold_count"]):
        raise ValueError("invalid fold")
    if args.seed not in protocol["scope"]["seeds"]:
        raise ValueError("invalid seed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal v45 training")
    seed_everything(args.seed)
    device = args.device
    fold_spec = dataset["folds"][args.fold]
    train_tasks = set(map(str, fold_spec["train_tasks"]))
    test_tasks = set(map(str, fold_spec["test_tasks"]))
    if train_tasks & test_tasks:
        raise ValueError("task leakage")
    train_indices = np.asarray([
        i for i, row in enumerate(dataset["events"]) if str(row["task_name"]) in train_tasks
    ], dtype=np.int64)

    model = LargeHybridWorldModel(model_config(protocol)).to(device)
    architecture = model.architecture()
    breakdown = parameter_breakdown(model)
    teacher_history = train_teacher(
        model, dataset["events"], arrays, train_indices, protocol, device, args.seed
    )
    residual_history, window_count = train_residual(
        model, dataset["events"], arrays, train_tasks, protocol, device, args.seed
    )
    one_step_predictions = evaluate_one_step(
        model, dataset["events"], arrays, test_tasks, device, args.fold, args.seed
    )
    rollout_predictions = evaluate_rollouts(
        model, dataset["events"], arrays, test_tasks, protocol, device, args.fold, args.seed
    )
    output = args.output_dir / f"fold{args.fold}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError(f"refusing to append to an existing formal result: {prediction_path}")
    append_jsonl(prediction_path, one_step_predictions + rollout_predictions)
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": architecture,
        "candidate_ids": arrays["candidate_ids"],
        "fold": args.fold,
        "seed": args.seed,
    }, output / "checkpoint.pt")
    metrics = {
        "schema_version": "wmagentattack.large_hybrid_training.v45",
        "fold": args.fold,
        "seed": args.seed,
        "training_tasks": sorted(train_tasks),
        "confirmation_tasks": sorted(test_tasks),
        "training_rows": len(train_indices),
        "one_step_prediction_rows": len(one_step_predictions),
        "rollout_prediction_rows": len(rollout_predictions),
        "residual_windows": window_count,
        "architecture": architecture,
        "parameter_breakdown": breakdown,
        "teacher_history": teacher_history,
        "residual_history": residual_history,
        "runtime_failures": 0,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "fold": args.fold, "seed": args.seed,
        "parameters": breakdown["total"],
        "one_step_prediction_rows": len(one_step_predictions),
        "rollout_prediction_rows": len(rollout_predictions),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
