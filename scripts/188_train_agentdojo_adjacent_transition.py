"""Frozen OOF training for observed AgentDojo adjacent dynamics."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import (
    OBSERVED_OUTCOME_TARGETS,
    ObservedAdjacentTransitionModel,
)
from wmagentattack.hybrid_semantic_world_model import tool_candidate_vector
from wmagentattack.multisource_suitability import (
    file_sha256,
    representation_vector,
)


CONDITIONS = ("tail_action_only", "tail_action_plus_observed_outcomes")
VARIANTS = ("semantic_markov", "structured_markov_v3")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _task_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["task_key"]) for row in rows)
    values = np.asarray(
        [1.0 / len(counts) / counts[str(row["task_key"])] for row in rows],
        dtype=np.float32,
    )
    values *= len(values) / float(values.sum())
    return values


def _fold_events(
    dataset: Mapping[str, Any], *, fold: int
) -> list[dict[str, Any]]:
    surface = dataset["folds"][fold]
    training_tasks = set(surface["train_tasks"])
    confirmation_tasks = set(surface["test_tasks"])
    output: list[dict[str, Any]] = []
    for source in dataset["events"]:
        task = str(source["task_name"])
        if task in training_tasks:
            split = "training"
        elif task in confirmation_tasks:
            split = "confirmation"
        else:
            continue
        row = dict(source)
        row["split"] = split
        output.append(row)
    if {row["task_name"] for row in output if row["split"] == "training"} != training_tasks:
        raise ValueError("training tasks differ from frozen fold")
    if {row["task_name"] for row in output if row["split"] == "confirmation"} != confirmation_tasks:
        raise ValueError("confirmation tasks differ from frozen fold")
    return output


def _arrays(
    events: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    hash_dimension: int,
) -> dict[str, Any]:
    candidates = sorted(catalog)
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    states = np.stack(
        [
            representation_vector(
                event, variant=variant, hash_dimension=hash_dimension
            )
            for event in events
        ]
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(catalog[candidate], hash_dimension=hash_dimension)
            for candidate in candidates
        ]
    )
    selected = np.asarray(
        [candidate_index[str(event["current_action_candidate_id"])] for event in events],
        dtype=np.int64,
    )
    legal = np.zeros((len(events), len(candidates)), dtype=bool)
    next_targets = np.full(len(events), -1, dtype=np.int64)
    outcomes = np.zeros((len(events), len(OBSERVED_OUTCOME_TARGETS)), dtype=np.float32)
    for row_index, event in enumerate(events):
        for candidate in event["next_legal_candidate_ids"]:
            legal[row_index, candidate_index[str(candidate)]] = True
        target = event["next_target_candidate_id"]
        if target is not None:
            next_targets[row_index] = candidate_index[str(target)]
            if not legal[row_index, next_targets[row_index]]:
                raise ValueError("next action is not legal")
        outcomes[row_index] = [
            float(event["observed_outcome"][name])
            for name in OBSERVED_OUTCOME_TARGETS
        ]
    return {
        "candidates": candidates,
        "states": states,
        "candidate_inputs": candidate_inputs,
        "selected": selected,
        "legal": legal,
        "next_targets": next_targets,
        "outcomes": outcomes,
    }


def _train(
    *,
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    condition: str,
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    training_indices = np.asarray(
        [index for index, row in enumerate(events) if row["split"] == "training"],
        dtype=np.int64,
    )
    tail_indices = np.asarray(
        [index for index in training_indices if int(arrays["next_targets"][index]) >= 0],
        dtype=np.int64,
    )
    training_rows = [events[int(index)] for index in training_indices]
    tail_rows = [events[int(index)] for index in tail_indices]
    all_weights = torch.as_tensor(
        _task_balanced_weights(training_rows), dtype=torch.float32, device=device
    )
    tail_weights = torch.as_tensor(
        _task_balanced_weights(tail_rows), dtype=torch.float32, device=device
    )
    states = torch.as_tensor(arrays["states"], dtype=torch.float32, device=device)
    candidate_inputs = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    selected_inputs = candidate_inputs[
        torch.as_tensor(arrays["selected"], dtype=torch.long, device=device)
    ]
    legal = torch.as_tensor(arrays["legal"], dtype=torch.bool, device=device)
    next_targets = torch.as_tensor(
        arrays["next_targets"], dtype=torch.long, device=device
    )
    outcomes = torch.as_tensor(
        arrays["outcomes"], dtype=torch.float32, device=device
    )
    training = protocol["training"]
    model = ObservedAdjacentTransitionModel(
        state_size=int(arrays["states"].shape[1]),
        candidate_size=int(arrays["candidate_inputs"].shape[1]),
        hidden_size=int(training["hidden_size"]),
        dropout=float(training["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positive = arrays["outcomes"][training_indices].sum(axis=0)
    negative = len(training_indices) - positive
    positive_weight = np.minimum(
        negative / np.maximum(positive, 1.0),
        float(training["outcome_positive_weight_cap"]),
    ).astype(np.float32)
    positive_weight_tensor = torch.as_tensor(
        positive_weight, dtype=torch.float32, device=device
    )
    history = []
    for epoch in range(int(training["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        action_logits, outcome_logits = model(
            states, selected_inputs, candidate_inputs
        )
        tail_logits = action_logits[tail_indices].masked_fill(
            ~legal[tail_indices], torch.finfo(action_logits.dtype).min
        )
        per_tail = F.cross_entropy(
            tail_logits, next_targets[tail_indices], reduction="none"
        )
        action_loss = (per_tail * tail_weights).sum() / tail_weights.sum()
        outcome_loss = torch.zeros((), device=device)
        if condition == "tail_action_plus_observed_outcomes":
            per_outcome = F.binary_cross_entropy_with_logits(
                outcome_logits[training_indices],
                outcomes[training_indices],
                reduction="none",
                pos_weight=positive_weight_tensor,
            ).mean(dim=1)
            outcome_loss = (per_outcome * all_weights).sum() / all_weights.sum()
        loss = action_loss + float(training["outcome_loss_weight"]) * outcome_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(training["fixed_epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.detach().cpu()),
                    "tail_action_loss": float(action_loss.detach().cpu()),
                    "outcome_loss": float(outcome_loss.detach().cpu()),
                }
            )
    model.eval()
    with torch.no_grad():
        next_probabilities = model.next_action_probabilities(
            states, selected_inputs, candidate_inputs, legal
        ).cpu().numpy()
        _, outcome_logits = model(states, selected_inputs, candidate_inputs)
        outcome_probabilities = torch.sigmoid(outcome_logits).cpu().numpy()
    train_prior = np.clip(
        arrays["outcomes"][training_indices].mean(axis=0), 1e-6, 1.0 - 1e-6
    )
    return next_probabilities, outcome_probabilities, {
        "training_rows": len(training_indices),
        "training_tail_rows": len(tail_indices),
        "training_tasks": len({row["task_key"] for row in training_rows}),
        "training_outcome_positive_rows": dict(
            zip(OBSERVED_OUTCOME_TARGETS, map(int, positive))
        ),
        "training_outcome_prior": dict(
            zip(OBSERVED_OUTCOME_TARGETS, map(float, train_prior))
        ),
        "outcome_positive_weight": dict(
            zip(OBSERVED_OUTCOME_TARGETS, map(float, positive_weight))
        ),
        "loss_history_endpoints": history,
    }


def _binary_bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))


def _prediction_rows(
    *,
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    next_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
    diagnostics: Mapping[str, Any],
    fold: int,
    condition: str,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = arrays["candidates"]
    train_prior = np.asarray(
        [diagnostics["training_outcome_prior"][name] for name in OBSERVED_OUTCOME_TARGETS],
        dtype=np.float32,
    )
    output = []
    for index, event in enumerate(events):
        if event["split"] != "confirmation":
            continue
        target_outcome = arrays["outcomes"][index]
        outcome_bce = _binary_bce(outcome_probabilities[index], target_outcome)
        baseline_bce = _binary_bce(train_prior, target_outcome)
        target_index = int(arrays["next_targets"][index])
        predicted_index = int(np.argmax(next_probabilities[index]))
        legal_indices = set(np.flatnonzero(arrays["legal"][index]).tolist())
        row = {
            "fold": fold,
            "condition": condition,
            "variant": variant,
            "training_seed": seed,
            "event_id": event["event_id"],
            "task_key": event["task_key"],
            "task_name": event["task_name"],
            "domain": str(event["task_name"]).split("|", 1)[0],
            "trajectory_id": event["trajectory_id"],
            "step_id": event["step_id"],
            "has_next_action": float(target_index >= 0),
            "next_target_candidate_id": (
                candidates[target_index] if target_index >= 0 else None
            ),
            "next_predicted_candidate_id": candidates[predicted_index],
            "next_action_nll": (
                float(-math.log(max(float(next_probabilities[index, target_index]), 1e-12)))
                if target_index >= 0
                else None
            ),
            "next_action_correct": (
                float(predicted_index == target_index) if target_index >= 0 else None
            ),
            "legal_prediction": float(predicted_index in legal_indices),
            "outcome_bce": float(np.mean(outcome_bce)),
            "outcome_prior_bce": float(np.mean(baseline_bce)),
        }
        for target_index_outcome, name in enumerate(OBSERVED_OUTCOME_TARGETS):
            row[f"{name}_target"] = float(target_outcome[target_index_outcome])
            row[f"{name}_probability"] = float(
                outcome_probabilities[index, target_index_outcome]
            )
            row[f"{name}_bce"] = float(outcome_bce[target_index_outcome])
            row[f"{name}_prior_bce"] = float(baseline_bce[target_index_outcome])
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preflight_passed_and_frozen_before_training":
        raise ValueError("protocol is not frozen after preflight")
    frozen = protocol["frozen_transition_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("transition dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("transition audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("transition preflight did not pass")
    if tuple(protocol["training"]["conditions"]) != CONDITIONS:
        raise ValueError("condition surface differs from frozen protocol")
    if tuple(protocol["training"]["variants"]) != VARIANTS:
        raise ValueError("variant surface differs from frozen protocol")
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]
    expected_runs = len(dataset["folds"]) * len(CONDITIONS) * len(VARIANTS) * len(seeds)
    if expected_runs != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("fixed neural budget is inconsistent")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    runs = []
    for fold in range(len(dataset["folds"])):
        events = _fold_events(dataset, fold=fold)
        for variant in VARIANTS:
            arrays = _arrays(
                events,
                dataset["candidate_catalog"],
                variant=variant,
                hash_dimension=int(protocol["training"]["hash_dimension"]),
            )
            for condition in CONDITIONS:
                for seed in seeds:
                    _set_seed(seed)
                    next_probabilities, outcome_probabilities, diagnostics = _train(
                        events=events,
                        arrays=arrays,
                        condition=condition,
                        protocol=protocol,
                        seed=seed,
                        device=device,
                    )
                    predictions = _prediction_rows(
                        events=events,
                        arrays=arrays,
                        next_probabilities=next_probabilities,
                        outcome_probabilities=outcome_probabilities,
                        diagnostics=diagnostics,
                        fold=fold,
                        condition=condition,
                        variant=variant,
                        seed=seed,
                    )
                    _append_jsonl(prediction_path, predictions)
                    runs.append(
                        {
                            "fold": fold,
                            "condition": condition,
                            "variant": variant,
                            "training_seed": seed,
                            "confirmation_rows": len(predictions),
                            "confirmation_tail_rows": sum(
                                row["has_next_action"] for row in predictions
                            ),
                            **diagnostics,
                        }
                    )
    if len(runs) != expected_runs:
        raise ValueError("neural run budget incomplete")
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "predictions_sha256": file_sha256(prediction_path),
        "device": device,
        "runs": runs,
        "neural_training_runs": len(runs),
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "real_external_endpoint_calls": 0,
        "new_attack_generation": 0,
        "dreamer_runs": 0,
    }
    _write_json(args.output_dir / "run_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
