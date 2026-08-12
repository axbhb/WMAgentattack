"""Train the four preregistered integrated validation conditions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import OBSERVED_OUTCOME_TARGETS
from wmagentattack.fresh_integrated_validation import (
    FROZEN_SOURCES,
    FreshIntegratedSemanticWorldModel,
    assert_no_unauthorized_heads,
)
from wmagentattack.hybrid_semantic_world_model import tool_candidate_vector
from wmagentattack.multisource_suitability import file_sha256, representation_vector


CONDITIONS = (
    "agentdojo_only_tail_plus_outcomes",
    "pooled_shared_tail_plus_outcomes",
    "pooled_source_head_tail_only",
    "pooled_source_head_tail_plus_outcomes",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _task_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["task_key"]) for row in rows)
    values = np.asarray(
        [1.0 / len(counts) / counts[str(row["task_key"])] for row in rows],
        dtype=np.float32,
    )
    values *= len(values) / float(values.sum())
    return values


def _source_task_balanced_weights(
    rows: Sequence[Mapping[str, Any]], source_mass: Mapping[str, float]
) -> np.ndarray:
    tasks: dict[str, set[str]] = {}
    counts = Counter((str(row["source"]), str(row["task_key"])) for row in rows)
    for row in rows:
        tasks.setdefault(str(row["source"]), set()).add(str(row["task_key"]))
    if set(tasks) != set(source_mass):
        raise ValueError(f"source mass differs from observed training surface: {set(tasks)}")
    values = np.asarray(
        [
            float(source_mass[str(row["source"])] )
            / len(tasks[str(row["source"])] )
            / counts[(str(row["source"]), str(row["task_key"]))]
            for row in rows
        ],
        dtype=np.float32,
    )
    values *= len(values) / float(values.sum())
    return values


def _candidate_surface(dataset: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    catalog = dataset["candidate_catalog"]
    candidates = sorted(catalog)
    vectors = np.stack(
        [tool_candidate_vector(catalog[key], hash_dimension=128) for key in candidates]
    )
    return candidates, vectors


def _state_matrix(rows: Sequence[Mapping[str, Any]], hash_dimension: int) -> np.ndarray:
    return np.stack(
        [
            representation_vector(
                row, variant="structured_markov_v3", hash_dimension=hash_dimension
            )
            for row in rows
        ]
    )


def _action_arrays(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    hash_dimension: int,
) -> dict[str, Any]:
    index = {key: i for i, key in enumerate(candidates)}
    legal = np.zeros((len(rows), len(candidates)), dtype=bool)
    targets = np.zeros(len(rows), dtype=np.int64)
    sources = np.zeros(len(rows), dtype=np.int64)
    source_index = {name: i for i, name in enumerate(FROZEN_SOURCES)}
    for row_index, row in enumerate(rows):
        for candidate in row["legal_candidate_ids"]:
            legal[row_index, index[str(candidate)]] = True
        targets[row_index] = index[str(row["target_candidate_id"])]
        sources[row_index] = source_index[str(row["source"])]
        if not legal[row_index, targets[row_index]]:
            raise ValueError("current target is not legal")
    return {
        "states": _state_matrix(rows, hash_dimension),
        "legal": legal,
        "targets": targets,
        "sources": sources,
    }


def _transition_arrays(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    hash_dimension: int,
) -> dict[str, Any]:
    index = {key: i for i, key in enumerate(candidates)}
    legal = np.zeros((len(rows), len(candidates)), dtype=bool)
    selected = np.zeros(len(rows), dtype=np.int64)
    targets = np.full(len(rows), -1, dtype=np.int64)
    outcomes = np.zeros((len(rows), len(OBSERVED_OUTCOME_TARGETS)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        selected[row_index] = index[str(row["current_action_candidate_id"])]
        for candidate in row["next_legal_candidate_ids"]:
            legal[row_index, index[str(candidate)]] = True
        target = row["next_target_candidate_id"]
        if target is not None:
            targets[row_index] = index[str(target)]
            if not legal[row_index, targets[row_index]]:
                raise ValueError("tail target is not legal")
        outcomes[row_index] = [
            float(row["observed_outcome"][name]) for name in OBSERVED_OUTCOME_TARGETS
        ]
    return {
        "states": _state_matrix(rows, hash_dimension),
        "legal": legal,
        "selected": selected,
        "targets": targets,
        "outcomes": outcomes,
    }


def _binary_bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))


def _train_condition(
    *,
    dataset: Mapping[str, Any],
    candidates: Sequence[str],
    candidate_inputs_np: np.ndarray,
    action_rows: Sequence[Mapping[str, Any]],
    action_arrays: Mapping[str, Any],
    transition_rows: Sequence[Mapping[str, Any]],
    transition_arrays: Mapping[str, Any],
    condition: str,
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    historical_action_count = len(dataset["historical_action_rows"])
    historical_transition_count = len(dataset["historical_transition_events"])
    historical_action_indices = np.arange(historical_action_count, dtype=np.int64)
    if condition == "agentdojo_only_tail_plus_outcomes":
        historical_action_indices = np.asarray(
            [
                index
                for index in historical_action_indices
                if action_rows[int(index)]["source"] == "agentdojo"
            ],
            dtype=np.int64,
        )
    historical_transition_indices = np.arange(
        historical_transition_count, dtype=np.int64
    )
    tail_indices = historical_transition_indices[
        transition_arrays["targets"][historical_transition_indices] >= 0
    ]
    action_train_rows = [action_rows[int(index)] for index in historical_action_indices]
    transition_train_rows = [
        transition_rows[int(index)] for index in historical_transition_indices
    ]
    tail_train_rows = [transition_rows[int(index)] for index in tail_indices]
    training = protocol["model_validation"]
    source_mass = training["source_mass"][condition]
    action_weights_np = _source_task_balanced_weights(action_train_rows, source_mass)
    transition_weights_np = _task_balanced_weights(transition_train_rows)
    tail_weights_np = _task_balanced_weights(tail_train_rows)

    candidate_inputs = torch.as_tensor(
        candidate_inputs_np, dtype=torch.float32, device=device
    )
    action_states = torch.as_tensor(
        action_arrays["states"], dtype=torch.float32, device=device
    )
    action_legal = torch.as_tensor(action_arrays["legal"], dtype=torch.bool, device=device)
    action_targets = torch.as_tensor(action_arrays["targets"], dtype=torch.long, device=device)
    action_sources = torch.as_tensor(action_arrays["sources"], dtype=torch.long, device=device)
    transition_states = torch.as_tensor(
        transition_arrays["states"], dtype=torch.float32, device=device
    )
    transition_legal = torch.as_tensor(
        transition_arrays["legal"], dtype=torch.bool, device=device
    )
    transition_selected = torch.as_tensor(
        transition_arrays["selected"], dtype=torch.long, device=device
    )
    transition_targets = torch.as_tensor(
        transition_arrays["targets"], dtype=torch.long, device=device
    )
    transition_outcomes = torch.as_tensor(
        transition_arrays["outcomes"], dtype=torch.float32, device=device
    )
    action_weights = torch.as_tensor(action_weights_np, dtype=torch.float32, device=device)
    transition_weights = torch.as_tensor(
        transition_weights_np, dtype=torch.float32, device=device
    )
    tail_weights = torch.as_tensor(tail_weights_np, dtype=torch.float32, device=device)
    source_specific = condition.startswith("pooled_source_head")
    outcomes_enabled = not condition.endswith("tail_only")
    model = FreshIntegratedSemanticWorldModel(
        state_size=int(action_arrays["states"].shape[1]),
        candidate_size=int(candidate_inputs_np.shape[1]),
        hidden_size=int(training["hidden_size"]),
        source_count=len(FROZEN_SOURCES),
        source_specific_action_head=source_specific,
        dropout=float(training["dropout"]),
    ).to(device)
    assert_no_unauthorized_heads(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positive = transition_arrays["outcomes"][historical_transition_indices].sum(axis=0)
    negative = len(historical_transition_indices) - positive
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
        current_logits = model.current_action_logits(
            action_states[historical_action_indices],
            candidate_inputs,
            action_sources[historical_action_indices],
        )
        current_logits = current_logits.masked_fill(
            ~action_legal[historical_action_indices],
            torch.finfo(current_logits.dtype).min,
        )
        current_loss_rows = F.cross_entropy(
            current_logits, action_targets[historical_action_indices], reduction="none"
        )
        current_loss = (current_loss_rows * action_weights).sum() / action_weights.sum()
        tail_logits, outcome_logits = model.transition_logits(
            transition_states,
            candidate_inputs[transition_selected],
            candidate_inputs,
        )
        train_tail_logits = tail_logits[tail_indices].masked_fill(
            ~transition_legal[tail_indices], torch.finfo(tail_logits.dtype).min
        )
        tail_loss_rows = F.cross_entropy(
            train_tail_logits, transition_targets[tail_indices], reduction="none"
        )
        tail_loss = (tail_loss_rows * tail_weights).sum() / tail_weights.sum()
        outcome_loss = torch.zeros((), device=device)
        if outcomes_enabled:
            outcome_loss_rows = F.binary_cross_entropy_with_logits(
                outcome_logits[historical_transition_indices],
                transition_outcomes[historical_transition_indices],
                reduction="none",
                pos_weight=positive_weight_tensor,
            ).mean(dim=1)
            outcome_loss = (
                outcome_loss_rows * transition_weights
            ).sum() / transition_weights.sum()
        loss = (
            current_loss
            + tail_loss
            + float(training["outcome_loss_weight"]) * outcome_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(training["fixed_epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.detach().cpu()),
                    "current_action_loss": float(current_loss.detach().cpu()),
                    "tail_action_loss": float(tail_loss.detach().cpu()),
                    "outcome_loss": float(outcome_loss.detach().cpu()),
                }
            )
    model.eval()
    with torch.no_grad():
        current_logits = model.current_action_logits(
            action_states, candidate_inputs, action_sources
        )
        current_probabilities = model.probabilities(
            current_logits, action_legal
        ).cpu().numpy()
        tail_logits, outcome_logits = model.transition_logits(
            transition_states,
            candidate_inputs[transition_selected],
            candidate_inputs,
        )
        tail_probabilities = model.probabilities(
            tail_logits, transition_legal
        ).cpu().numpy()
        outcome_probabilities = torch.sigmoid(outcome_logits).cpu().numpy()
    train_prior = np.clip(
        transition_arrays["outcomes"][historical_transition_indices].mean(axis=0),
        1e-6,
        1.0 - 1e-6,
    )
    realized = Counter()
    for row, weight in zip(action_train_rows, action_weights_np):
        realized[str(row["source"])] += float(weight)
    total = sum(realized.values())
    diagnostics = {
        "historical_current_action_rows": len(historical_action_indices),
        "historical_transition_events": len(historical_transition_indices),
        "historical_tail_events": len(tail_indices),
        "training_tasks_by_source": {
            source: len(
                {row["task_key"] for row in action_train_rows if row["source"] == source}
            )
            for source in sorted({str(row["source"]) for row in action_train_rows})
        },
        "realized_source_mass": {
            source: value / total for source, value in sorted(realized.items())
        },
        "source_specific_current_action_head": source_specific,
        "observed_outcome_objective_enabled": outcomes_enabled,
        "training_outcome_prior": dict(
            zip(OBSERVED_OUTCOME_TARGETS, map(float, train_prior))
        ),
        "training_outcome_positive_rows": dict(
            zip(OBSERVED_OUTCOME_TARGETS, map(int, positive))
        ),
        "loss_history_endpoints": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return current_probabilities, tail_probabilities, outcome_probabilities, diagnostics


def _prediction_rows(
    *,
    dataset: Mapping[str, Any],
    candidates: Sequence[str],
    action_rows: Sequence[Mapping[str, Any]],
    action_arrays: Mapping[str, Any],
    transition_rows: Sequence[Mapping[str, Any]],
    transition_arrays: Mapping[str, Any],
    current_probabilities: np.ndarray,
    tail_probabilities: np.ndarray,
    outcome_probabilities: np.ndarray,
    diagnostics: Mapping[str, Any],
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    current_start = len(dataset["historical_action_rows"])
    transition_start = len(dataset["historical_transition_events"])
    train_prior = np.asarray(
        [diagnostics["training_outcome_prior"][name] for name in OBSERVED_OUTCOME_TARGETS],
        dtype=np.float32,
    )
    for index in range(current_start, len(action_rows)):
        row = action_rows[index]
        target = int(action_arrays["targets"][index])
        predicted = int(np.argmax(current_probabilities[index]))
        legal_indices = set(np.flatnonzero(action_arrays["legal"][index]).tolist())
        output.append(
            {
                "prediction_type": "current_action",
                "condition": condition,
                "training_seed": seed,
                "row_id": row["row_id"],
                "task_key": row["task_key"],
                "task_name": row["task_name"],
                "domain": row["task_name"].split("|", 1)[0],
                "trajectory_id": row["repeat_id"],
                "step_id": int(row["row_id"].rsplit("step", 1)[-1]),
                "target_candidate_id": candidates[target],
                "predicted_candidate_id": candidates[predicted],
                "action_nll": float(
                    -math.log(max(float(current_probabilities[index, target]), 1e-12))
                ),
                "action_correct": float(predicted == target),
                "legal_prediction": float(predicted in legal_indices),
            }
        )
    for index in range(transition_start, len(transition_rows)):
        row = transition_rows[index]
        target = int(transition_arrays["targets"][index])
        predicted = int(np.argmax(tail_probabilities[index]))
        legal_indices = set(np.flatnonzero(transition_arrays["legal"][index]).tolist())
        target_outcomes = transition_arrays["outcomes"][index]
        outcome_bce = _binary_bce(outcome_probabilities[index], target_outcomes)
        prior_bce = _binary_bce(train_prior, target_outcomes)
        prediction = {
            "prediction_type": "transition",
            "condition": condition,
            "training_seed": seed,
            "event_id": row["event_id"],
            "task_key": row["task_key"],
            "task_name": row["task_name"],
            "domain": row["task_name"].split("|", 1)[0],
            "trajectory_id": row["trajectory_id"],
            "step_id": row["step_id"],
            "has_next_action": float(target >= 0),
            "next_target_candidate_id": candidates[target] if target >= 0 else None,
            "next_predicted_candidate_id": candidates[predicted],
            "next_action_nll": (
                float(-math.log(max(float(tail_probabilities[index, target]), 1e-12)))
                if target >= 0
                else None
            ),
            "next_action_correct": float(predicted == target) if target >= 0 else None,
            "legal_prediction": float(predicted in legal_indices),
            "outcome_bce": float(np.mean(outcome_bce)),
            "outcome_prior_bce": float(np.mean(prior_bce)),
        }
        for outcome_index, name in enumerate(OBSERVED_OUTCOME_TARGETS):
            prediction[f"{name}_target"] = float(target_outcomes[outcome_index])
            prediction[f"{name}_probability"] = float(
                outcome_probabilities[index, outcome_index]
            )
            prediction[f"{name}_bce"] = float(outcome_bce[outcome_index])
            prediction[f"{name}_prior_bce"] = float(prior_bce[outcome_index])
        output.append(prediction)
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
    if protocol["status"] != "clean_gate_passed_and_integrated_dataset_frozen_before_training":
        raise ValueError("integrated protocol is not frozen before neural training")
    frozen = protocol["frozen_integrated_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("integrated dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("integrated dataset audit hash mismatch")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("integrated dataset audit did not pass")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    training = protocol["model_validation"]
    if tuple(training["conditions"]) != CONDITIONS:
        raise ValueError("condition surface differs from frozen protocol")
    seeds = [int(seed) for seed in training["training_seeds"]]
    expected_runs = len(CONDITIONS) * len(seeds)
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

    hash_dimension = int(training["hash_dimension"])
    candidates = sorted(dataset["candidate_catalog"])
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                dataset["candidate_catalog"][candidate],
                hash_dimension=hash_dimension,
            )
            for candidate in candidates
        ]
    )
    action_rows = [
        *dataset["historical_action_rows"],
        *dataset["fresh_action_rows"],
    ]
    transition_rows = [
        *dataset["historical_transition_events"],
        *dataset["fresh_transition_events"],
    ]
    action_arrays = _action_arrays(action_rows, candidates, hash_dimension)
    transition_arrays = _transition_arrays(
        transition_rows, candidates, hash_dimension
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    runs = []
    for condition in CONDITIONS:
        for seed in seeds:
            _set_seed(seed)
            current, tail, outcomes, diagnostics = _train_condition(
                dataset=dataset,
                candidates=candidates,
                candidate_inputs_np=candidate_inputs,
                action_rows=action_rows,
                action_arrays=action_arrays,
                transition_rows=transition_rows,
                transition_arrays=transition_arrays,
                condition=condition,
                protocol=protocol,
                seed=seed,
                device=device,
            )
            predictions = _prediction_rows(
                dataset=dataset,
                candidates=candidates,
                action_rows=action_rows,
                action_arrays=action_arrays,
                transition_rows=transition_rows,
                transition_arrays=transition_arrays,
                current_probabilities=current,
                tail_probabilities=tail,
                outcome_probabilities=outcomes,
                diagnostics=diagnostics,
                condition=condition,
                seed=seed,
            )
            _append(prediction_path, predictions)
            runs.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "fresh_prediction_rows": len(predictions),
                    **diagnostics,
                }
            )
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "device": device,
        "conditions": list(CONDITIONS),
        "training_seeds": seeds,
        "neural_training_runs": len(runs),
        "runs": runs,
        "predictions_sha256": file_sha256(prediction_path),
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "real_external_endpoint_calls": 0,
        "attack_episodes": 0,
        "dreamer_runs": 0,
    }
    if len(runs) != expected_runs:
        raise ValueError("neural training budget incomplete")
    _write(args.output_dir / "run_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
