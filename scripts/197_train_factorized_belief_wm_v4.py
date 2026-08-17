"""Frozen task-disjoint pilot for the factorized belief world model v4."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
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
from wmagentattack.factorized_belief_world_model import (
    FactorizedBeliefWorldModel,
    TYPED_STATE_NODES,
    assert_factorized_scope,
    masked_action_probabilities,
    stack_typed_state_nodes,
)
from wmagentattack.hybrid_semantic_world_model import tool_candidate_vector
from wmagentattack.multisource_suitability import file_sha256, representation_vector


ARMS = ("structured_mlp", "fns_bwm_onestep", "fns_bwm_multihorizon")


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _task_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["task_key"]) for row in rows)
    values = np.asarray(
        [1.0 / len(counts) / counts[str(row["task_key"])] for row in rows],
        dtype=np.float32,
    )
    values *= len(values) / float(values.sum())
    return values


def _fold_events(dataset: Mapping[str, Any], *, fold: int) -> list[dict[str, Any]]:
    surface = dataset["folds"][fold]
    training_tasks = set(surface["train_tasks"])
    confirmation_tasks = set(surface["test_tasks"])
    output = []
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
        raise ValueError("training tasks differ from the frozen fold")
    if {row["task_name"] for row in output if row["split"] == "confirmation"} != confirmation_tasks:
        raise ValueError("confirmation tasks differ from the frozen fold")
    return output


def _arrays(
    events: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    state_hash_dimension: int,
    typed_hash_dimension: int,
    maximum_horizon: int,
) -> dict[str, Any]:
    candidates = sorted(catalog)
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    structured_states = np.stack(
        [
            representation_vector(
                event,
                variant="structured_markov_v3",
                hash_dimension=state_hash_dimension,
            )
            for event in events
        ]
    )
    typed_nodes = stack_typed_state_nodes(
        events, hash_dimension=typed_hash_dimension
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                catalog[candidate], hash_dimension=state_hash_dimension
            )
            for candidate in candidates
        ]
    )
    selected = np.asarray(
        [candidate_index[str(event["current_action_candidate_id"])] for event in events],
        dtype=np.int64,
    )
    legal = np.zeros((len(events), len(candidates)), dtype=bool)
    outcomes = np.zeros((len(events), len(OBSERVED_OUTCOME_TARGETS)), dtype=np.float32)
    for row_index, event in enumerate(events):
        for candidate in event["next_legal_candidate_ids"]:
            legal[row_index, candidate_index[str(candidate)]] = True
        outcomes[row_index] = [
            float(event["observed_outcome"][name])
            for name in OBSERVED_OUTCOME_TARGETS
        ]

    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_trajectory[str(event["trajectory_id"])].append(index)
    horizons: dict[int, dict[str, np.ndarray]] = {}
    for horizon in range(1, maximum_horizon + 1):
        starts: list[int] = []
        action_paths: list[list[int]] = []
        legal_paths: list[list[np.ndarray]] = []
        targets: list[int] = []
        for indices in by_trajectory.values():
            ordered = sorted(indices, key=lambda index: int(events[index]["step_id"]))
            for position in range(len(ordered) - horizon):
                path = ordered[position : position + horizon + 1]
                if [int(events[index]["step_id"]) for index in path] != list(
                    range(
                        int(events[path[0]]["step_id"]),
                        int(events[path[0]]["step_id"]) + horizon + 1,
                    )
                ):
                    raise ValueError("non-contiguous horizon path")
                starts.append(path[0])
                action_paths.append([int(selected[index]) for index in path[:-1]])
                legal_paths.append([legal[index].copy() for index in path[:-1]])
                targets.append(int(selected[path[-1]]))
                if not legal_paths[-1][-1][targets[-1]]:
                    raise ValueError("horizon target is outside the legal interface")
        horizons[horizon] = {
            "starts": np.asarray(starts, dtype=np.int64),
            "action_paths": np.asarray(action_paths, dtype=np.int64),
            "legal_paths": np.asarray(legal_paths, dtype=bool),
            "targets": np.asarray(targets, dtype=np.int64),
        }
    return {
        "candidates": candidates,
        "structured_states": structured_states,
        "typed_nodes": typed_nodes,
        "candidate_inputs": candidate_inputs,
        "selected": selected,
        "legal": legal,
        "outcomes": outcomes,
        "horizons": horizons,
    }


def _positive_weights(
    outcomes: np.ndarray, training_indices: np.ndarray, cap: float
) -> np.ndarray:
    positive = outcomes[training_indices].sum(axis=0)
    negative = len(training_indices) - positive
    return np.minimum(negative / np.maximum(positive, 1.0), cap).astype(np.float32)


def _build_model(
    arm: str, arrays: Mapping[str, Any], training: Mapping[str, Any]
) -> torch.nn.Module:
    if arm == "structured_mlp":
        return ObservedAdjacentTransitionModel(
            state_size=int(arrays["structured_states"].shape[1]),
            candidate_size=int(arrays["candidate_inputs"].shape[1]),
            hidden_size=int(training["hidden_size"]),
            dropout=float(training["dropout"]),
        )
    model = FactorizedBeliefWorldModel(
        structured_state_size=int(arrays["structured_states"].shape[1]),
        node_feature_size=int(arrays["typed_nodes"].shape[2]),
        node_count=int(arrays["typed_nodes"].shape[1]),
        candidate_size=int(arrays["candidate_inputs"].shape[1]),
        hidden_size=int(training["hidden_size"]),
        attention_heads=int(training["attention_heads"]),
        attention_layers=int(training["attention_layers"]),
        dropout=float(training["dropout"]),
    )
    assert_factorized_scope(model)
    return model


def _train(
    *,
    arm: str,
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    training = protocol["training"]
    training_indices = np.asarray(
        [index for index, row in enumerate(events) if row["split"] == "training"],
        dtype=np.int64,
    )
    all_weights = torch.as_tensor(
        _task_balanced_weights([events[int(index)] for index in training_indices]),
        dtype=torch.float32,
        device=device,
    )
    structured = torch.as_tensor(
        arrays["structured_states"], dtype=torch.float32, device=device
    )
    typed_nodes = torch.as_tensor(arrays["typed_nodes"], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    selected = torch.as_tensor(arrays["selected"], dtype=torch.long, device=device)
    legal = torch.as_tensor(arrays["legal"], dtype=torch.bool, device=device)
    outcomes = torch.as_tensor(arrays["outcomes"], dtype=torch.float32, device=device)
    positive_weight = _positive_weights(
        arrays["outcomes"],
        training_indices,
        float(training["outcome_positive_weight_cap"]),
    )
    positive_weight_tensor = torch.as_tensor(
        positive_weight, dtype=torch.float32, device=device
    )
    model = _build_model(arm, arrays, training).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    horizons = tuple(int(value) for value in protocol["training"]["horizons"])
    horizon_weights = {
        int(key): float(value)
        for key, value in protocol["training"]["horizon_loss_weights"].items()
    }
    history = []
    for epoch in range(int(training["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        if arm == "structured_mlp":
            selected_inputs = candidates[selected]
            logits, outcome_logits = model(structured, selected_inputs, candidates)
            surface = arrays["horizons"][1]
            train_surface = np.asarray(
                [events[int(index)]["split"] == "training" for index in surface["starts"]],
                dtype=bool,
            )
            starts_np = surface["starts"][train_surface]
            targets_np = surface["targets"][train_surface]
            starts = torch.as_tensor(starts_np, dtype=torch.long, device=device)
            targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
            weights = torch.as_tensor(
                _task_balanced_weights([events[int(index)] for index in starts_np]),
                dtype=torch.float32,
                device=device,
            )
            masked = logits[starts].masked_fill(
                ~legal[starts], torch.finfo(logits.dtype).min
            )
            per_action = F.cross_entropy(masked, targets, reduction="none")
            action_loss = (per_action * weights).sum() / weights.sum()
        else:
            initial = model.encode_state(structured, typed_nodes)
            first = model.advance(initial, candidates[selected])
            outcome_logits = model.outcome_head(first)
            action_loss = torch.zeros((), device=device)
            active_horizons = (1,) if arm == "fns_bwm_onestep" else horizons
            for horizon in active_horizons:
                surface = arrays["horizons"][horizon]
                keep = np.asarray(
                    [events[int(index)]["split"] == "training" for index in surface["starts"]],
                    dtype=bool,
                )
                starts_np = surface["starts"][keep]
                paths_np = surface["action_paths"][keep]
                targets_np = surface["targets"][keep]
                starts = torch.as_tensor(starts_np, dtype=torch.long, device=device)
                paths = torch.as_tensor(paths_np, dtype=torch.long, device=device)
                targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
                belief = initial[starts]
                for step in range(horizon):
                    belief = model.advance(belief, candidates[paths[:, step]])
                horizon_logits = model.score_candidates(belief, candidates)
                legal_final = torch.as_tensor(
                    surface["legal_paths"][keep, -1], dtype=torch.bool, device=device
                )
                horizon_logits = horizon_logits.masked_fill(
                    ~legal_final, torch.finfo(horizon_logits.dtype).min
                )
                per_action = F.cross_entropy(
                    horizon_logits, targets, reduction="none"
                )
                weights = torch.as_tensor(
                    _task_balanced_weights([events[int(index)] for index in starts_np]),
                    dtype=torch.float32,
                    device=device,
                )
                horizon_loss = (per_action * weights).sum() / weights.sum()
                action_loss = action_loss + horizon_weights[horizon] * horizon_loss
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        if epoch in (0, int(training["fixed_epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.detach().cpu()),
                    "action_loss": float(action_loss.detach().cpu()),
                    "outcome_loss": float(outcome_loss.detach().cpu()),
                }
            )
    train_prior = np.clip(
        arrays["outcomes"][training_indices].mean(axis=0), 1e-6, 1.0 - 1e-6
    )
    return model, {
        "training_rows": len(training_indices),
        "training_tasks": len({events[int(index)]["task_key"] for index in training_indices}),
        "training_outcome_prior": dict(zip(OBSERVED_OUTCOME_TARGETS, map(float, train_prior))),
        "outcome_positive_weight": dict(zip(OBSERVED_OUTCOME_TARGETS, map(float, positive_weight))),
        "loss_history_endpoints": history,
    }


def _binary_bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))


def _evaluate(
    *,
    model: torch.nn.Module,
    arm: str,
    events: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    fold: int,
    seed: int,
    device: str,
    maximum_horizon: int,
) -> list[dict[str, Any]]:
    model.eval()
    structured = torch.as_tensor(
        arrays["structured_states"], dtype=torch.float32, device=device
    )
    typed_nodes = torch.as_tensor(arrays["typed_nodes"], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    selected = torch.as_tensor(arrays["selected"], dtype=torch.long, device=device)
    legal = torch.as_tensor(arrays["legal"], dtype=torch.bool, device=device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        if arm == "structured_mlp":
            logits, outcome_logits = model(structured, candidates[selected], candidates)
            one_step_probabilities = masked_action_probabilities(logits, legal).cpu().numpy()
        else:
            initial = model.encode_state(structured, typed_nodes)
            first = model.advance(initial, candidates[selected])
            outcome_logits = model.outcome_head(first)
            one_step_probabilities = None
        outcome_probabilities = torch.sigmoid(outcome_logits).cpu().numpy()

        prior = np.asarray(
            [diagnostics["training_outcome_prior"][name] for name in OBSERVED_OUTCOME_TARGETS],
            dtype=np.float32,
        )
        for index, event in enumerate(events):
            if event["split"] != "confirmation":
                continue
            target = arrays["outcomes"][index]
            bce = _binary_bce(outcome_probabilities[index], target)
            prior_bce = _binary_bce(prior, target)
            rows.append(
                {
                    "kind": "outcome",
                    "fold": fold,
                    "arm": arm,
                    "training_seed": seed,
                    "event_id": event["event_id"],
                    "task_name": event["task_name"],
                    "trajectory_id": event["trajectory_id"],
                    "step_id": event["step_id"],
                    "outcome_bce": float(np.mean(bce)),
                    "outcome_brier": float(np.mean((outcome_probabilities[index] - target) ** 2)),
                    "outcome_prior_bce": float(np.mean(prior_bce)),
                    "execution_error_bce": float(bce[0]),
                    "execution_error_prior_bce": float(prior_bce[0]),
                }
            )

        rollout_horizons = (1,) if arm == "structured_mlp" else range(1, maximum_horizon + 1)
        for horizon in rollout_horizons:
            surface = arrays["horizons"][horizon]
            keep = np.asarray(
                [events[int(index)]["split"] == "confirmation" for index in surface["starts"]],
                dtype=bool,
            )
            starts_np = surface["starts"][keep]
            targets_np = surface["targets"][keep]
            legal_paths_np = surface["legal_paths"][keep]
            if arm == "structured_mlp":
                probabilities = one_step_probabilities[starts_np]
            else:
                starts = torch.as_tensor(starts_np, dtype=torch.long, device=device)
                belief = initial[starts]
                first_actions = torch.as_tensor(
                    surface["action_paths"][keep, 0], dtype=torch.long, device=device
                )
                action_input = candidates[first_actions]
                probabilities_tensor = None
                for step in range(horizon):
                    belief = model.advance(belief, action_input)
                    horizon_logits = model.score_candidates(belief, candidates)
                    legal_step = torch.as_tensor(
                        legal_paths_np[:, step], dtype=torch.bool, device=device
                    )
                    probabilities_tensor = masked_action_probabilities(
                        horizon_logits, legal_step
                    )
                    action_input = probabilities_tensor @ candidates
                probabilities = probabilities_tensor.cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            for local, start in enumerate(starts_np):
                event = events[int(start)]
                target = int(targets_np[local])
                legal_final = legal_paths_np[local, -1]
                target_probability = max(float(probabilities[local, target]), 1e-12)
                legal_probabilities = probabilities[local, legal_final]
                rows.append(
                    {
                        "kind": "rollout",
                        "fold": fold,
                        "arm": arm,
                        "training_seed": seed,
                        "horizon": horizon,
                        "event_id": event["event_id"],
                        "task_name": event["task_name"],
                        "trajectory_id": event["trajectory_id"],
                        "step_id": event["step_id"],
                        "action_nll": float(-math.log(target_probability)),
                        "action_correct": float(int(predicted[local]) == target),
                        "action_brier": float(
                            np.mean(
                                (
                                    legal_probabilities
                                    - legal_final.nonzero()[0].astype(np.int64).__eq__(target).astype(np.float32)
                                )
                                ** 2
                            )
                        ),
                        "predictive_entropy": float(
                            -np.sum(
                                legal_probabilities
                                * np.log(np.clip(legal_probabilities, 1e-12, 1.0))
                            )
                        ),
                        "legal_prediction": float(bool(legal_final[int(predicted[local])])),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_and_frozen_before_training":
        raise ValueError("v4 pilot protocol is not frozen")
    frozen = protocol["frozen_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("audit hash mismatch")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("frozen adjacent-transition audit failed")
    if tuple(protocol["training"]["arms"]) != ARMS:
        raise ValueError("arm surface differs from frozen protocol")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]
    expected_runs = len(dataset["folds"]) * len(ARMS) * len(seeds)
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
        arrays = _arrays(
            events,
            dataset["candidate_catalog"],
            state_hash_dimension=int(protocol["training"]["state_hash_dimension"]),
            typed_hash_dimension=int(protocol["training"]["typed_hash_dimension"]),
            maximum_horizon=int(protocol["training"]["maximum_horizon"]),
        )
        for arm in ARMS:
            for seed in seeds:
                _set_seed(seed)
                model, diagnostics = _train(
                    arm=arm,
                    events=events,
                    arrays=arrays,
                    protocol=protocol,
                    seed=seed,
                    device=device,
                )
                predictions = _evaluate(
                    model=model,
                    arm=arm,
                    events=events,
                    arrays=arrays,
                    diagnostics=diagnostics,
                    fold=fold,
                    seed=seed,
                    device=device,
                    maximum_horizon=int(protocol["training"]["maximum_horizon"]),
                )
                _append_jsonl(prediction_path, predictions)
                runs.append(
                    {
                        "fold": fold,
                        "arm": arm,
                        "training_seed": seed,
                        "prediction_rows": len(predictions),
                        **diagnostics,
                    }
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
