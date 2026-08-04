"""Train the frozen three-representation Markov-sufficiency comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from wmagentattack.hybrid_semantic_world_model import (
    EVIDENCE_DELTA_TARGETS,
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
    evidence_delta_target,
    tool_candidate_vector,
)
from wmagentattack.markov_sufficiency import (
    FROZEN_SUFFICIENCY_VARIANTS,
    representation_feature_vector,
    validate_dataset_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_balanced_weights(task_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(task_ids)
    if not counts:
        raise ValueError("cannot weight an empty row set")
    scale = len(task_ids) / len(counts)
    return np.asarray(
        [scale / counts[task_id] for task_id in task_ids], dtype=np.float32
    )


def _f1(target: set[str], predicted: set[str]) -> float:
    if not target and not predicted:
        return 1.0
    if not target or not predicted:
        return 0.0
    overlap = len(target & predicted)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def _task_macro(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            grouped[str(row["task_id"])].append(float(row[key]))
    if not grouped:
        return None
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _summarize(
    dynamics: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = {}
    for split in ("training", "calibration", "confirmation"):
        dynamic_rows = [row for row in dynamics if row["split"] == split]
        evidence_rows = [row for row in evidence if row["split"] == split]
        split_metrics = {
            "tasks": len({row["task_id"] for row in dynamic_rows}),
            "prefixes": len(dynamic_rows),
            "evidence_transitions": len(evidence_rows),
            "task_macro_action_nll": _task_macro(dynamic_rows, "action_nll"),
            "task_macro_action_accuracy": _task_macro(
                dynamic_rows, "action_correct"
            ),
            "task_macro_stop_accuracy": _task_macro(dynamic_rows, "stop_correct"),
            "task_macro_argument_key_f1": _task_macro(
                dynamic_rows, "argument_key_f1"
            ),
            "task_macro_evidence_bce": _task_macro(evidence_rows, "evidence_bce"),
            "task_macro_evidence_brier": _task_macro(
                evidence_rows, "evidence_brier"
            ),
        }
        split_metrics["evidence_targets"] = {
            name: {
                "positives": int(sum(row[f"target_{name}"] for row in evidence_rows)),
                "task_macro_bce": _task_macro(evidence_rows, f"bce_{name}"),
                "task_macro_brier": _task_macro(evidence_rows, f"brier_{name}"),
            }
            for name in EVIDENCE_DELTA_TARGETS
        }
        output[split] = split_metrics
    return output


def _flatten(
    source: Mapping[str, Any], semantic: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefixes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for source_episode, semantic_episode in zip(
        source["episodes"], semantic["episodes"]
    ):
        source_prefixes = source_episode["prefixes"]
        semantic_prefixes = semantic_episode["prefixes"]
        episode_row_indices = []
        for index, (source_prefix, semantic_prefix) in enumerate(
            zip(source_prefixes, semantic_prefixes)
        ):
            episode_row_indices.append(len(prefixes))
            prefixes.append(
                {
                    "row_id": f"{source_episode['episode_id']}::p{index}",
                    "episode_id": source_episode["episode_id"],
                    "task_id": source_episode["task_id"],
                    "suite": source_episode["suite"],
                    "split": source_episode["split"],
                    "track": source_episode["track"],
                    "prefix_index": index,
                    "source_prefixes": source_prefixes,
                    "semantic_prefixes": semantic_prefixes,
                    "source_prefix": source_prefix,
                    "semantic_prefix": semantic_prefix,
                }
            )
        for index, row_index in enumerate(episode_row_indices[:-1]):
            action = str(source_prefixes[index]["targets"]["next_action"])
            if action == "STOP":
                raise ValueError("STOP occurs before an observed next prefix")
            current = semantic_prefixes[index]["features"]["semantic_state_v3"]
            following = semantic_prefixes[index + 1]["features"]["semantic_state_v3"]
            transitions.append(
                {
                    "row_id": f"{source_episode['episode_id']}::t{index}",
                    "prefix_row_index": row_index,
                    "episode_id": source_episode["episode_id"],
                    "task_id": source_episode["task_id"],
                    "suite": source_episode["suite"],
                    "split": source_episode["split"],
                    "action": action,
                    "target": evidence_delta_target(current, following),
                }
            )
        if str(source_prefixes[-1]["targets"]["next_action"]) != "STOP":
            raise ValueError("final prefix is not a STOP target")
    return prefixes, transitions


def _train_one(
    *,
    model: HybridSemanticWorldModel,
    state_inputs: np.ndarray,
    candidate_inputs: np.ndarray,
    legal_masks: np.ndarray,
    action_targets: np.ndarray,
    argument_targets: np.ndarray,
    transitions: Sequence[Mapping[str, Any]],
    candidate_index: Mapping[str, int],
    prefixes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> list[dict[str, float]]:
    train_prefix_indices = np.asarray(
        [index for index, row in enumerate(prefixes) if row["split"] == "training"],
        dtype=np.int64,
    )
    train_transition_indices = np.asarray(
        [
            index
            for index, row in enumerate(transitions)
            if row["split"] == "training"
        ],
        dtype=np.int64,
    )
    prefix_x = torch.as_tensor(
        state_inputs[train_prefix_indices], dtype=torch.float32, device=device
    )
    candidates = torch.as_tensor(candidate_inputs, dtype=torch.float32, device=device)
    legal = torch.as_tensor(
        legal_masks[train_prefix_indices], dtype=torch.bool, device=device
    )
    actions = torch.as_tensor(
        action_targets[train_prefix_indices], dtype=torch.long, device=device
    )
    arguments = torch.as_tensor(
        argument_targets[train_prefix_indices], dtype=torch.float32, device=device
    )
    prefix_weights = torch.as_tensor(
        _task_balanced_weights(
            [prefixes[index]["task_id"] for index in train_prefix_indices]
        ),
        dtype=torch.float32,
        device=device,
    )

    evidence_prefix_rows = np.asarray(
        [
            int(transitions[index]["prefix_row_index"])
            for index in train_transition_indices
        ],
        dtype=np.int64,
    )
    evidence_local_rows = np.asarray(
        [
            int(np.where(train_prefix_indices == global_index)[0][0])
            for global_index in evidence_prefix_rows
        ],
        dtype=np.int64,
    )
    evidence_candidates = torch.as_tensor(
        [
            candidate_index[str(transitions[index]["action"])]
            for index in train_transition_indices
        ],
        dtype=torch.long,
        device=device,
    )
    evidence_targets = torch.as_tensor(
        np.stack([transitions[index]["target"] for index in train_transition_indices]),
        dtype=torch.float32,
        device=device,
    )
    evidence_weights = torch.as_tensor(
        _task_balanced_weights(
            [transitions[index]["task_id"] for index in train_transition_indices]
        ),
        dtype=torch.float32,
        device=device,
    )
    positives = evidence_targets.sum(dim=0)
    negatives = evidence_targets.shape[0] - positives
    positive_weight = torch.where(
        positives > 0,
        torch.clamp(
            negatives / torch.clamp(positives, min=1.0),
            1.0,
            float(protocol["training"]["evidence_positive_weight_cap"]),
        ),
        torch.ones_like(positives),
    )

    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for epoch in range(int(training["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        action_logits, argument_logits, evidence_logits = model(
            prefix_x, candidates
        )
        masked_action = action_logits.masked_fill(
            ~legal, torch.finfo(action_logits.dtype).min
        )
        per_action = F.cross_entropy(masked_action, actions, reduction="none")
        action_loss = (per_action * prefix_weights).sum() / prefix_weights.sum()
        per_argument = F.binary_cross_entropy_with_logits(
            argument_logits, arguments, reduction="none"
        ).mean(dim=1)
        argument_loss = (
            per_argument * prefix_weights
        ).sum() / prefix_weights.sum()
        selected_evidence = evidence_logits[
            torch.as_tensor(evidence_local_rows, dtype=torch.long, device=device),
            evidence_candidates,
        ]
        per_evidence = F.binary_cross_entropy_with_logits(
            selected_evidence,
            evidence_targets,
            reduction="none",
            pos_weight=positive_weight,
        ).mean(dim=1)
        evidence_loss = (
            per_evidence * evidence_weights
        ).sum() / evidence_weights.sum()
        loss = (
            action_loss
            + float(training["argument_loss_weight"]) * argument_loss
            + float(training["evidence_loss_weight"]) * evidence_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(
            {
                "total": float(loss.detach().cpu()),
                "action": float(action_loss.detach().cpu()),
                "argument": float(argument_loss.detach().cpu()),
                "evidence": float(evidence_loss.detach().cpu()),
            }
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--semantic-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = json.loads(args.source_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_training":
        raise ValueError("Stage 3 protocol was not frozen before training")
    if tuple(protocol["frozen_variants"]) != FROZEN_SUFFICIENCY_VARIANTS:
        raise ValueError("Stage 3 representation set differs from the protocol")
    if [int(seed) for seed in protocol["training"]["training_seeds"]] != [7, 17, 29]:
        raise ValueError("Stage 3 training seeds differ from the frozen set")
    if _sha256(args.source_dataset) != protocol["source"]["raw_dataset_sha256"]:
        raise ValueError("raw source dataset hash mismatch")
    if _sha256(args.semantic_dataset) != protocol["source"]["semantic_dataset_sha256"]:
        raise ValueError("semantic source dataset hash mismatch")
    validate_dataset_alignment(source, semantic)
    prefixes, transitions = _flatten(source, semantic)
    budget = protocol["fixed_budget"]
    if len(prefixes) != int(budget["prefixes"]):
        raise ValueError("prefix budget mismatch")
    if len(transitions) != int(budget["evidence_transitions"]):
        raise ValueError("transition budget mismatch")

    candidates = sorted(source["tool_catalog"])
    if candidates != sorted(semantic["tool_catalog"]):
        raise ValueError("tool catalogs differ")
    candidate_index = {name: index for index, name in enumerate(candidates)}
    argument_keys = list(source["argument_key_vocab"])
    if argument_keys != list(semantic["argument_key_vocab"]):
        raise ValueError("argument vocabularies differ")
    argument_index = {name: index for index, name in enumerate(argument_keys)}
    hash_dimension = int(protocol["training"]["hash_dimension"])
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                source["tool_catalog"][candidate], hash_dimension=hash_dimension
            )
            for candidate in candidates
        ]
    )
    legal_masks = np.zeros((len(prefixes), len(candidates)), dtype=bool)
    action_targets = np.zeros(len(prefixes), dtype=np.int64)
    argument_targets = np.zeros(
        (len(prefixes), len(argument_keys)), dtype=np.float32
    )
    for index, row in enumerate(prefixes):
        state = row["semantic_prefix"]["features"]["semantic_state_v3"]
        for candidate in state["legal_actions"]:
            legal_masks[index, candidate_index[candidate]] = True
        target = str(row["source_prefix"]["targets"]["next_action"])
        action_targets[index] = candidate_index[target]
        if not legal_masks[index, action_targets[index]]:
            raise ValueError("target action is not legal")
        for key in row["source_prefix"]["targets"]["argument_keys"]:
            argument_targets[index, argument_index[key]] = 1.0

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(int(protocol["training"].get("cpu_threads", 8)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    run_metrics = []

    for variant in FROZEN_SUFFICIENCY_VARIANTS:
        state_inputs = np.stack(
            [
                representation_feature_vector(
                    variant=variant,
                    source_prefixes=row["source_prefixes"],
                    semantic_prefixes=row["semantic_prefixes"],
                    prefix_index=int(row["prefix_index"]),
                    hash_dimension=hash_dimension,
                )
                for row in prefixes
            ]
        )
        for seed_value in protocol["training"]["training_seeds"]:
            seed = int(seed_value)
            _set_seed(seed)
            model = HybridSemanticWorldModel(
                state_size=state_inputs.shape[1],
                candidate_size=candidate_inputs.shape[1],
                argument_keys=len(argument_keys),
                hidden_size=int(protocol["training"]["hidden_size"]),
                dropout=float(protocol["training"]["dropout"]),
            ).to(device)
            assert_no_planning_or_value_heads(model)
            history = _train_one(
                model=model,
                state_inputs=state_inputs,
                candidate_inputs=candidate_inputs,
                legal_masks=legal_masks,
                action_targets=action_targets,
                argument_targets=argument_targets,
                transitions=transitions,
                candidate_index=candidate_index,
                prefixes=prefixes,
                protocol=protocol,
                seed=seed,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                x = torch.as_tensor(state_inputs, dtype=torch.float32, device=device)
                c = torch.as_tensor(candidate_inputs, dtype=torch.float32, device=device)
                legal = torch.as_tensor(legal_masks, dtype=torch.bool, device=device)
                action_logits, argument_logits, evidence_logits = model(x, c)
                action_probabilities = model.action_probabilities(x, c, legal).cpu().numpy()
                argument_probabilities = torch.sigmoid(argument_logits).cpu().numpy()
                evidence_probabilities = torch.sigmoid(evidence_logits).cpu().numpy()

            dynamic_predictions = []
            for index, row in enumerate(prefixes):
                target = int(action_targets[index])
                predicted = int(np.argmax(action_probabilities[index]))
                target_keys = set(row["source_prefix"]["targets"]["argument_keys"])
                predicted_keys = {
                    key
                    for key, probability in zip(
                        argument_keys, argument_probabilities[index]
                    )
                    if probability >= 0.5
                }
                target_stop = candidates[target] == "STOP"
                predicted_stop = candidates[predicted] == "STOP"
                dynamic_predictions.append(
                    {
                        "prediction_type": "dynamics",
                        "variant": variant,
                        "training_seed": seed,
                        "row_id": row["row_id"],
                        "episode_id": row["episode_id"],
                        "task_id": row["task_id"],
                        "suite": row["suite"],
                        "split": row["split"],
                        "prefix_index": row["prefix_index"],
                        "target_action": candidates[target],
                        "predicted_action": candidates[predicted],
                        "target_action_probability": float(
                            action_probabilities[index, target]
                        ),
                        "action_nll": float(
                            -math.log(max(action_probabilities[index, target], 1e-12))
                        ),
                        "action_correct": float(predicted == target),
                        "stop_correct": float(predicted_stop == target_stop),
                        "argument_key_f1": _f1(target_keys, predicted_keys),
                    }
                )

            evidence_predictions = []
            for transition in transitions:
                row_index = int(transition["prefix_row_index"])
                candidate = candidate_index[str(transition["action"])]
                target = np.asarray(transition["target"], dtype=np.float64)
                probability = np.clip(
                    evidence_probabilities[row_index, candidate].astype(np.float64),
                    1e-12,
                    1.0 - 1e-12,
                )
                bce = -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
                brier = (probability - target) ** 2
                result = {
                    "prediction_type": "evidence",
                    "variant": variant,
                    "training_seed": seed,
                    "row_id": transition["row_id"],
                    "episode_id": transition["episode_id"],
                    "task_id": transition["task_id"],
                    "suite": transition["suite"],
                    "split": transition["split"],
                    "executed_action": transition["action"],
                    "evidence_bce": float(np.mean(bce)),
                    "evidence_brier": float(np.mean(brier)),
                }
                for target_index, name in enumerate(EVIDENCE_DELTA_TARGETS):
                    result[f"target_{name}"] = float(target[target_index])
                    result[f"probability_{name}"] = float(probability[target_index])
                    result[f"bce_{name}"] = float(bce[target_index])
                    result[f"brier_{name}"] = float(brier[target_index])
                evidence_predictions.append(result)

            with predictions_path.open("a", encoding="utf-8") as handle:
                for result in (*dynamic_predictions, *evidence_predictions):
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            torch.save(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "state_dict": model.state_dict(),
                    "candidate_ids": candidates,
                    "argument_keys": argument_keys,
                    "evidence_delta_targets": EVIDENCE_DELTA_TARGETS,
                },
                checkpoints / f"{variant}_seed{seed}.pt",
            )
            metrics = _summarize(dynamic_predictions, evidence_predictions)
            run_metrics.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "device": device,
                    "parameter_count": sum(
                        int(parameter.numel()) for parameter in model.parameters()
                    ),
                    "final_training_loss": history[-1],
                    "metrics": metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "final_training_loss": history[-1],
                        "confirmation": metrics["confirmation"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    output = {
        "protocol_sha256": _sha256(args.protocol),
        "raw_dataset_sha256": _sha256(args.source_dataset),
        "semantic_dataset_sha256": _sha256(args.semantic_dataset),
        "variants": list(FROZEN_SUFFICIENCY_VARIANTS),
        "training_seeds": [
            int(seed) for seed in protocol["training"]["training_seeds"]
        ],
        "prefixes": len(prefixes),
        "evidence_transitions": len(transitions),
        "candidate_count": len(candidates),
        "argument_key_count": len(argument_keys),
        "runs": run_metrics,
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("MARKOV_SUFFICIENCY_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
