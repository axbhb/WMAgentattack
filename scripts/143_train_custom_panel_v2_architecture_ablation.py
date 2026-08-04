"""Train the preregistered three-arm clean dynamics/evidence probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.panel_v2_architecture_probe import (
    EVIDENCE_PROGRESS_STATUSES,
    FROZEN_ARCHITECTURE_VARIANTS,
    CandidateDynamicsProbe,
    EvidenceProgressProbe,
    candidate_descriptor_vector,
    obligation_descriptor_vector,
    prefix_feature_vector,
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


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def _task_balanced_weights(task_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(task_ids)
    if not counts:
        raise ValueError("cannot weight empty rows")
    scale = len(task_ids) / len(counts)
    return np.asarray([scale / counts[task_id] for task_id in task_ids], dtype=np.float32)


def _flatten(dataset: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefixes = []
    evidence = []
    for episode in dataset["episodes"]:
        for prefix in episode["prefixes"]:
            index = len(prefixes)
            row = {
                "row_id": f"{episode['episode_id']}::p{prefix['prefix_index']}",
                "episode_id": episode["episode_id"],
                "task_id": episode["task_id"],
                "suite": episode["suite"],
                "split": episode["split"],
                "track": episode["track"],
                "prefix_index": int(prefix["prefix_index"]),
                "prefix": prefix,
            }
            prefixes.append(row)
            for obligation in prefix["targets"]["evidence_obligations"]:
                evidence.append(
                    {
                        **{key: value for key, value in row.items() if key != "prefix"},
                        "prefix_row_index": index,
                        "obligation": obligation,
                    }
                )
    return prefixes, evidence


def _mask_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def _f1(target: set[str], predicted: set[str]) -> float:
    if not target and not predicted:
        return 1.0
    if not target or not predicted:
        return 0.0
    overlap = len(target & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def _task_macro(rows: list[dict[str, Any]], value: str) -> float | None:
    grouped = defaultdict(list)
    for row in rows:
        if row.get(value) is not None:
            grouped[row["task_id"]].append(float(row[value]))
    if not grouped:
        return None
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _summarize_predictions(
    dynamics: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    output = {}
    for split in ("training", "calibration", "confirmation"):
        dynamic_rows = [row for row in dynamics if row["split"] == split]
        evidence_rows = [row for row in evidence if row["split"] == split]
        error_rows = [row for row in dynamic_rows if row["previous_execution_error"]]
        output[split] = {
            "tasks": len({row["task_id"] for row in dynamic_rows}),
            "prefixes": len(dynamic_rows),
            "evidence_rows": len(evidence_rows),
            "task_macro_action_nll": _task_macro(dynamic_rows, "action_nll"),
            "task_macro_action_accuracy": _task_macro(dynamic_rows, "action_correct"),
            "task_macro_stop_accuracy": _task_macro(dynamic_rows, "stop_correct"),
            "task_macro_argument_key_f1": _task_macro(dynamic_rows, "argument_key_f1"),
            "task_macro_error_recovery_action_nll": _task_macro(error_rows, "action_nll"),
            "error_recovery_prefixes": len(error_rows),
            "task_macro_evidence_status_nll": _task_macro(evidence_rows, "status_nll"),
            "task_macro_evidence_status_accuracy": _task_macro(evidence_rows, "status_correct"),
            "task_macro_supported_brier": _task_macro(evidence_rows, "supported_brier"),
        }
    return output


def _train_dynamics(
    *,
    model: CandidateDynamicsProbe,
    prefix_inputs: np.ndarray,
    candidate_inputs: np.ndarray,
    legal_masks: np.ndarray,
    action_targets: np.ndarray,
    argument_targets: np.ndarray,
    train_indices: np.ndarray,
    train_task_ids: list[str],
    config: Mapping[str, Any],
    seed: int,
    device: str,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.as_tensor(prefix_inputs[train_indices], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(candidate_inputs, dtype=torch.float32, device=device)
    masks = torch.as_tensor(legal_masks[train_indices], dtype=torch.bool, device=device)
    targets = torch.as_tensor(action_targets[train_indices], dtype=torch.long, device=device)
    arguments = torch.as_tensor(argument_targets[train_indices], dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        _task_balanced_weights(train_task_ids), dtype=torch.float32, device=device
    )
    losses = []
    for epoch in range(int(config["fixed_epochs_dynamics"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        logits, argument_logits = model(x, candidates)
        logits = _mask_logits(logits, masks)
        action = F.cross_entropy(logits, targets, reduction="none")
        argument = F.binary_cross_entropy_with_logits(
            argument_logits, arguments, reduction="none"
        ).mean(dim=1)
        loss = (
            (action + float(config["argument_loss_weight"]) * argument) * weights
        ).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _train_evidence(
    *,
    model: EvidenceProgressProbe,
    evidence_inputs: np.ndarray,
    evidence_targets: np.ndarray,
    train_indices: np.ndarray,
    train_task_ids: list[str],
    config: Mapping[str, Any],
    seed: int,
    device: str,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.as_tensor(evidence_inputs[train_indices], dtype=torch.float32, device=device)
    targets = torch.as_tensor(evidence_targets[train_indices], dtype=torch.long, device=device)
    weights = torch.as_tensor(
        _task_balanced_weights(train_task_ids), dtype=torch.float32, device=device
    )
    losses = []
    for epoch in range(int(config["fixed_epochs_evidence"])):
        _set_seed(seed * 2003 + epoch)
        model.train()
        logits = model(x)
        per_row = F.cross_entropy(logits, targets, reduction="none")
        loss = (per_row * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_training":
        raise ValueError("architecture protocol was not frozen before training")
    if tuple(protocol["frozen_variants"]) != FROZEN_ARCHITECTURE_VARIANTS:
        raise ValueError("implementation variants differ from protocol")
    if _sha256(args.dataset) != protocol["source"]["dataset_sha256"]:
        raise ValueError("dataset hash differs from frozen protocol")
    if len(dataset["episodes"]) != protocol["fixed_budget"]["episodes"]:
        raise ValueError("dataset episode budget mismatch")

    prefixes, evidence_rows = _flatten(dataset)
    if len(prefixes) != protocol["fixed_budget"]["prefixes"]:
        raise ValueError("dataset prefix budget mismatch")
    candidates = sorted(dataset["tool_catalog"])
    candidate_index = {name: index for index, name in enumerate(candidates)}
    argument_keys = list(dataset["argument_key_vocab"])
    argument_index = {name: index for index, name in enumerate(argument_keys)}
    unknown_argument_keys = sorted(
        {
            key
            for row in prefixes
            for key in row["prefix"]["targets"]["argument_keys"]
            if key not in argument_index
        }
    )
    if unknown_argument_keys:
        raise ValueError(
            "argument targets are outside the frozen schema vocabulary: "
            f"{unknown_argument_keys}"
        )
    hash_dimension = int(protocol["training"]["hash_dimension"])
    candidate_inputs = np.stack(
        [
            candidate_descriptor_vector(
                dataset["tool_catalog"][candidate], hash_dimension=hash_dimension
            )
            for candidate in candidates
        ]
    )
    legal_masks = np.zeros((len(prefixes), len(candidates)), dtype=bool)
    action_targets = np.zeros(len(prefixes), dtype=np.int64)
    argument_targets = np.zeros((len(prefixes), len(argument_keys)), dtype=np.float32)
    for index, row in enumerate(prefixes):
        for candidate in row["prefix"]["features"]["legal_tools"]:
            legal_masks[index, candidate_index[candidate]] = True
        action = row["prefix"]["targets"]["next_action"]
        action_targets[index] = candidate_index[action]
        for key in row["prefix"]["targets"]["argument_keys"]:
            argument_targets[index, argument_index[key]] = 1.0
        if not legal_masks[index, action_targets[index]]:
            raise ValueError("target action is not legal")
    status_index = {name: index for index, name in enumerate(EVIDENCE_PROGRESS_STATUSES)}
    evidence_targets = np.asarray(
        [status_index[row["obligation"]["status"]] for row in evidence_rows],
        dtype=np.int64,
    )

    device = _resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(int(protocol["training"].get("cpu_threads", 8)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    run_metrics = []

    for variant in FROZEN_ARCHITECTURE_VARIANTS:
        prefix_inputs = np.stack(
            [
                prefix_feature_vector(
                    row["prefix"],
                    variant=variant,
                    hash_dimension=hash_dimension,
                )
                for row in prefixes
            ]
        )
        evidence_inputs = np.stack(
            [
                np.concatenate(
                    (
                        prefix_inputs[row["prefix_row_index"]],
                        obligation_descriptor_vector(
                            prefixes[row["prefix_row_index"]]["prefix"],
                            row["obligation"],
                            hash_dimension=hash_dimension,
                        ),
                    )
                )
                for row in evidence_rows
            ]
        )
        train_prefix = np.asarray(
            [index for index, row in enumerate(prefixes) if row["split"] == "training"],
            dtype=np.int64,
        )
        train_evidence = np.asarray(
            [index for index, row in enumerate(evidence_rows) if row["split"] == "training"],
            dtype=np.int64,
        )
        for seed in protocol["training"]["training_seeds"]:
            seed = int(seed)
            _set_seed(seed)
            dynamics_model = CandidateDynamicsProbe(
                prefix_size=prefix_inputs.shape[1],
                candidate_size=candidate_inputs.shape[1],
                argument_keys=len(argument_keys),
                hidden_size=int(protocol["training"]["hidden_size"]),
                dropout=float(protocol["training"]["dropout"]),
            )
            dynamics_losses = _train_dynamics(
                model=dynamics_model,
                prefix_inputs=prefix_inputs,
                candidate_inputs=candidate_inputs,
                legal_masks=legal_masks,
                action_targets=action_targets,
                argument_targets=argument_targets,
                train_indices=train_prefix,
                train_task_ids=[prefixes[index]["task_id"] for index in train_prefix],
                config=protocol["training"],
                seed=seed,
                device=device,
            )
            _set_seed(seed + 100000)
            evidence_model = EvidenceProgressProbe(
                input_size=evidence_inputs.shape[1],
                hidden_size=int(protocol["training"]["hidden_size"]),
                dropout=float(protocol["training"]["dropout"]),
            )
            evidence_losses = _train_evidence(
                model=evidence_model,
                evidence_inputs=evidence_inputs,
                evidence_targets=evidence_targets,
                train_indices=train_evidence,
                train_task_ids=[evidence_rows[index]["task_id"] for index in train_evidence],
                config=protocol["training"],
                seed=seed,
                device=device,
            )

            dynamics_model.eval()
            evidence_model.eval()
            with torch.no_grad():
                px = torch.as_tensor(prefix_inputs, dtype=torch.float32, device=device)
                cx = torch.as_tensor(candidate_inputs, dtype=torch.float32, device=device)
                masks = torch.as_tensor(legal_masks, dtype=torch.bool, device=device)
                action_logits, argument_logits = dynamics_model(px, cx)
                action_probabilities = torch.softmax(
                    _mask_logits(action_logits, masks), dim=1
                ).cpu().numpy()
                argument_probabilities = torch.sigmoid(argument_logits).cpu().numpy()
                ex = torch.as_tensor(evidence_inputs, dtype=torch.float32, device=device)
                evidence_probabilities = torch.softmax(evidence_model(ex), dim=1).cpu().numpy()

            dynamic_predictions = []
            for index, row in enumerate(prefixes):
                target = int(action_targets[index])
                predicted = int(np.argmax(action_probabilities[index]))
                target_keys = set(row["prefix"]["targets"]["argument_keys"])
                predicted_keys = {
                    key
                    for key, probability in zip(argument_keys, argument_probabilities[index])
                    if probability >= 0.5
                }
                target_stop = candidates[target] == "STOP"
                predicted_stop = candidates[predicted] == "STOP"
                result = {
                    "prediction_type": "dynamics",
                    "variant": variant,
                    "training_seed": seed,
                    **{key: value for key, value in row.items() if key != "prefix"},
                    "target_action": candidates[target],
                    "predicted_action": candidates[predicted],
                    "target_action_probability": float(action_probabilities[index, target]),
                    "action_nll": float(-math.log(max(action_probabilities[index, target], 1e-12))),
                    "action_correct": float(predicted == target),
                    "stop_probability": float(action_probabilities[index, candidate_index["STOP"]]),
                    "stop_correct": float(predicted_stop == target_stop),
                    "target_argument_keys": sorted(target_keys),
                    "predicted_argument_keys": sorted(predicted_keys),
                    "argument_key_f1": _f1(target_keys, predicted_keys),
                    "previous_execution_error": row["prefix"]["features"]["execution_receipt"].get("status") == "error",
                }
                dynamic_predictions.append(result)

            evidence_predictions = []
            supported_index = status_index["SUPPORTED"]
            for index, row in enumerate(evidence_rows):
                target = int(evidence_targets[index])
                predicted = int(np.argmax(evidence_probabilities[index]))
                supported_target = float(target == supported_index)
                supported_probability = float(evidence_probabilities[index, supported_index])
                evidence_predictions.append(
                    {
                        "prediction_type": "evidence",
                        "variant": variant,
                        "training_seed": seed,
                        **{key: value for key, value in row.items() if key not in {"obligation", "prefix_row_index"}},
                        "obligation_id": row["obligation"]["obligation_id"],
                        "target_status": EVIDENCE_PROGRESS_STATUSES[target],
                        "predicted_status": EVIDENCE_PROGRESS_STATUSES[predicted],
                        "target_status_probability": float(evidence_probabilities[index, target]),
                        "status_nll": float(-math.log(max(evidence_probabilities[index, target], 1e-12))),
                        "status_correct": float(predicted == target),
                        "supported_probability": supported_probability,
                        "supported_brier": float((supported_probability - supported_target) ** 2),
                    }
                )

            with predictions_path.open("a", encoding="utf-8") as handle:
                for result in (*dynamic_predictions, *evidence_predictions):
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            torch.save(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "dynamics": dynamics_model.state_dict(),
                    "evidence": evidence_model.state_dict(),
                    "candidate_ids": candidates,
                    "argument_keys": argument_keys,
                },
                checkpoints / f"{variant}_seed{seed}.pt",
            )
            run_metrics.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "device": device,
                    "dynamics_final_loss": dynamics_losses[-1],
                    "evidence_final_loss": evidence_losses[-1],
                    "metrics": _summarize_predictions(
                        dynamic_predictions, evidence_predictions
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "dynamics_final_loss": dynamics_losses[-1],
                        "evidence_final_loss": evidence_losses[-1],
                    }
                ),
                flush=True,
            )

    output = {
        "protocol_sha256": _sha256(args.protocol),
        "dataset_sha256": _sha256(args.dataset),
        "variants": list(FROZEN_ARCHITECTURE_VARIANTS),
        "training_seeds": [int(seed) for seed in protocol["training"]["training_seeds"]],
        "candidate_count": len(candidates),
        "argument_key_count": len(argument_keys),
        "prefixes": len(prefixes),
        "evidence_rows": len(evidence_rows),
        "runs": run_metrics,
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("CUSTOM_PANEL_V2_ARCHITECTURE_TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
