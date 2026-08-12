"""Frozen same-fold comparison of AgentDojo-only and three-source training."""

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

from wmagentattack.hybrid_semantic_world_model import (
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
    tool_candidate_vector,
)
from wmagentattack.multisource_suitability import file_sha256, representation_vector


CONDITIONS = ("agentdojo_only", "agentdojo_plus_auxiliary")
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_task_balanced_weights(
    rows: Sequence[Mapping[str, Any]], source_mass: Mapping[str, float]
) -> np.ndarray:
    """Allocate frozen mass by source, then equally by task within source."""

    source_tasks: dict[str, set[str]] = {}
    task_counts = Counter((str(row["source"]), str(row["task_key"])) for row in rows)
    for row in rows:
        source_tasks.setdefault(str(row["source"]), set()).add(str(row["task_key"]))
    observed = set(source_tasks)
    if observed != set(source_mass):
        raise ValueError(f"source-mass surface differs: {observed} != {set(source_mass)}")
    if not math.isclose(sum(source_mass.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("source masses must sum to one")
    weights = []
    for row in rows:
        source = str(row["source"])
        task = str(row["task_key"])
        weights.append(
            float(source_mass[source])
            / len(source_tasks[source])
            / task_counts[(source, task)]
        )
    values = np.asarray(weights, dtype=np.float32)
    values *= len(values) / float(values.sum())
    return values


def _fold_rows(
    dataset: Mapping[str, Any], *, fold: int, condition: str
) -> list[dict[str, Any]]:
    surface = dataset["folds"][fold]
    train_tasks = set(surface["train_tasks"])
    test_tasks = set(surface["test_tasks"])
    rows: list[dict[str, Any]] = []
    for source_row in dataset["rows"]:
        source = str(source_row["source"])
        task_name = str(source_row["task_name"])
        if source == "agentdojo" and task_name in train_tasks:
            row = dict(source_row)
            row["split"] = "training"
            rows.append(row)
        elif source == "agentdojo" and task_name in test_tasks:
            row = dict(source_row)
            row["split"] = "confirmation"
            rows.append(row)
        elif source != "agentdojo" and condition == "agentdojo_plus_auxiliary":
            row = dict(source_row)
            row["split"] = "training"
            rows.append(row)
    if not rows:
        raise ValueError("empty fold surface")
    training = [row for row in rows if row["split"] == "training"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    if {row["task_name"] for row in confirmation} != test_tasks:
        raise ValueError("confirmation task surface differs from frozen fold")
    if {row["task_name"] for row in training if row["source"] == "agentdojo"} != train_tasks:
        raise ValueError("training task surface differs from frozen fold")
    return rows


def _arrays(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    hash_dimension: int,
) -> dict[str, Any]:
    candidates = sorted(
        {candidate for row in rows for candidate in row["legal_candidate_ids"]}
    )
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    states = np.stack(
        [
            representation_vector(row, variant=variant, hash_dimension=hash_dimension)
            for row in rows
        ]
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(catalog[candidate], hash_dimension=hash_dimension)
            for candidate in candidates
        ]
    )
    legal = np.zeros((len(rows), len(candidates)), dtype=bool)
    targets = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        for candidate in row["legal_candidate_ids"]:
            legal[row_index, candidate_index[candidate]] = True
        targets[row_index] = candidate_index[str(row["target_candidate_id"])]
        if not legal[row_index, targets[row_index]]:
            raise ValueError("target action is not legal")
    return {
        "candidates": candidates,
        "states": states,
        "candidate_inputs": candidate_inputs,
        "legal": legal,
        "targets": targets,
    }


def _train(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_mass: Mapping[str, float],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "training"],
        dtype=np.int64,
    )
    train_rows = [rows[int(index)] for index in train_indices]
    weights_np = source_task_balanced_weights(train_rows, source_mass)
    states = torch.as_tensor(
        arrays["states"][train_indices], dtype=torch.float32, device=device
    )
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    legal = torch.as_tensor(
        arrays["legal"][train_indices], dtype=torch.bool, device=device
    )
    targets = torch.as_tensor(
        arrays["targets"][train_indices], dtype=torch.long, device=device
    )
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    training = protocol["training"]
    model = HybridSemanticWorldModel(
        state_size=int(arrays["states"].shape[1]),
        candidate_size=int(arrays["candidate_inputs"].shape[1]),
        argument_keys=1,
        hidden_size=int(training["hidden_size"]),
        dropout=float(training["dropout"]),
    ).to(device)
    assert_no_planning_or_value_heads(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for epoch in range(int(training["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        action_logits, _, _ = model(states, candidates)
        masked = action_logits.masked_fill(
            ~legal, torch.finfo(action_logits.dtype).min
        )
        per_row = F.cross_entropy(masked, targets, reduction="none")
        loss = (per_row * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(training["fixed_epochs"]) - 1):
            history.append({"epoch": epoch, "action_loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        probabilities = model.action_probabilities(
            torch.as_tensor(arrays["states"], dtype=torch.float32, device=device),
            candidates,
            torch.as_tensor(arrays["legal"], dtype=torch.bool, device=device),
        ).cpu().numpy()
    realized_mass = Counter()
    for row, weight in zip(train_rows, weights_np):
        realized_mass[str(row["source"])] += float(weight)
    total_mass = sum(realized_mass.values())
    return probabilities, {
        "training_rows": len(train_rows),
        "training_tasks_by_source": {
            source: len({row["task_key"] for row in train_rows if row["source"] == source})
            for source in sorted({str(row["source"]) for row in train_rows})
        },
        "realized_source_mass": {
            source: mass / total_mass for source, mass in sorted(realized_mass.items())
        },
        "loss_history_endpoints": history,
    }


def _predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    probabilities: np.ndarray,
    fold: int,
    condition: str,
    variant: str,
    seed: int,
    catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    candidates = arrays["candidates"]
    for index, row in enumerate(rows):
        if row["split"] != "confirmation":
            continue
        target = int(arrays["targets"][index])
        predicted = int(np.argmax(probabilities[index]))
        legal_indices = {
            candidate_index
            for candidate_index, candidate in enumerate(candidates)
            if candidate in row["legal_candidate_ids"]
        }
        tool_probability = sum(
            float(probability)
            for candidate, probability in zip(candidates, probabilities[index])
            if candidate in row["legal_candidate_ids"]
            and catalog[candidate]["kind"] == "tool"
        )
        output.append(
            {
                "fold": fold,
                "condition": condition,
                "variant": variant,
                "training_seed": seed,
                "row_id": row["row_id"],
                "task_key": row["task_key"],
                "task_name": row["task_name"],
                "domain": row["task_name"].split("|", 1)[0],
                "group_id": row["group_id"],
                "target_candidate_id": candidates[target],
                "predicted_candidate_id": candidates[predicted],
                "target_probability": float(probabilities[index, target]),
                "action_nll": float(-math.log(max(float(probabilities[index, target]), 1e-12))),
                "action_correct": float(predicted == target),
                "tool_brier": float((tool_probability - float(row["target_is_tool"])) ** 2),
                "legal_prediction": float(predicted in legal_indices),
            }
        )
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
        raise ValueError("protocol is not frozen before training")
    frozen = protocol["frozen_unified_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("unified dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("unified audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("unified dataset audit did not pass")
    if tuple(protocol["training"]["conditions"]) != CONDITIONS:
        raise ValueError("condition surface differs from frozen protocol")
    if tuple(protocol["training"]["variants"]) != VARIANTS:
        raise ValueError("variant surface differs from frozen protocol")
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    expected_runs = len(dataset["folds"]) * len(CONDITIONS) * len(VARIANTS) * len(seeds)
    if expected_runs != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("frozen neural run budget is inconsistent")

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
    catalog = dataset["candidate_catalog"]
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    runs = []
    for fold in range(len(dataset["folds"])):
        for condition in CONDITIONS:
            rows = _fold_rows(dataset, fold=fold, condition=condition)
            source_mass = protocol["training"]["source_mass"][condition]
            for variant in VARIANTS:
                arrays = _arrays(
                    rows,
                    catalog,
                    variant=variant,
                    hash_dimension=int(protocol["training"]["hash_dimension"]),
                )
                for seed in seeds:
                    _set_seed(seed)
                    probabilities, diagnostics = _train(
                        rows=rows,
                        arrays=arrays,
                        source_mass=source_mass,
                        protocol=protocol,
                        seed=seed,
                        device=device,
                    )
                    prediction_rows = _predictions(
                        rows=rows,
                        arrays=arrays,
                        probabilities=probabilities,
                        fold=fold,
                        condition=condition,
                        variant=variant,
                        seed=seed,
                        catalog=catalog,
                    )
                    _append_jsonl(prediction_path, prediction_rows)
                    runs.append(
                        {
                            "fold": fold,
                            "condition": condition,
                            "variant": variant,
                            "training_seed": seed,
                            "candidate_count": len(arrays["candidates"]),
                            "confirmation_rows": len(prediction_rows),
                            **diagnostics,
                        }
                    )
    if len(runs) != expected_runs:
        raise ValueError("neural training run budget is incomplete")
    confirmation_surface = Counter(
        (row["fold"], row["condition"], row["variant"], row["training_seed"])
        for row in (
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "predictions_sha256": file_sha256(prediction_path),
        "device": device,
        "runs": runs,
        "neural_training_runs": len(runs),
        "confirmation_surface_counts": {
            "|".join(map(str, key)): value for key, value in sorted(confirmation_surface.items())
        },
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "real_external_endpoint_calls": 0,
        "attack_generation": 0,
        "dreamer_runs": 0,
    }
    _write_json(args.output_dir / "run_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
