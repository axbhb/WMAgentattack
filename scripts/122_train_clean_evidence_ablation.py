"""Run the preregistered clean evidence-ledger ablation without model selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_evidence_probe import (
    FROZEN_VARIANTS,
    CleanEvidenceProbe,
    EventTransformerEncoder,
    VectorEncoder,
    build_within_task_cyclic_donors,
    fit_progress_then_utility,
    predict_probe,
    set_deterministic_seed,
    task_macro_errors,
    transformer_step_features,
    vector_features,
)


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty evidence-ledger dataset")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for episode in episodes:
        final_count = sum(
            bool(prefix["targets"]["is_final_prefix"])
            for prefix in episode["prefixes"]
        )
        if final_count != 1:
            raise ValueError(f"episode does not have one final prefix: {episode['episode_id']}")
        for prefix in episode["prefixes"]:
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "panel": episode["panel"],
                    "data_seed": int(episode["seed"]),
                    "task_id": episode["task_id"],
                    "prefix_index": int(prefix["prefix_index"]),
                    "is_final_prefix": bool(prefix["targets"]["is_final_prefix"]),
                    "progress_target": float(prefix["targets"]["expert_slot_coverage"]),
                    "utility_target": float(episode["targets"]["final_utility"]),
                    "prefix": prefix,
                }
            )
    return records


def _encode_variant(
    records: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    variant: str,
    hash_dimension: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if variant == "event_transformer_state_evidence":
        episode_lookup = {episode["episode_id"]: episode for episode in episodes}
        sequences = []
        for record in records:
            episode = episode_lookup[record["episode_id"]]
            prefix_index = record["prefix_index"]
            sequences.append(
                np.stack(
                    [
                        transformer_step_features(prefix, hash_dimension=hash_dimension)
                        for prefix in episode["prefixes"][: prefix_index + 1]
                    ]
                )
            )
        max_length = max(sequence.shape[0] for sequence in sequences)
        input_size = sequences[0].shape[1]
        inputs = np.zeros((len(sequences), max_length, input_size), dtype=np.float32)
        masks = np.zeros((len(sequences), max_length), dtype=bool)
        for index, sequence in enumerate(sequences):
            inputs[index, : sequence.shape[0]] = sequence
            masks[index, : sequence.shape[0]] = True
        return inputs, masks

    episode_lookup = {episode["episode_id"]: episode for episode in episodes}
    donors = build_within_task_cyclic_donors(episodes)
    vectors = []
    for record in records:
        evidence_override = None
        if variant == "semantic_markov_state_shuffled_evidence":
            donor = episode_lookup[donors[record["episode_id"]]]
            donor_index = min(record["prefix_index"], len(donor["prefixes"]) - 1)
            evidence_override = donor["prefixes"][donor_index]["features"]["evidence_text"]
        vectors.append(
            vector_features(
                record["prefix"],
                variant=variant,
                hash_dimension=hash_dimension,
                evidence_override=evidence_override,
            )
        )
    return np.stack(vectors), None


def _make_model(
    variant: str,
    inputs: np.ndarray,
    *,
    hidden_size: int,
    dropout: float,
    transformer_layers: int,
    transformer_heads: int,
) -> CleanEvidenceProbe:
    if variant == "event_transformer_state_evidence":
        encoder = EventTransformerEncoder(
            input_size=inputs.shape[-1],
            hidden_size=hidden_size,
            dropout=dropout,
            layers=transformer_layers,
            heads=transformer_heads,
            max_length=inputs.shape[1],
        )
    else:
        encoder = VectorEncoder(inputs.shape[-1], hidden_size, dropout)
    return CleanEvidenceProbe(encoder, hidden_size)


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("protocol is not frozen before execution")
    if tuple(protocol["frozen_variants"]) != FROZEN_VARIANTS:
        raise ValueError("variant list differs from the implementation contract")
    episodes = _read_episodes(args.dataset)
    records = _flatten(episodes)
    observed_tasks = {record["task_id"] for record in records}
    frozen_tasks = {
        task_id
        for task_ids in protocol["task_folds"].values()
        for task_id in task_ids
    }
    if observed_tasks != frozen_tasks:
        raise ValueError("dataset tasks differ from frozen task folds")
    if len(episodes) != int(protocol["fixed_budget"]["episodes"]):
        raise ValueError("dataset episode count differs from frozen budget")

    training = protocol["training"]
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "oof_predictions.jsonl"
    run_metrics_path = args.output_dir / "run_metrics.json"
    predictions_path.write_text("", encoding="utf-8")
    all_predictions = []
    run_metrics = []

    task_ids = np.asarray([record["task_id"] for record in records])
    progress_targets = np.asarray(
        [record["progress_target"] for record in records], dtype=np.float32
    )
    final_mask = np.asarray([record["is_final_prefix"] for record in records], dtype=bool)
    utility_targets = np.asarray(
        [record["utility_target"] for record in records], dtype=np.float32
    )

    for variant in FROZEN_VARIANTS:
        inputs, masks = _encode_variant(
            records,
            episodes,
            variant=variant,
            hash_dimension=int(training["hash_dimension"]),
        )
        for training_seed in training["training_seeds"]:
            for fold_name, heldout_tasks in protocol["task_folds"].items():
                train_global = np.flatnonzero(~np.isin(task_ids, heldout_tasks))
                test_global = np.flatnonzero(np.isin(task_ids, heldout_tasks))
                if set(task_ids[test_global]) != set(heldout_tasks):
                    raise ValueError(f"incomplete held-out fold {fold_name}")
                train_final_local = np.flatnonzero(final_mask[train_global])
                train_utility = utility_targets[train_global[train_final_local]]
                set_deterministic_seed(int(training_seed))
                model = _make_model(
                    variant,
                    inputs,
                    hidden_size=int(training["hidden_size"]),
                    dropout=float(training["dropout"]),
                    transformer_layers=int(training["transformer_layers"]),
                    transformer_heads=int(training["transformer_heads"]),
                )
                losses = fit_progress_then_utility(
                    model,
                    inputs=inputs[train_global],
                    masks=masks[train_global] if masks is not None else None,
                    progress_targets=progress_targets[train_global],
                    task_ids=task_ids[train_global].tolist(),
                    final_indices=train_final_local,
                    utility_targets=train_utility,
                    progress_epochs=int(training["fixed_epochs_progress"]),
                    utility_epochs=int(training["fixed_epochs_utility_head"]),
                    batch_size=int(training["batch_size"]),
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                    seed=int(training_seed),
                    device=device,
                )
                progress_prediction, utility_probability = predict_probe(
                    model,
                    inputs=inputs[test_global],
                    masks=masks[test_global] if masks is not None else None,
                    batch_size=int(training["batch_size"]),
                    device=device,
                )
                fold_rows = []
                for local_index, global_index in enumerate(test_global):
                    record = records[int(global_index)]
                    row = {
                        "variant": variant,
                        "training_seed": int(training_seed),
                        "fold": fold_name,
                        "episode_id": record["episode_id"],
                        "panel": record["panel"],
                        "data_seed": record["data_seed"],
                        "task_id": record["task_id"],
                        "prefix_index": record["prefix_index"],
                        "is_final_prefix": record["is_final_prefix"],
                        "progress_target": record["progress_target"],
                        "progress_prediction": float(progress_prediction[local_index]),
                        "utility_target": (
                            record["utility_target"] if record["is_final_prefix"] else None
                        ),
                        "utility_probability": (
                            float(utility_probability[local_index])
                            if record["is_final_prefix"]
                            else None
                        ),
                    }
                    fold_rows.append(row)
                metrics = task_macro_errors(fold_rows)
                run_metrics.append(
                    {
                        "variant": variant,
                        "training_seed": int(training_seed),
                        "fold": fold_name,
                        "heldout_tasks": heldout_tasks,
                        "train_prefixes": len(train_global),
                        "test_prefixes": len(test_global),
                        "train_utility_successes": int(train_utility.sum()),
                        **losses,
                        **{key: value for key, value in metrics.items() if key != "task_metrics"},
                    }
                )
                with predictions_path.open("a", encoding="utf-8") as handle:
                    for row in fold_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_predictions.extend(fold_rows)
                _write_json(run_metrics_path, run_metrics)
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "seed": training_seed,
                            "fold": fold_name,
                            "progress_mae": metrics["task_macro_progress_mae"],
                            "utility_brier": metrics["task_macro_utility_brier"],
                        }
                    ),
                    flush=True,
                )

    expected_runs = len(FROZEN_VARIANTS) * len(training["training_seeds"]) * len(protocol["task_folds"])
    if len(run_metrics) != expected_runs:
        raise ValueError("incomplete fixed ablation grid")
    key_counts = Counter(
        (row["variant"], row["training_seed"], row["episode_id"], row["prefix_index"])
        for row in all_predictions
    )
    if set(key_counts.values()) != {1}:
        raise ValueError("OOF prediction coverage is not exactly one per key")
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "device": device,
        "variants": list(FROZEN_VARIANTS),
        "training_seeds": training["training_seeds"],
        "folds": protocol["task_folds"],
        "counts": {
            "episodes": len(episodes),
            "tasks": len(observed_tasks),
            "prefixes": len(records),
            "runs": len(run_metrics),
            "oof_prediction_rows": len(all_predictions),
        },
        "outcome_gradient_into_progress_encoder": False,
        "model_selection_on_test": False,
        "hyperparameter_grid": False,
    }
    _write_json(args.output_dir / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
