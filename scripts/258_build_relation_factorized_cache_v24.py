"""Build the label-blind relation-factorized E5 cache for v24."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.relation_factorized_semantic_world_model import (
    action_relation_descriptions,
    aggregate_channels,
    effect_relation_descriptions,
    relation_kernel,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(texts: list[str], model_path: Path, batch_size: int) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            rows.append(F.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def flatten_descriptions(groups: list[tuple[list[str], np.ndarray]]):
    texts, positions = [], []
    for row, (descriptions, mask) in enumerate(groups):
        for channel, (description, present) in enumerate(zip(descriptions, mask)):
            if present:
                positions.append((row, channel))
                texts.append(description)
    return texts, positions


def restore(
    projected: np.ndarray, positions: list[tuple[int, int]], rows: int, channels: int
) -> np.ndarray:
    output = np.zeros((rows, channels, projected.shape[1]), dtype=np.float32)
    for value, (row, channel) in zip(projected, positions):
        output[row, channel] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = protocol["semantic_cache"]
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v24 protocol is not frozen")
    if sha256(args.dataset) != protocol["data"]["sha256"]:
        raise ValueError("v24 dataset hash mismatch")
    model_path = Path(cfg["model_snapshot"])
    for name, expected in cfg["model_file_sha256"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"E5 file hash mismatch: {name}")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    label_groups = [
        effect_relation_descriptions(token) for token in dataset["effect_token_vocabulary"]
    ]
    action_groups = [
        action_relation_descriptions(row["model_input"]["normalized_action"])
        for row in dataset["transitions"]
    ]
    label_texts, label_positions = flatten_descriptions(label_groups)
    action_texts, action_positions = flatten_descriptions(action_groups)
    all_texts = label_texts + action_texts
    full = encode(all_texts, model_path, int(cfg["batch_size"]))
    _, singular, vh = np.linalg.svd(full, full_matrices=False)
    dimension = int(cfg["semantic_dimension"])
    components = vh[:dimension].copy()
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1
    projected = full @ components.T
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    label_projected = projected[: len(label_texts)]
    action_projected = projected[len(label_texts) :]
    label_mask = np.stack([group[1] for group in label_groups])
    action_mask = np.stack([group[1] for group in action_groups])
    label_channels = restore(
        label_projected, label_positions, len(label_groups), label_mask.shape[1]
    )
    action_channels = restore(
        action_projected, action_positions, len(action_groups), action_mask.shape[1]
    )
    label_features = aggregate_channels(
        label_channels, label_mask, cfg["label_channel_weights"]
    )
    action_features = aggregate_channels(
        action_channels, action_mask, cfg["action_channel_weights"]
    )
    kernel = relation_kernel(
        label_channels,
        label_mask,
        cfg["label_channel_weights"],
        float(cfg["relation_kernel_temperature"]),
    )
    metadata = {
        "schema_version": "wmagentattack.relation_factorized_cache.v24",
        "model_name": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "dataset_sha256": sha256(args.dataset),
        "label_count": len(label_features),
        "action_count": len(action_features),
        "label_channels": label_mask.shape[1],
        "action_channels": action_mask.shape[1],
        "semantic_dimension": dimension,
        "explained_energy": float((singular[:dimension] ** 2).sum() / (singular ** 2).sum()),
        "outcome_fields_consumed": [],
        "inputs": ["effect_token_vocabulary", "model_input.normalized_action"],
        "task_disjoint_unseen_atom_coverage": protocol["data"]["task_disjoint_unseen_atom_coverage"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        label_features=label_features,
        action_features=action_features,
        label_channels=label_channels,
        action_channels=action_channels,
        label_mask=label_mask,
        action_mask=action_mask,
        relation_kernel=kernel,
        components=components.astype(np.float32),
        label_texts=np.asarray(label_texts),
        action_texts=np.asarray(action_texts),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    audit = {
        **metadata,
        "cache_sha256": sha256(args.output),
        "finite": bool(
            np.isfinite(label_features).all()
            and np.isfinite(action_features).all()
            and np.isfinite(kernel).all()
        ),
        "unit_norm_max_error": float(max(
            np.max(np.abs(np.linalg.norm(label_features, axis=1) - 1.0)),
            np.max(np.abs(np.linalg.norm(action_features, axis=1) - 1.0)),
        )),
        "relation_kernel_min": float(kernel.min()),
        "relation_kernel_max": float(kernel.max()),
        "relation_kernel_symmetric_error": float(np.max(np.abs(kernel - kernel.T))),
        "deterministic_text_order": True,
        "real_external_endpoint_calls": 0,
    }
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cache": str(args.output), "sha256": audit["cache_sha256"]}))


if __name__ == "__main__":
    main()
