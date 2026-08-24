"""Build a label-blind frozen E5 effect/action cache for v23."""

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

from wmagentattack.pretrained_semantic_effect_world_model import (
    effect_token_description,
    normalized_action_description,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = protocol["pretrained_semantic_cache"]
    if sha256(args.dataset) != protocol["data"]["sha256"]:
        raise ValueError("v23 dataset hash mismatch")
    model_path = Path(cfg["model_snapshot"])
    for name, expected in cfg["model_file_sha256"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"E5 file hash mismatch: {name}")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    vocabulary = dataset["effect_token_vocabulary"]
    label_texts = [effect_token_description(token) for token in vocabulary]
    action_texts = [
        normalized_action_description(row["model_input"]["normalized_action"])
        for row in dataset["transitions"]
    ]
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
    labels = projected[: len(label_texts)].astype(np.float32)
    actions = projected[len(label_texts) :].astype(np.float32)
    metadata = {
        "schema_version": "wmagentattack.pretrained_semantic_cache.v23",
        "model_name": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "dataset_sha256": sha256(args.dataset),
        "label_count": len(labels),
        "action_count": len(actions),
        "semantic_dimension": dimension,
        "explained_energy": float((singular[:dimension] ** 2).sum() / (singular ** 2).sum()),
        "outcome_fields_consumed": [],
        "inputs": ["effect_token_vocabulary", "model_input.normalized_action"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        label_features=labels,
        action_features=actions,
        components=components.astype(np.float32),
        label_texts=np.asarray(label_texts),
        action_texts=np.asarray(action_texts),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    audit = {
        **metadata,
        "cache_sha256": sha256(args.output),
        "finite": bool(np.isfinite(labels).all() and np.isfinite(actions).all()),
        "unit_norm_max_error": float(max(
            np.max(np.abs(np.linalg.norm(labels, axis=1) - 1.0)),
            np.max(np.abs(np.linalg.norm(actions, axis=1) - 1.0)),
        )),
        "deterministic_text_order": True,
        "real_external_endpoint_calls": 0,
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cache": str(args.output), "sha256": audit["cache_sha256"]}))


if __name__ == "__main__":
    main()
