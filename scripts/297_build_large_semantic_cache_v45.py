"""Build inference-visible E5 field/candidate embeddings for the v45 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.large_hybrid_world_model import (
    STATE_FIELDS,
    candidate_text,
    structured_state_texts,
)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(last_hidden.dtype).unsqueeze(-1)
    return (last_hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def cuda_dtype_for_capability(major: int) -> torch.dtype:
    """V100/Volta requires FP16; Ampere and newer can use native BF16."""
    return torch.bfloat16 if major >= 8 else torch.float16


def encode_texts(model, tokenizer, texts, *, batch_size, max_length, device):
    autocast_dtype = (
        cuda_dtype_for_capability(torch.cuda.get_device_capability()[0])
        if device == "cuda"
        else torch.float32
    )
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=device == "cuda"
        ):
            hidden = model(**tokens).last_hidden_state
            pooled = F.normalize(mean_pool(hidden, tokens["attention_mask"]), dim=-1)
        outputs.append(pooled.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    events = dataset["events"]
    candidate_ids = sorted(dataset["candidate_catalog"])
    event_ids = [str(row["event_id"]) for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event IDs are not unique")

    all_fields = []
    field_mask = []
    for event in events:
        fields = structured_state_texts(event["causal_model_input"])
        all_fields.extend(fields)
        field_mask.append([bool(text.strip()) for text in fields])
    candidate_texts = [
        candidate_text(candidate_id, dataset["candidate_catalog"][candidate_id])
        for candidate_id in candidate_ids
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model_dtype = (
        cuda_dtype_for_capability(torch.cuda.get_device_capability()[0])
        if args.device == "cuda"
        else torch.float32
    )
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=model_dtype,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    field_embeddings = encode_texts(
        model, tokenizer, all_fields, batch_size=args.batch_size,
        max_length=args.max_length, device=args.device,
    ).reshape(len(events), len(STATE_FIELDS), -1)
    candidate_embeddings = encode_texts(
        model, tokenizer, candidate_texts, batch_size=args.batch_size,
        max_length=args.max_length, device=args.device,
    )
    if field_embeddings.shape[-1] != candidate_embeddings.shape[-1]:
        raise ValueError("state/candidate semantic dimensions differ")
    if not np.isfinite(field_embeddings).all() or not np.isfinite(candidate_embeddings).all():
        raise ValueError("semantic cache contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_ids=np.asarray(event_ids),
        candidate_ids=np.asarray(candidate_ids),
        field_embeddings=field_embeddings.astype(np.float16),
        field_mask=np.asarray(field_mask, dtype=np.bool_),
        candidate_embeddings=candidate_embeddings.astype(np.float16),
    )
    metadata = {
        "schema_version": "wmagentattack.large_semantic_cache.v45",
        "model": args.model,
        "local_files_only": True,
        "trust_remote_code": False,
        "events": len(events),
        "state_fields": list(STATE_FIELDS),
        "field_rows": len(all_fields),
        "candidates": len(candidate_ids),
        "semantic_size": int(field_embeddings.shape[-1]),
        "max_length": args.max_length,
        "dtype": "float16",
        "model_compute_dtype": str(model_dtype),
        "outcome_fields_encoded": 0,
        "task_identifiers_encoded": 0,
        "real_external_endpoint_calls": 0,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
