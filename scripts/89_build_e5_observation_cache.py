"""Precompute frozen E5 plus hashed structured observations for DreamerV3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.dreamer_world_model import hash_text_features, step_to_dreamer_text
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord
from wmagentattack.semantic_observations import (
    combine_feature_blocks,
    hashed_structured_features,
    observation_cache_key,
)


PROBE_PATH = ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py"
SPEC = importlib.util.spec_from_file_location("semantic_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, nargs="+", required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--structured-dim", type=int, default=32)
    parser.add_argument("--include-hash", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        StepRecord.model_validate(row)
        for path in args.steps
        for row in read_jsonl(path)
    ]
    by_key = {}
    for record in records:
        by_key.setdefault(observation_cache_key(record), record)
    texts = sorted({step_to_dreamer_text(record) for record in by_key.values()})
    embeddings = PROBE._e5_embeddings(
        texts,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
    )
    keys = sorted(by_key)
    feature_rows = []
    for key in keys:
        record = by_key[key]
        text = step_to_dreamer_text(record)
        semantic = embeddings[text]
        structured = hashed_structured_features(record, dim=args.structured_dim)
        feature_rows.append(
            combine_feature_blocks(
                semantic,
                *([hash_text_features(text, 768)] if args.include_hash else []),
                structured,
            )
        )
    features = np.stack(feature_rows).astype(np.float32)
    metadata = {
        "format": "wmagentattack_e5_structured_observations_v1",
        "model_name": args.model_name,
        "e5_prefix": "query:",
        "e5_max_length": 512,
        "semantic_dim": int(next(iter(embeddings.values())).shape[0]),
        "structured_dim": args.structured_dim,
        "hash_dim": 768 if args.include_hash else 0,
        "include_hash": args.include_hash,
        "feature_dim": int(features.shape[1]),
        "record_count": len(records),
        "unique_observation_count": len(keys),
        "unique_text_count": len(texts),
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.steps
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        keys=np.asarray(keys, dtype="<U64"),
        features=features,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
