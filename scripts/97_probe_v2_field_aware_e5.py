"""Frozen field-aware E5 ranking probe for AgentDojo-v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROBE = _load_module(
    "semantic_probe_field_aware",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)
AUDIT = _load_module(
    "e5_truncation_audit_field_aware",
    ROOT / "scripts" / "96_audit_v2_e5_truncation.py",
)


ALPHA = 10.0


def _configuration_rows(steps: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for index in PROBE._decision_indices(steps).values():
        step = steps[index]
        group = str(step.multiseed_group_id or "")
        if group:
            grouped[group].append(step)
    rows = PROBE._configuration_rows(steps)
    for row in rows:
        records = grouped.get(str(row["group_id"]), [])
        if len(records) != len(row["texts"]["full"]):
            raise ValueError(f"Incomplete critical view: {row['group_id']}")
        row["texts"]["critical"] = [
            AUDIT.critical_attack_text(step) for step in records
        ]
    return rows


def _build_matrices(
    rows: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    text_maps = {}
    semantic = {}
    for view in ("full", "critical"):
        texts = sorted(
            {
                text
                for split_rows in rows.values()
                for row in split_rows
                for text in row["texts"][view]
            }
        )
        text_maps[view] = PROBE._e5_embeddings(
            texts,
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
        semantic[view] = {
            split: PROBE._mean_group_embeddings(
                rows[split], text_maps[view], view=view
            )
            for split in ("train", "val", "test")
        }

    structured_vocab = PROBE._structured_vocab(rows["train"])
    structured = {
        split: PROBE._structured_matrix(rows[split], structured_vocab)
        for split in ("train", "val", "test")
    }
    matrices = {}
    for split in ("train", "val", "test"):
        two_tower = PROBE._concatenate_normalized(
            semantic["full"][split], semantic["critical"][split]
        )
        matrices[split] = PROBE._concatenate_normalized(
            two_tower, structured[split]
        )
    return matrices, {
        "structured": structured_vocab,
        "full_unique_texts": len(text_maps["full"]),
        "critical_unique_texts": len(text_maps["critical"]),
        "semantic_dimensions": 1536,
        "final_dimensions": int(matrices["train"].shape[1]),
    }


def evaluate_fold(
    rows: dict[str, list[dict[str, Any]]], matrices: dict[str, np.ndarray]
) -> dict[str, Any]:
    fit_rows = rows["train"] + rows["val"]
    fit_matrix = np.concatenate((matrices["train"], matrices["val"]), axis=0)
    model = PROBE._ridge_fit(
        fit_matrix, fit_rows, estimator="pairwise_ridge", alpha=ALPHA
    )
    rank_scores, predictions = PROBE._ridge_predict(model, matrices["test"])
    metrics = PROBE._evaluate(
        rows["test"], rank_scores=rank_scores, predictions=predictions
    )
    candidate = {
        "representation": "e5_full_plus_critical_structured",
        "view": "full_plus_critical",
        "estimator": "pairwise_ridge_largest_gap",
        "alpha": ALPHA,
    }
    return {
        "scope": "frozen field-aware E5 fold evaluation",
        "protocol": {
            "frozen_candidate": candidate,
            "fit_scope": "train_plus_validation",
            "test_retuning": False,
            "decision_step": "first",
            "critical_fields_available_before_outcome": True,
        },
        "fit_summary": {"pair_count": int(model["pair_count"])},
        "test": metrics,
        "test_candidate_scores": [
            {
                "group_id": str(row["group_id"]),
                "task_key": str(row["task_key"]),
                "target": float(row["target"]),
                "target_asr": float(row["target_asr"]),
                "target_bup": float(row["target_bup"]),
                "observed": float(row["observed"]),
                "observed_asr": float(row["observed_asr"]),
                "observed_bup": float(row["observed_bup"]),
                "rank_score": float(rank_scores[index]),
                "prediction": float(predictions[index]),
            }
            for index, row in enumerate(rows["test"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-name", default="intfloat/e5-base-v2")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {
        split: _configuration_rows(
            PROBE._steps(args.data_root / f"{split}_steps.jsonl")
        )
        for split in ("train", "val", "test")
    }
    matrices, metadata = _build_matrices(
        rows,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
    )
    result = evaluate_fold(rows, matrices)
    result["counts"] = {
        split: {
            "configurations": len(rows[split]),
            "tasks": len({row["task_key"] for row in rows[split]}),
        }
        for split in ("train", "val", "test")
    }
    result["representation_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
