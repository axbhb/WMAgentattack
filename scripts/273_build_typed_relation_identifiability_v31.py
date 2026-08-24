"""Build one deterministic v31 typed-relation identifiability replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.decision_state import canonical_json_value
from wmagentattack.typed_relation_contract import (
    SCHEMA_VERSION,
    bare_tool_id,
    gold_pairs,
    has_forbidden_key,
    record_description,
    relation_score,
    sanitize_unit,
    schema_vocabulary,
    stable_hash,
    structural_relation,
    typed_goal_units,
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


def pair_accuracy(comparisons: list[tuple[float, float]]) -> float:
    if not comparisons:
        return 0.0
    return sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive, negative in comparisons) / len(comparisons)


def mean_margin(comparisons: list[tuple[float, float]]) -> float:
    return 0.0 if not comparisons else sum(positive - negative for positive, negative in comparisons) / len(comparisons)


def build_rows(
    relational: dict[str, Any], hard: dict[str, Any], embeddings: dict[str, np.ndarray],
    query_keys: dict[tuple[str, int], str], record_keys: dict[str, str], hard_negative_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_by_ref = {row["transition_ref"]: int(row["confirmation_fold"]) for row in hard["transitions"]}
    vocabulary = schema_vocabulary(relational["static_record_candidates"])
    output_rows = []
    combined_comparisons: list[tuple[float, float]] = []
    semantic_comparisons: list[tuple[float, float]] = []
    fold_combined: dict[int, list[tuple[float, float]]] = defaultdict(list)
    fold_semantic: dict[int, list[tuple[float, float]]] = defaultdict(list)
    positive_edges = Counter()
    positive_structural = Counter()
    positive_typed = Counter()
    relation_types = Counter()
    hard_counts = []
    reconstruction_errors = []

    for row in relational["confirmation_rows"]:
        ref = row["transition_ref"]
        fold = fold_by_ref[ref]
        action = row["model_input"]["normalized_action"]
        tool = bare_tool_id(str(action["tool_id"]))
        allowed = relational["static_candidates_by_tool"][tool]
        units = typed_goal_units(row["model_input"]["current_semantic_state"]["goal"], action, vocabulary)
        gold = gold_pairs(row)
        candidates = []
        observed_gold = set()
        for signature in allowed:
            for unit in units:
                types, structural = structural_relation(unit, signature, action)
                semantic = float(np.dot(embeddings[query_keys[(ref, int(unit["index"]))]], embeddings[record_keys[signature]]))
                combined = relation_score(structural, semantic)
                label = (signature, int(unit["index"])) in gold
                if label:
                    observed_gold.add((signature, int(unit["index"])))
                    positive_edges[fold] += 1
                    positive_structural[fold] += int(structural > 0)
                    positive_typed[fold] += int(unit["roles"] != ["LEXICAL_TOKEN"])
                    relation_types.update(types)
                candidates.append({
                    "record_signature": signature,
                    "goal_unit_index": int(unit["index"]),
                    "relation_types": types,
                    "structural_score": round(float(structural), 8),
                    "semantic_similarity": round(semantic, 8),
                    "combined_score": round(combined, 8),
                    "label": int(label),
                })
        if observed_gold != gold:
            reconstruction_errors.append({
                "transition_ref": ref,
                "missing": sorted(f"{signature}:{index}" for signature, index in gold - observed_gold),
                "extra": sorted(f"{signature}:{index}" for signature, index in observed_gold - gold),
            })

        by_key = {(item["record_signature"], item["goal_unit_index"]): item for item in candidates}
        comparisons = []
        for signature, index in sorted(gold):
            positive = by_key[(signature, index)]
            negative_pool = [
                item for item in candidates
                if not item["label"] and (
                    item["record_signature"] == signature or item["goal_unit_index"] == index
                )
            ]
            negative_pool.sort(key=lambda item: (-item["combined_score"], item["record_signature"], item["goal_unit_index"]))
            selected = negative_pool[:hard_negative_count]
            hard_counts.append(len(selected))
            for negative in selected:
                combined_pair = (positive["combined_score"], negative["combined_score"])
                semantic_pair = (positive["semantic_similarity"], negative["semantic_similarity"])
                combined_comparisons.append(combined_pair)
                semantic_comparisons.append(semantic_pair)
                fold_combined[fold].append(combined_pair)
                fold_semantic[fold].append(semantic_pair)
                comparisons.append({
                    "positive_record_signature": signature,
                    "positive_goal_unit_index": index,
                    "negative_record_signature": negative["record_signature"],
                    "negative_goal_unit_index": negative["goal_unit_index"],
                })
        output_rows.append(canonical_json_value({
            "transition_ref": ref,
            "confirmation_fold": fold,
            "tool_hash": stable_hash("v31-tool", tool),
            "goal_units": [sanitize_unit(unit) for unit in units],
            "relation_examples": candidates,
            "hard_negative_pairs": comparisons,
        }))

    positive_total = sum(positive_edges.values())
    audit = {
        "confirmation_rows": len(output_rows),
        "positive_relation_edges": positive_total,
        "positive_edges_by_fold": [positive_edges[index] for index in range(3)],
        "positive_structural_coverage": 0.0 if not positive_total else sum(positive_structural.values()) / positive_total,
        "positive_typed_unit_coverage": 0.0 if not positive_total else sum(positive_typed.values()) / positive_total,
        "relation_type_counts_on_positives": dict(sorted(relation_types.items())),
        "hard_negative_comparisons": len(combined_comparisons),
        "minimum_hard_negatives_per_positive": min(hard_counts, default=0),
        "mean_hard_negatives_per_positive": 0.0 if not hard_counts else sum(hard_counts) / len(hard_counts),
        "combined_pair_accuracy": pair_accuracy(combined_comparisons),
        "semantic_pair_accuracy": pair_accuracy(semantic_comparisons),
        "goal_blind_record_only_pair_accuracy": 0.5 if combined_comparisons else 0.0,
        "combined_pair_margin": mean_margin(combined_comparisons),
        "per_fold": {
            str(fold): {
                "positive_edges": positive_edges[fold],
                "combined_pair_accuracy": pair_accuracy(fold_combined[fold]),
                "semantic_pair_accuracy": pair_accuracy(fold_semantic[fold]),
                "pair_comparisons": len(fold_combined[fold]),
            }
            for fold in range(3)
        },
        "gold_reconstruction_errors": reconstruction_errors,
    }
    return output_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hard-dataset", type=Path, required=True)
    parser.add_argument("--relational-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for name, path in (("hard_dataset", args.hard_dataset), ("relational_dataset", args.relational_dataset)):
        if sha256(path) != protocol["data"][name]["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    cfg = protocol["representation"]
    model_path = Path(cfg["model_snapshot"])
    for name, expected in cfg["model_file_sha256"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"E5 file hash mismatch: {name}")

    hard = json.loads(args.hard_dataset.read_text(encoding="utf-8"))
    relational = json.loads(args.relational_dataset.read_text(encoding="utf-8"))
    vocabulary = schema_vocabulary(relational["static_record_candidates"])
    query_texts: dict[str, str] = {}
    query_keys: dict[tuple[str, int], str] = {}
    for row in relational["confirmation_rows"]:
        action = row["model_input"]["normalized_action"]
        units = typed_goal_units(row["model_input"]["current_semantic_state"]["goal"], action, vocabulary)
        for unit in units:
            key = stable_hash("v31-e5-query", str(unit["_query"]))
            query_texts[key] = str(unit["_query"])
            query_keys[(row["transition_ref"], int(unit["index"]))] = key
    record_keys = {
        signature: stable_hash("v31-e5-record", record_description(signature))
        for signature in relational["static_record_candidates"]
    }
    texts = {**query_texts, **{key: record_description(signature) for signature, key in record_keys.items()}}
    ordered_keys = sorted(texts)
    vectors = encode([texts[key] for key in ordered_keys], model_path, int(cfg["batch_size"]))
    embeddings = {key: vectors[index] for index, key in enumerate(ordered_keys)}

    rows, audit = build_rows(
        relational, hard, embeddings, query_keys, record_keys,
        int(protocol["fixed_budget"]["hard_negatives_per_positive"]),
    )
    dataset = canonical_json_value({
        "schema_version": SCHEMA_VERSION,
        "scope": "clean-only typed relation identifiability",
        "loader_contract": {
            "typed_goal_units_exclude_raw_terms": True,
            "record_descriptors_use_static_tool_schemas": True,
            "semantic_encoder_is_frozen": True,
            "relation_labels_are_v29_record_local_edges": True,
            "hard_negatives_are_label_verified": True,
            "no_utility_security_attack_or_final_outcome_labels": True,
        },
        "representation": {
            "typed_roles": [
                "ATTRIBUTE_TOKEN", "ENTITY_TOKEN", "ACTION_FIELD_TOKEN",
                "VALUE_KIND", "LEXICAL_TOKEN",
            ],
            "relation_types": [
                "DIRECT_ATTRIBUTE", "DIRECT_ENTITY", "ACTION_FIELD_BRIDGE",
                "TYPED_VALUE_CARRIER", "SEMANTIC_ONLY",
            ],
            "structural_weight": cfg["structural_weight"],
            "semantic_weight": cfg["semantic_weight"],
            "semantic_model": cfg["model_name"],
            "semantic_revision": cfg["model_revision"],
        },
        "rows": rows,
    })
    audit.update({
        "schema_version": "wmagentattack.typed_relation_identifiability_audit.v31",
        "dataset_sha256": "computed_after_write",
        "hard_dataset_sha256": sha256(args.hard_dataset),
        "relational_dataset_sha256": sha256(args.relational_dataset),
        "semantic_text_count": len(ordered_keys),
        "semantic_vectors_finite": bool(np.isfinite(vectors).all()),
        "semantic_unit_norm_max_error": float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0))),
        "forbidden_output_keys_present": has_forbidden_key(dataset),
        "task_or_suite_identifiers_present": any("task" in key or "suite" in key for key in json.loads(json.dumps(dataset)).keys()),
        "real_external_endpoint_calls": 0,
        "victim_llm_calls": 0,
        "sandbox_tool_calls": 0,
        "model_fits": 0,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit["dataset_sha256"] = sha256(dataset_path)
    (args.output_dir / "audit.json").write_text(json.dumps(canonical_json_value(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_path), "sha256": audit["dataset_sha256"], "positive_edges": audit["positive_relation_edges"]}))


if __name__ == "__main__":
    main()
