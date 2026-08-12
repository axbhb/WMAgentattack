"""Train frozen ontology-aligned candidates on the three-source dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.hybrid_semantic_world_model import tool_candidate_vector
from wmagentattack.multisource_suitability import file_sha256, representation_vector
from wmagentattack.shared_action_ontology import (
    ONTOLOGY_VECTOR_MODES,
    ontology_candidate_vector,
)


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "three_source_expansion_base",
        ROOT / "scripts" / "182_train_three_source_expansion.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen three-source trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


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


def ontology_arrays(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    state_variant: str,
    candidate_mode: str,
    hash_dimension: int,
) -> dict[str, Any]:
    candidates = sorted(
        {candidate for row in rows for candidate in row["legal_candidate_ids"]}
    )
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    states = np.stack(
        [
            representation_vector(
                row, variant=state_variant, hash_dimension=hash_dimension
            )
            for row in rows
        ]
    )
    candidate_inputs = np.stack(
        [
            ontology_candidate_vector(
                catalog[candidate], mode=candidate_mode, hash_dimension=hash_dimension
            )
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
        raise ValueError("ontology protocol is not frozen before training")
    frozen = protocol["frozen_ontology_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"]:
        raise ValueError("ontology dataset hash mismatch")
    if file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("ontology audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("ontology audit did not pass")
    modes = tuple(protocol["training"]["candidate_modes"])
    if modes != ONTOLOGY_VECTOR_MODES:
        raise ValueError("candidate modes differ from frozen ontology contract")
    variants = tuple(protocol["training"]["state_variants"])
    if variants != ("semantic_markov", "structured_markov_v3"):
        raise ValueError("state variants differ from frozen protocol")
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    expected_runs = len(dataset["folds"]) * len(modes) * len(variants) * len(seeds)
    if expected_runs != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("frozen run budget is inconsistent")

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        import torch
        torch.set_num_threads(8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    runs = []
    catalog = dataset["candidate_catalog"]
    source_mass = protocol["training"]["source_mass"]
    for fold in range(len(dataset["folds"])):
        rows = BASE._fold_rows(
            dataset, fold=fold, condition="agentdojo_plus_auxiliary"
        )
        for candidate_mode in modes:
            for variant in variants:
                arrays = ontology_arrays(
                    rows,
                    catalog,
                    state_variant=variant,
                    candidate_mode=candidate_mode,
                    hash_dimension=int(protocol["training"]["hash_dimension"]),
                )
                for seed in seeds:
                    BASE._set_seed(seed)
                    probabilities, diagnostics = BASE._train(
                        rows=rows,
                        arrays=arrays,
                        source_mass=source_mass,
                        protocol=protocol,
                        seed=seed,
                        device=device,
                    )
                    predictions = BASE._predictions(
                        rows=rows,
                        arrays=arrays,
                        probabilities=probabilities,
                        fold=fold,
                        condition=candidate_mode,
                        variant=variant,
                        seed=seed,
                        catalog=catalog,
                    )
                    _append_jsonl(predictions_path, predictions)
                    runs.append(
                        {
                            "fold": fold,
                            "candidate_mode": candidate_mode,
                            "state_variant": variant,
                            "training_seed": seed,
                            "candidate_count": len(arrays["candidates"]),
                            "confirmation_rows": len(predictions),
                            **diagnostics,
                        }
                    )
    if len(runs) != expected_runs:
        raise ValueError("ontology run budget is incomplete")
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "predictions_sha256": file_sha256(predictions_path),
        "device": device,
        "runs": runs,
        "neural_training_runs": len(runs),
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
