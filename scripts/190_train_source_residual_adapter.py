"""Frozen source-residual adapter training on the unified action dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.source_residual_adapter import (
    FROZEN_SOURCES,
    SourceResidualActionModel,
    SourceSpecificHeadActionModel,
    source_indices,
)

SPEC = importlib.util.spec_from_file_location(
    "three_source_parent", ROOT / "scripts" / "182_train_three_source_expansion.py"
)
assert SPEC is not None and SPEC.loader is not None
PARENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARENT)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _train(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "training"],
        dtype=np.int64,
    )
    training_rows = [rows[int(index)] for index in train_indices]
    weights_np = PARENT.source_task_balanced_weights(
        training_rows, protocol["training"]["source_mass"]
    )
    row_sources, candidate_sources = source_indices(
        rows, arrays["candidates"], catalog
    )
    states = torch.as_tensor(arrays["states"], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(
        arrays["candidate_inputs"], dtype=torch.float32, device=device
    )
    legal = torch.as_tensor(arrays["legal"], dtype=torch.bool, device=device)
    targets = torch.as_tensor(arrays["targets"], dtype=torch.long, device=device)
    row_source_tensor = torch.as_tensor(
        row_sources, dtype=torch.long, device=device
    )
    candidate_source_tensor = torch.as_tensor(
        candidate_sources, dtype=torch.long, device=device
    )
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    training = protocol["training"]
    head_only = protocol["protocol_id"] == "0814_source_specific_action_head_v1"
    if head_only:
        model = SourceSpecificHeadActionModel(
            state_size=int(arrays["states"].shape[1]),
            candidate_size=int(arrays["candidate_inputs"].shape[1]),
            hidden_size=int(training["hidden_size"]),
            source_count=len(FROZEN_SOURCES),
            dropout=float(training["dropout"]),
        ).to(device)
    else:
        architecture = protocol["architecture"]
        model = SourceResidualActionModel(
            state_size=int(arrays["states"].shape[1]),
            candidate_size=int(arrays["candidate_inputs"].shape[1]),
            hidden_size=int(training["hidden_size"]),
            bottleneck_size=int(architecture["bottleneck_size"]),
            source_count=len(FROZEN_SOURCES),
            residual_scale=float(architecture["residual_scale"]),
            dropout=float(training["dropout"]),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for epoch in range(int(training["fixed_epochs"])):
        _set_seed(seed * 1009 + epoch)
        model.train()
        if head_only:
            logits = model(states[train_indices], candidates, row_source_tensor[train_indices])
        else:
            logits = model(states[train_indices], candidates, row_source_tensor[train_indices], candidate_source_tensor)
        masked = logits.masked_fill(
            ~legal[train_indices], torch.finfo(logits.dtype).min
        )
        per_row = F.cross_entropy(
            masked, targets[train_indices], reduction="none"
        )
        loss = (per_row * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, int(training["fixed_epochs"]) - 1):
            history.append({"epoch": epoch, "action_loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        if head_only:
            probabilities = model.action_probabilities(states, candidates, row_source_tensor, legal).cpu().numpy()
        else:
            probabilities = model.action_probabilities(states, candidates, row_source_tensor, candidate_source_tensor, legal).cpu().numpy()
    realized = Counter()
    for row, weight in zip(training_rows, weights_np):
        realized[str(row["source"])] += float(weight)
    total = sum(realized.values())
    return probabilities, {
        "training_rows": len(training_rows),
        "training_tasks_by_source": {
            source: len(
                {row["task_key"] for row in training_rows if row["source"] == source}
            )
            for source in FROZEN_SOURCES
        },
        "realized_source_mass": {
            source: value / total for source, value in sorted(realized.items())
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "loss_history_endpoints": history,
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
    if protocol["status"] != "preregistered_before_training":
        raise ValueError("source-adapter protocol is not frozen")
    source = protocol["source"]
    if file_sha256(args.dataset) != source["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    if file_sha256(args.audit) != source["audit_sha256"]:
        raise ValueError("audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("unified dataset preflight failed")
    seeds = [int(value) for value in protocol["training"]["training_seeds"]]
    variants = tuple(protocol["training"]["variants"])
    expected_runs = len(dataset["folds"]) * len(variants) * len(seeds)
    if expected_runs != int(protocol["fixed_budget"]["neural_training_runs"]):
        raise ValueError("fixed neural budget is inconsistent")
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
    runs = []
    for fold in range(len(dataset["folds"])):
        rows = PARENT._fold_rows(
            dataset, fold=fold, condition="agentdojo_plus_auxiliary"
        )
        for variant in variants:
            arrays = PARENT._arrays(
                rows,
                dataset["candidate_catalog"],
                variant=variant,
                hash_dimension=int(protocol["training"]["hash_dimension"]),
            )
            for seed in seeds:
                _set_seed(seed)
                probabilities, diagnostics = _train(
                    rows=rows,
                    arrays=arrays,
                    catalog=dataset["candidate_catalog"],
                    protocol=protocol,
                    seed=seed,
                    device=device,
                )
                predictions = PARENT._predictions(
                    rows=rows,
                    arrays=arrays,
                    probabilities=probabilities,
                    fold=fold,
                    condition=str(protocol["training"]["condition"]),
                    variant=variant,
                    seed=seed,
                    catalog=dataset["candidate_catalog"],
                )
                _append(prediction_path, predictions)
                runs.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "training_seed": seed,
                        "confirmation_rows": len(predictions),
                        **diagnostics,
                    }
                )
    metrics = {
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": file_sha256(args.dataset),
        "audit_sha256": file_sha256(args.audit),
        "device": device,
        "neural_training_runs": len(runs),
        "runs": runs,
        "predictions_sha256": file_sha256(prediction_path),
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "real_external_endpoint_calls": 0,
        "new_attack_generation": 0,
        "dreamer_runs": 0,
    }
    if len(runs) != expected_runs:
        raise ValueError("neural run budget incomplete")
    _write(args.output_dir / "run_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
