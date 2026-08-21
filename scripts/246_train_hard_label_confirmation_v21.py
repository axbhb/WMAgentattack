"""Run the frozen v21 hard-label, held-out, and component-ablation comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.intervention_modular_world_model import (
    InterventionSharedEffectTransition,
    trainable_parameter_count,
)


spec = importlib.util.spec_from_file_location(
    "v20_train", ROOT / "scripts" / "243_train_intervention_models_v20.py"
)
v20 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v20)


ALIASES = (
    "structured_residual_v6",
    "intervention_modular_v20",
    "intervention_no_pair_v21",
    "intervention_no_execution_experts_v21",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_split(
    base: dict[str, Any],
    split: dict[str, list[str]],
    fold_marker: int,
    *,
    sequences_enabled: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    data = dict(base)
    data["rows"] = copy.deepcopy(base["rows"])
    train_refs = set(split["train_refs"])
    test_refs = set(split["test_refs"])
    if train_refs & test_refs:
        raise ValueError("v21 train/test transition overlap")
    if train_refs | test_refs != {row["transition_ref"] for row in data["rows"]}:
        raise ValueError("v21 split does not cover the full hard-label view")
    train_indices = []
    test_indices = []
    train_marker = (fold_marker + 1) % 3
    for index, row in enumerate(data["rows"]):
        if row["transition_ref"] in test_refs:
            row["confirmation_fold"] = fold_marker
            test_indices.append(index)
        else:
            row["confirmation_fold"] = train_marker
            train_indices.append(index)
    if not sequences_enabled:
        data["sequences"] = []
    return data, np.asarray(train_indices), np.asarray(test_indices)


def train_alias(
    alias: str,
    fold_marker: int,
    seed: int,
    data: dict[str, Any],
    cfg: dict[str, Any],
):
    runtime_arm = alias
    run_cfg = copy.deepcopy(cfg)
    original_factory = v20.make_model
    if alias == "intervention_no_pair_v21":
        runtime_arm = "intervention_modular_v20"
        run_cfg["pair_weight"] = 0.0
    elif alias == "intervention_no_execution_experts_v21":
        runtime_arm = "intervention_modular_v20"

        def shared_factory(arm, state_size, action_size, targets, values):
            if arm != "intervention_modular_v20":
                return original_factory(arm, state_size, action_size, targets, values)
            return InterventionSharedEffectTransition(
                state_size,
                action_size,
                int(values["hidden_sizes"]["intervention_modular_v20"]),
                targets,
            )

        v20.make_model = shared_factory
    try:
        return v20.train_one(runtime_arm, fold_marker, seed, data, run_cfg), runtime_arm
    finally:
        v20.make_model = original_factory


def _bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))


def _task_macro(values: np.ndarray, indices: np.ndarray, rows: list[dict]) -> float:
    grouped = defaultdict(list)
    for local, global_index in enumerate(indices):
        grouped[rows[global_index]["task_id"]].append(float(values[local]))
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def _pair_accuracy(probabilities, targets, test_lookup, pairs) -> tuple[float | None, int]:
    scores = []
    for left, right in pairs:
        if left not in test_lookup or right not in test_lookup:
            continue
        lp, rp = probabilities[test_lookup[left]], probabilities[test_lookup[right]]
        ly, ry = targets[left], targets[right]
        own = _bce(lp, ly).mean() + _bce(rp, ry).mean()
        swapped = _bce(lp, ry).mean() + _bce(rp, ly).mean()
        scores.append(1.0 if own < swapped else 0.5 if own == swapped else 0.0)
    return (float(np.mean(scores)) if scores else None), len(scores)


def evaluate(
    model,
    runtime_arm: str,
    alias: str,
    split_suite: str,
    split_name: str,
    fold_marker: int,
    seed: int,
    data: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    vocabulary: list[str],
) -> dict[str, Any]:
    states = torch.tensor(data["states"][test_indices])
    actions = torch.tensor(data["actions"][test_indices])
    model.eval()
    with torch.no_grad():
        logits, execution_logits = model(states, actions)
        probabilities = torch.sigmoid(logits).numpy()
        execution_probability = torch.sigmoid(execution_logits).numpy()
    target = data["targets"][test_indices]
    execution = data["execution"][test_indices]
    per_row_bce = _bce(probabilities, target).mean(axis=1)
    positive_nll = np.asarray([
        float((-np.log(np.clip(probabilities[index][row == 1], 1e-7, 1.0))).mean())
        for index, row in enumerate(target)
    ])
    positive_recall = np.asarray([
        float((probabilities[index][row == 1] >= 0.5).mean())
        for index, row in enumerate(target)
    ])
    train_positive = data["targets"][train_indices].sum(axis=0) > 0
    unseen_values = []
    unseen_hits = []
    for local, row in enumerate(target):
        mask = np.logical_and(row == 1, ~train_positive)
        if mask.any():
            unseen_values.extend((-np.log(np.clip(probabilities[local][mask], 1e-7, 1.0))).tolist())
            unseen_hits.extend((probabilities[local][mask] >= 0.5).astype(float).tolist())
    test_lookup = {global_index: local for local, global_index in enumerate(test_indices)}
    pair_accuracy, pair_count = _pair_accuracy(
        probabilities, data["targets"], test_lookup, data["pairs"]
    )
    rollout_bce = None
    rollout_positive_nll = None
    if split_suite == "task_disjoint":
        rollout_bce_values = []
        rollout_positive_values = []
        for sequence in data["sequences"]:
            if not all(index in test_lookup for index in sequence):
                continue
            rollout = v20.rollout_probabilities(model, runtime_arm, sequence, data)
            sequence_target = data["targets"][sequence]
            rollout_bce_values.append(float(_bce(rollout, sequence_target).mean()))
            rollout_positive_values.extend(
                (-np.log(np.clip(rollout[sequence_target == 1], 1e-7, 1.0))).tolist()
            )
        rollout_bce = float(np.mean(rollout_bce_values))
        rollout_positive_nll = float(np.mean(rollout_positive_values))
    execution_mask = np.asarray([
        not token.startswith("execution=") for token in vocabulary
    ])
    error_rows = np.flatnonzero(execution == 1)
    error_effect_positive = []
    for local in error_rows:
        mask = np.logical_and(target[local] == 1, execution_mask)
        error_effect_positive.extend(
            (-np.log(np.clip(probabilities[local][mask], 1e-7, 1.0))).tolist()
        )
    return {
        "arm": alias,
        "runtime_arm": runtime_arm,
        "split_suite": split_suite,
        "split_name": split_name,
        "fold_marker": fold_marker,
        "seed": seed,
        "training_rows": len(train_indices),
        "confirmation_rows": len(test_indices),
        "hard_task_macro_bce": _task_macro(per_row_bce, test_indices, data["rows"]),
        "hard_positive_task_macro_nll": _task_macro(positive_nll, test_indices, data["rows"]),
        "hard_positive_task_macro_recall": _task_macro(positive_recall, test_indices, data["rows"]),
        "unseen_positive_nll": float(np.mean(unseen_values)) if unseen_values else None,
        "unseen_positive_recall": float(np.mean(unseen_hits)) if unseen_hits else None,
        "unseen_positive_occurrences": len(unseen_values),
        "execution_brier": float(np.mean((execution_probability - execution) ** 2)),
        "execution_accuracy": float(np.mean((execution_probability >= 0.5) == execution)),
        "error_effect_positive_nll": float(np.mean(error_effect_positive)) if error_effect_positive else None,
        "pair_assignment_accuracy": pair_accuracy,
        "pair_comparisons": pair_count,
        "v19_rollout_hard_bce": rollout_bce,
        "v19_rollout_positive_nll": rollout_positive_nll,
        "parameter_count": trainable_parameter_count(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "model_preregistered_before_results":
        raise ValueError("v21 model protocol is not frozen before results")
    if sha256(args.dataset) != protocol["hard_view"]["sha256"]:
        raise ValueError("frozen v21 hard-view hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    base = v20.arrays(
        dataset, int(cfg["state_hash_dimension"]), int(cfg["action_hash_dimension"])
    )
    torch.set_num_threads(int(cfg["torch_threads"]))
    split_manifest = dataset["split_manifest"]
    suites = (
        ("task_disjoint", split_manifest["task_disjoint"], cfg["task_training_seeds"], True),
        ("tool_family_heldout", split_manifest["tool_family_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
        ("source_heldout", split_manifest["source_heldout_diagnostic"], cfg["diagnostic_training_seeds"], False),
    )
    runs = []
    for split_suite, splits, seeds, sequences_enabled in suites:
        for fold_marker, (split_name, split) in enumerate(sorted(splits.items())):
            data, train_indices, test_indices = materialize_split(
                base, split, fold_marker, sequences_enabled=sequences_enabled
            )
            run_cfg = copy.deepcopy(cfg)
            if not sequences_enabled:
                run_cfg["sequence_weight"] = 0.0
            for alias in ALIASES:
                for seed in seeds:
                    (model, history), runtime_arm = train_alias(
                        alias, fold_marker, int(seed), data, run_cfg
                    )
                    row = evaluate(
                        model,
                        runtime_arm,
                        alias,
                        split_suite,
                        split_name,
                        fold_marker,
                        int(seed),
                        data,
                        train_indices,
                        test_indices,
                        dataset["effect_token_vocabulary"],
                    )
                    row["history"] = history
                    runs.append(row)
    expected = int(protocol["fixed_budget"]["maximum_model_fits_if_view_go"])
    if len(runs) != expected:
        raise RuntimeError(f"incomplete v21 budget: {len(runs)} != {expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wmagentattack.hard_label_models.v21",
        "dataset_sha256": sha256(args.dataset),
        "expected_runs": expected,
        "completed_runs": len(runs),
        "runtime_failures": 0,
        "arms": list(ALIASES),
        "runs": runs,
    }
    write(args.output_dir / "run_metrics.json", payload)
    print(json.dumps({"completed_runs": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
