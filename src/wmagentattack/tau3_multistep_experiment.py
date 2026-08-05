"""Frozen helpers for the tau3 multi-step scale-readiness method test."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .hybrid_semantic_world_model import tool_candidate_vector
from .markov_sufficiency import representation_feature_vector
from .tau3_multistep import observed_semantic_markov_v4_feature_vector


NEURAL_VARIANTS = (
    "semantic_markov",
    "structured_markov_v3",
    "observed_semantic_markov_v4",
    "full_history_diagnostic",
)
BASELINE_VARIANTS = ("frequency_prior", "tfidf_candidate_logistic")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_balanced_weights(task_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(map(str, task_ids))
    if not counts:
        raise ValueError("cannot weight an empty task surface")
    scale = len(task_ids) / len(counts)
    return np.asarray(
        [scale / counts[str(task)] for task in task_ids], dtype=np.float32
    )


def flatten_dataset(
    dataset: Mapping[str, Any], transition_target_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefixes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    catalog = dataset["candidate_catalog"]
    for episode in dataset["episodes"]:
        source_prefixes = episode["prefixes"]
        semantic_prefixes = episode["semantic_prefixes"]
        if len(source_prefixes) != len(semantic_prefixes):
            raise ValueError("source/semantic prefix lengths differ")
        offset = len(prefixes)
        for index, (source, semantic) in enumerate(
            zip(source_prefixes, semantic_prefixes)
        ):
            if source["prefix_index"] != semantic["prefix_index"]:
                raise ValueError("source/semantic prefix indices differ")
            if source["targets"] != semantic["targets"]:
                raise ValueError("source/semantic prefix targets differ")
            legal = list(source["features"]["legal_tools"])
            target = str(source["targets"]["next_action"])
            if target not in legal or any(candidate not in catalog for candidate in legal):
                raise ValueError("prefix target or legal action is outside the catalog")
            prefixes.append(
                {
                    "row_id": f"{episode['episode_id']}::p{index}",
                    "episode_id": episode["episode_id"],
                    "task_id": episode["task_id"],
                    "suite": episode["suite"],
                    "domain": episode["domain"],
                    "split": episode["split"],
                    "prefix_index": int(source["prefix_index"]),
                    "source_prefixes": source_prefixes,
                    "semantic_prefixes": semantic_prefixes,
                    "source_prefix": source,
                    "semantic_prefix": semantic,
                }
            )
        for transition in episode["transitions"]:
            index = int(transition["prefix_index"])
            if index < 0 or index >= len(source_prefixes):
                raise ValueError("transition prefix is outside its episode")
            action = str(transition["action"])
            if action != source_prefixes[index]["targets"]["next_action"]:
                raise ValueError("transition action differs from the prefix target")
            if catalog[action]["kind"] != "tool":
                raise ValueError("an executed transition cannot target text")
            target = transition["target"]
            if set(target) != set(transition_target_names):
                raise ValueError("transition target schema differs from the protocol")
            transitions.append(
                {
                    "row_id": f"{episode['episode_id']}::t{index}",
                    "episode_id": episode["episode_id"],
                    "task_id": episode["task_id"],
                    "suite": episode["suite"],
                    "domain": episode["domain"],
                    "split": episode["split"],
                    "prefix_index": index,
                    "prefix_row_index": offset + index,
                    "action": action,
                    "target": np.asarray(
                        [float(target[name]) for name in transition_target_names],
                        dtype=np.float32,
                    ),
                }
            )
    if not prefixes or not transitions:
        raise ValueError("multi-step experiment surface is empty")
    return prefixes, transitions


def representation_vector(
    row: Mapping[str, Any], *, variant: str, hash_dimension: int
) -> np.ndarray:
    if variant == "observed_semantic_markov_v4":
        return observed_semantic_markov_v4_feature_vector(
            row["source_prefix"], hash_dimension=hash_dimension
        )
    return representation_feature_vector(
        variant=variant,
        source_prefixes=row["source_prefixes"],
        semantic_prefixes=row["semantic_prefixes"],
        prefix_index=int(row["prefix_index"]),
        hash_dimension=hash_dimension,
    )


def build_arrays(
    prefixes: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
    hash_dimension: int,
) -> dict[str, Any]:
    if variant not in NEURAL_VARIANTS:
        raise ValueError(f"unsupported representation: {variant}")
    candidates = sorted(catalog)
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    states = np.stack(
        [
            representation_vector(
                row, variant=variant, hash_dimension=hash_dimension
            )
            for row in prefixes
        ]
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                catalog[candidate], hash_dimension=hash_dimension
            )
            for candidate in candidates
        ]
    )
    legal = np.zeros((len(prefixes), len(candidates)), dtype=bool)
    targets = np.zeros(len(prefixes), dtype=np.int64)
    for row_index, row in enumerate(prefixes):
        for candidate in row["source_prefix"]["features"]["legal_tools"]:
            legal[row_index, candidate_index[str(candidate)]] = True
        targets[row_index] = candidate_index[
            str(row["source_prefix"]["targets"]["next_action"])
        ]
        if not legal[row_index, targets[row_index]]:
            raise ValueError("target action is not legal")
    return {
        "states": states.astype(np.float32, copy=False),
        "candidate_inputs": candidate_inputs.astype(np.float32, copy=False),
        "legal": legal,
        "targets": targets,
        "candidates": candidates,
        "candidate_index": candidate_index,
    }


def frequency_action_probabilities(
    prefixes: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any]
) -> np.ndarray:
    candidates = arrays["candidates"]
    index = arrays["candidate_index"]
    counts = np.ones(len(candidates), dtype=np.float64)
    training = [row for row in prefixes if row["split"] == "training"]
    weights = task_balanced_weights([row["task_id"] for row in training])
    for row, weight in zip(training, weights):
        target = str(row["source_prefix"]["targets"]["next_action"])
        counts[index[target]] += float(weight)
    output = np.zeros_like(arrays["legal"], dtype=np.float64)
    for row_index, legal in enumerate(arrays["legal"]):
        values = counts * legal
        if values.sum() <= 0:
            raise ValueError("frequency baseline has no legal support")
        output[row_index] = values / values.sum()
    return output


def frequency_transition_probabilities(
    transitions: Sequence[Mapping[str, Any]], *, target_count: int
) -> np.ndarray:
    training = [row for row in transitions if row["split"] == "training"]
    weights = task_balanced_weights([row["task_id"] for row in training])
    targets = np.stack([row["target"] for row in training]).astype(np.float64)
    weighted_positive = (targets * weights[:, None]).sum(axis=0)
    total = float(weights.sum())
    probabilities = (weighted_positive + 1.0) / (total + 2.0)
    if probabilities.shape != (target_count,):
        raise ValueError("transition frequency target dimension differs")
    return np.tile(probabilities[None, :], (len(transitions), 1))


def tfidf_state_text(row: Mapping[str, Any]) -> str:
    features = row["source_prefix"]["features"]
    causal = {
        "trusted_goal": features["trusted_goal"],
        "track": features["track"],
        "prefix_index": features["prefix_index"],
        "last_action": features["last_action"],
        "last_observation": features["last_observation"],
        "execution_receipt": features["execution_receipt"],
        "ledger_v2": features["ledger_v2"],
    }
    return json.dumps(causal, ensure_ascii=False, sort_keys=True, default=str)


def candidate_text(descriptor: Mapping[str, Any]) -> str:
    return json.dumps(descriptor, ensure_ascii=False, sort_keys=True, default=str)


def task_metric_map(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            grouped[str(row["task_id"])].append(float(row[metric]))
    if not grouped:
        raise ValueError(f"empty task metric: {metric}")
    return {
        task: float(np.mean(values)) for task, values in sorted(grouped.items())
    }


def task_macro(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(np.mean(list(task_metric_map(rows, metric).values())))


def average_task_maps(maps: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not maps:
        raise ValueError("cannot average an empty task map surface")
    tasks = set(maps[0])
    if any(set(mapping) != tasks for mapping in maps):
        raise ValueError("task maps differ across seeds")
    return {
        task: float(np.mean([mapping[task] for mapping in maps]))
        for task in sorted(tasks)
    }


def two_step_task_map(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[(str(row["task_id"]), str(row["episode_id"]))].append(row)
    task_values: dict[str, list[float]] = defaultdict(list)
    for (task, _episode), episode_rows in grouped.items():
        ordered = sorted(episode_rows, key=lambda row: int(row["prefix_index"]))
        for left, right in zip(ordered, ordered[1:]):
            if int(right["prefix_index"]) != int(left["prefix_index"]) + 1:
                raise ValueError("two-step prediction surface is not adjacent")
            task_values[task].append(
                float(bool(left["action_correct"]) and bool(right["action_correct"]))
            )
    if not task_values:
        raise ValueError("no two-step sequence surface")
    return {
        task: float(np.mean(values)) for task, values in sorted(task_values.items())
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability.astype(np.float64), 1e-12, 1.0 - 1e-12)
    values = target.astype(np.float64)
    bce = -(values * np.log(clipped) + (1.0 - values) * np.log(1.0 - clipped))
    brier = (clipped - values) ** 2
    return float(np.mean(bce)), float(np.mean(brier))


def evaluate_method_gate(
    *,
    nll_seed_gains: Sequence[float],
    accuracy_seed_gains: Sequence[float],
    paired_task_nll_gains: Sequence[float],
    candidate_minus_full_history_nll: float,
    two_step_seed_gains: Sequence[float],
    transition_brier_seed_gains: Sequence[float],
    legal_prediction_rate: float,
    data_gate_passed: bool,
    two_step_surface_available: bool,
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    minimum_seeds = int(gate["minimum_threshold_positive_seeds"])
    nll_threshold = float(gate["minimum_candidate_nll_gain_over_frequency"])
    accuracy_threshold = float(
        gate["minimum_candidate_accuracy_gain_over_frequency"]
    )
    two_step_threshold = float(
        gate["minimum_two_step_sequence_accuracy_gain_over_frequency"]
    )
    brier_threshold = float(gate["minimum_transition_brier_gain_over_frequency"])
    return {
        "data_sufficiency_gate": bool(data_gate_passed),
        "candidate_nll_mean_gain": float(np.mean(nll_seed_gains)) >= nll_threshold,
        "candidate_nll_seed_replication": sum(
            value >= nll_threshold for value in nll_seed_gains
        )
        >= minimum_seeds,
        "candidate_accuracy_mean_gain": float(np.mean(accuracy_seed_gains))
        >= accuracy_threshold,
        "candidate_accuracy_seed_replication": sum(
            value >= accuracy_threshold for value in accuracy_seed_gains
        )
        >= minimum_seeds,
        "candidate_positive_task_fraction": (
            sum(value > 0.0 for value in paired_task_nll_gains)
            / len(paired_task_nll_gains)
        )
        >= float(gate["minimum_positive_task_fraction"]),
        "candidate_within_full_history_nll_gap": candidate_minus_full_history_nll
        <= float(gate["maximum_candidate_nll_gap_to_full_history"]),
        "two_step_surface_available": bool(two_step_surface_available),
        "two_step_mean_gain": bool(two_step_seed_gains)
        and float(np.mean(two_step_seed_gains)) >= two_step_threshold,
        "two_step_seed_replication": sum(
            value >= two_step_threshold for value in two_step_seed_gains
        )
        >= minimum_seeds,
        "transition_brier_mean_gain": float(
            np.mean(transition_brier_seed_gains)
        )
        >= brier_threshold,
        "transition_brier_seed_replication": sum(
            value >= brier_threshold for value in transition_brier_seed_gains
        )
        >= minimum_seeds,
        "all_predictions_legal": legal_prediction_rate
        == float(gate["require_legal_prediction_rate"]),
    }
