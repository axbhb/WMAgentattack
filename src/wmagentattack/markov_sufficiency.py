"""Frozen causal representations for the Stage 3 Markov-sufficiency test."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .clean_evidence_probe import hashed_text
from .hybrid_semantic_world_model import semantic_state_v3_feature_vector


FROZEN_SUFFICIENCY_VARIANTS = (
    "semantic_markov",
    "structured_markov_v3",
    "full_history_diagnostic",
)

_FORBIDDEN_SOURCE_KEYS = {
    "attack_success",
    "checker",
    "expert_calls",
    "factorized",
    "final_output",
    "final_report",
    "future_calls",
    "future_observations",
    "ground_truth",
    "proof_contract",
    "required_calls",
    "reward",
    "security",
    "targets",
    "task_success",
    "utility",
    "value",
}


def _log_count(value: int | float) -> float:
    return math.log1p(float(value))


def _validate_source_features(features: Mapping[str, Any]) -> None:
    leaked = sorted(_FORBIDDEN_SOURCE_KEYS & {str(key).lower() for key in features})
    if leaked:
        raise ValueError(f"outcome/future/expert source leakage: {leaked}")
    required = {
        "trusted_goal",
        "track",
        "prefix_index",
        "legal_tools",
        "last_action",
        "last_observation",
        "execution_receipt",
    }
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"missing causal source fields: {missing}")


def _shared_source_parts(
    features: Mapping[str, Any], *, hash_dimension: int
) -> list[np.ndarray]:
    _validate_source_features(features)
    return [
        hashed_text(features["trusted_goal"], hash_dimension, "sufficiency-goal"),
        hashed_text(
            sorted(features["legal_tools"]),
            hash_dimension,
            "sufficiency-legal",
        ),
        hashed_text(
            features["last_action"],
            hash_dimension,
            "sufficiency-last-action",
        ),
        hashed_text(
            {"track": features["track"]},
            hash_dimension,
            "sufficiency-policy-track",
        ),
    ]


def semantic_markov_feature_vector(
    source_prefix: Mapping[str, Any], *, hash_dimension: int
) -> np.ndarray:
    features = source_prefix["features"]
    parts = _shared_source_parts(features, hash_dimension=hash_dimension)
    zeros = np.zeros(hash_dimension, dtype=np.float32)
    numeric = np.zeros(18, dtype=np.float32)
    numeric[0] = _log_count(int(features["prefix_index"]))
    return np.concatenate((*parts, zeros, zeros, zeros, numeric)).astype(
        np.float32, copy=False
    )


def full_history_diagnostic_feature_vector(
    source_prefixes: Sequence[Mapping[str, Any]],
    *,
    prefix_index: int,
    hash_dimension: int,
) -> np.ndarray:
    if prefix_index < 0 or prefix_index >= len(source_prefixes):
        raise ValueError("full-history prefix index is outside the episode")
    visible = source_prefixes[: prefix_index + 1]
    for row in visible:
        _validate_source_features(row["features"])
    current = visible[-1]["features"]
    parts = _shared_source_parts(current, hash_dimension=hash_dimension)
    actions = [row["features"]["last_action"] for row in visible]
    observations = [row["features"]["last_observation"] for row in visible]
    receipts = [row["features"]["execution_receipt"] for row in visible]
    parts.extend(
        [
            hashed_text(actions, hash_dimension, "sufficiency-full-actions"),
            hashed_text(
                observations,
                hash_dimension,
                "sufficiency-full-observations",
            ),
            hashed_text(receipts, hash_dimension, "sufficiency-full-receipts"),
        ]
    )
    error_count = sum(
        str(row["features"]["execution_receipt"].get("status")) == "error"
        for row in visible
    )
    nonempty_observations = sum(
        bool(row["features"]["last_observation"]) for row in visible
    )
    observation_characters = sum(
        len(str(row["features"]["last_observation"])) for row in visible
    )
    numeric = np.zeros(18, dtype=np.float32)
    numeric[:6] = (
        _log_count(int(current["prefix_index"])),
        _log_count(len(visible)),
        _log_count(error_count),
        _log_count(nonempty_observations),
        _log_count(observation_characters),
        float(str(current["execution_receipt"].get("status")) == "error"),
    )
    return np.concatenate((*parts, numeric)).astype(np.float32, copy=False)


def representation_feature_vector(
    *,
    variant: str,
    source_prefixes: Sequence[Mapping[str, Any]],
    semantic_prefixes: Sequence[Mapping[str, Any]],
    prefix_index: int,
    hash_dimension: int,
) -> np.ndarray:
    if variant not in FROZEN_SUFFICIENCY_VARIANTS:
        raise ValueError(f"unknown sufficiency representation: {variant}")
    if len(source_prefixes) != len(semantic_prefixes):
        raise ValueError("source and semantic episode lengths differ")
    source = source_prefixes[prefix_index]
    semantic = semantic_prefixes[prefix_index]
    if int(source["prefix_index"]) != int(semantic["prefix_index"]):
        raise ValueError("source and semantic prefix indices differ")
    if source["targets"] != semantic["targets"]:
        raise ValueError("source and semantic prefix targets differ")
    if variant == "semantic_markov":
        output = semantic_markov_feature_vector(
            source, hash_dimension=hash_dimension
        )
    elif variant == "structured_markov_v3":
        output = semantic_state_v3_feature_vector(
            semantic["features"]["semantic_state_v3"],
            hash_dimension=hash_dimension,
        )
    else:
        output = full_history_diagnostic_feature_vector(
            source_prefixes,
            prefix_index=prefix_index,
            hash_dimension=hash_dimension,
        )
    expected = representation_feature_size(hash_dimension)
    if output.shape != (expected,):
        raise ValueError(
            f"representation size mismatch for {variant}: {output.shape} != {expected}"
        )
    return output


def representation_feature_size(hash_dimension: int) -> int:
    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive")
    return 7 * hash_dimension + 18


def validate_dataset_alignment(
    source: Mapping[str, Any], semantic: Mapping[str, Any]
) -> None:
    source_episodes = source.get("episodes", ())
    semantic_episodes = semantic.get("episodes", ())
    if len(source_episodes) != len(semantic_episodes):
        raise ValueError("source and semantic episode counts differ")
    for source_episode, semantic_episode in zip(source_episodes, semantic_episodes):
        identity_fields = (
            "episode_id",
            "task_id",
            "suite",
            "split",
            "track",
            "run_seed",
        )
        for field in identity_fields:
            if source_episode.get(field) != semantic_episode.get(field):
                raise ValueError(f"episode alignment differs at {field}")
        source_prefixes = source_episode["prefixes"]
        semantic_prefixes = semantic_episode["prefixes"]
        if len(source_prefixes) != len(semantic_prefixes):
            raise ValueError("aligned episode prefix counts differ")
        for source_prefix, semantic_prefix in zip(
            source_prefixes, semantic_prefixes
        ):
            if source_prefix["prefix_index"] != semantic_prefix["prefix_index"]:
                raise ValueError("aligned prefix indices differ")
            if source_prefix["targets"] != semantic_prefix["targets"]:
                raise ValueError("aligned targets differ")


def evaluate_sufficiency_gate(
    *,
    action_seed_gains: Sequence[float],
    evidence_seed_gains: Sequence[float],
    action_task_gains: Sequence[float],
    evidence_task_gains: Sequence[float],
    structured_minus_full_action_nll: float,
    structured_minus_full_evidence_bce: float,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate the frozen Stage 3 gate without changing any threshold."""

    action_threshold = float(gates["minimum_action_nll_gain"])
    evidence_threshold = float(gates["minimum_evidence_bce_gain"])
    minimum_seeds = int(gates["minimum_threshold_positive_seeds"])
    minimum_tasks = int(gates["minimum_confirmation_positive_tasks"])
    checks = {
        "structured_action_mean_gain": float(np.mean(action_seed_gains))
        >= action_threshold,
        "structured_action_seed_replication": sum(
            value >= action_threshold for value in action_seed_gains
        )
        >= minimum_seeds,
        "structured_action_paired_tasks": sum(
            value > 0.0 for value in action_task_gains
        )
        >= minimum_tasks,
        "structured_evidence_mean_gain": float(np.mean(evidence_seed_gains))
        >= evidence_threshold,
        "structured_evidence_seed_replication": sum(
            value >= evidence_threshold for value in evidence_seed_gains
        )
        >= minimum_seeds,
        "structured_evidence_paired_tasks": sum(
            value > 0.0 for value in evidence_task_gains
        )
        >= minimum_tasks,
        "structured_action_noninferior_to_full_history": (
            structured_minus_full_action_nll
            <= float(gates["maximum_action_nll_gap_to_full_history"])
        ),
        "structured_evidence_noninferior_to_full_history": (
            structured_minus_full_evidence_bce
            <= float(gates["maximum_evidence_bce_gap_to_full_history"])
        ),
    }
    return checks
