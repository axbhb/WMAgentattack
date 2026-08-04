"""Full offline DreamerV3 architecture for AgentDojo skill trajectories.

This backend uses SheepRL's DreamerV3 building blocks rather than only its
RSSM.  It contains an observation/reward/continue world model, discrete skill
actor, critic and EMA target critic, latent imagination, lambda returns, and
DreamerV3 two-hot value distributions.  AgentDojo-specific auxiliary heads
predict attack risk, continuous utility/preservation probabilities, selected
skills, and valid candidate skills.

The learner is offline.  A behavior-cloning term and candidate-skill masks keep
the imagined policy near the support of the collected AgentDojo trajectories.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wmagentattack.dreamer_world_model import (
    _binary_auc,
    _build_vocab,
    _group_steps,
    hash_text_features,
    step_to_dreamer_text,
)
from wmagentattack.schema import StepRecord
from wmagentattack.semantic_observations import observation_cache_key


@dataclass
class FullDreamerV3Config:
    obs_dim: int = 768
    observation_feature_mode: str = "hash"
    observation_feature_path: str | None = None
    encoder_layers: int = 2
    decoder_layers: int = 2
    dense_units: int = 256
    recurrent_state_size: int = 256
    stochastic_size: int = 32
    discrete_size: int = 32
    actor_layers: int = 2
    critic_layers: int = 2
    head_layers: int = 1
    reward_bins: int = 255
    unimix: float = 0.01
    world_learning_rate: float = 3e-4
    actor_learning_rate: float = 8e-5
    critic_learning_rate: float = 8e-5
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 20
    imagination_horizon: int = 5
    imagination_batch_size: int = 256
    gamma: float = 0.99
    lmbda: float = 0.95
    target_critic_tau: float = 0.02
    entropy_scale: float = 3e-4
    behavior_cloning_scale: float = 1.0
    risk_reward_scale: float = 1.0
    risk_reward_binary_mix: float = 1.0
    utility_reward_scale: float = 1.0
    target_skill_reward_scale: float = 0.25
    observation_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    kl_loss_scale: float = 1.0
    kl_dynamic_scale: float = 0.5
    kl_representation_scale: float = 0.1
    kl_free_nats: float = 1.0
    skill_loss_scale: float = 1.0
    candidate_loss_scale: float = 0.25
    risk_loss_scale: float = 1.0
    binary_risk_loss_scale: float = 1.0
    soft_risk_loss_scale: float = 0.0
    risk_final_step_only: bool = False
    group_risk_calibration_loss_scale: float = 0.0
    group_risk_calibration_detach_latent: bool = False
    grouped_risk_calibration_batches: bool = False
    utility_loss_scale: float = 1.0
    binary_utility_loss_scale: float = 0.0
    soft_utility_loss_scale: float = 1.0
    utility_ranking_loss_scale: float = 0.0
    utility_ranking_margin: float = 0.2
    utility_ranking_detach_latent: bool = False
    ranking_pairs_per_batch: int = 0
    group_utility_calibration_loss_scale: float = 0.0
    group_utility_calibration_detach_latent: bool = False
    group_utility_ranking_loss_scale: float = 0.0
    group_utility_ranking_detach_latent: bool = False
    group_utility_min_target_gap: float = 0.1
    group_utility_pairs_per_task: int = 8
    grouped_utility_batches: bool = False
    group_utility_head_only_updates: bool = False
    configuration_value_head_enabled: bool = False
    group_value_calibration_loss_scale: float = 0.0
    group_value_ranking_loss_scale: float = 0.0
    group_value_min_target_gap: float = 0.1
    group_value_pairs_per_task: int = 8
    group_value_head_only_updates: bool = False
    preservation_loss_scale: float = 1.0
    utility_reward_binary_mix: float = 0.0
    validation_risk_mode: str = "binary"
    validation_utility_mode: str = "continuous"
    validation_aggregation: str = "step"
    validation_group_step: str = "final"
    checkpoint_objective: str = "validation_objective"
    grouped_ranking_batches: bool = False
    probability_confidence_floor: float = 0.1
    actor_gradient_clip: float = 100.0
    critic_gradient_clip: float = 100.0
    world_gradient_clip: float = 1000.0
    candidate_threshold: float = 0.5
    seed: int = 7
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.observation_feature_mode not in {"hash", "precomputed"}:
            raise ValueError(
                "observation_feature_mode must be 'hash' or 'precomputed'"
            )
        if (
            self.observation_feature_mode == "precomputed"
            and not self.observation_feature_path
        ):
            raise ValueError(
                "precomputed observations require observation_feature_path"
            )
        if (
            self.observation_feature_mode == "hash"
            and self.observation_feature_path is not None
        ):
            raise ValueError(
                "observation_feature_path is only valid for precomputed mode"
            )
        if not 0.0 <= self.risk_reward_binary_mix <= 1.0:
            raise ValueError("risk_reward_binary_mix must be between 0 and 1")
        if not 0.0 <= self.utility_reward_binary_mix <= 1.0:
            raise ValueError("utility_reward_binary_mix must be between 0 and 1")
        if self.validation_risk_mode not in {"continuous", "binary"}:
            raise ValueError(
                "validation_risk_mode must be 'continuous' or 'binary'"
            )
        if self.validation_utility_mode not in {"continuous", "binary"}:
            raise ValueError(
                "validation_utility_mode must be 'continuous' or 'binary'"
            )
        if self.validation_aggregation not in {"step", "multiseed_group"}:
            raise ValueError(
                "validation_aggregation must be 'step' or 'multiseed_group'"
            )
        if self.validation_group_step not in {"first", "final"}:
            raise ValueError("validation_group_step must be 'first' or 'final'")
        for name in (
            "binary_risk_loss_scale",
            "soft_risk_loss_scale",
            "group_risk_calibration_loss_scale",
            "binary_utility_loss_scale",
            "soft_utility_loss_scale",
            "utility_ranking_loss_scale",
            "group_utility_calibration_loss_scale",
            "group_utility_ranking_loss_scale",
            "group_value_calibration_loss_scale",
            "group_value_ranking_loss_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.group_risk_calibration_loss_scale > 0.0
            and not self.grouped_risk_calibration_batches
        ):
            raise ValueError(
                "group_risk_calibration_loss_scale requires "
                "grouped_risk_calibration_batches"
            )
        if (
            self.group_utility_calibration_loss_scale > 0.0
            or self.group_utility_ranking_loss_scale > 0.0
        ) and not self.grouped_utility_batches:
            raise ValueError(
                "group utility losses require grouped_utility_batches"
            )
        if not 0.0 <= self.group_utility_min_target_gap <= 1.0:
            raise ValueError(
                "group_utility_min_target_gap must be between 0 and 1"
            )
        if self.group_utility_pairs_per_task < 0:
            raise ValueError("group_utility_pairs_per_task must be non-negative")
        if (
            self.group_utility_ranking_loss_scale > 0.0
            and self.group_utility_pairs_per_task == 0
        ):
            raise ValueError(
                "group_utility_ranking_loss_scale requires at least one pair per task"
            )
        if self.group_utility_head_only_updates:
            if not self.grouped_utility_batches:
                raise ValueError(
                    "group_utility_head_only_updates requires grouped utility batches"
                )
            if (
                self.group_utility_calibration_loss_scale > 0.0
                and not self.group_utility_calibration_detach_latent
            ):
                raise ValueError(
                    "head-only group utility calibration must detach the latent"
                )
            if (
                self.group_utility_ranking_loss_scale > 0.0
                and not self.group_utility_ranking_detach_latent
            ):
                raise ValueError(
                    "head-only group utility ranking must detach the latent"
                )
        if not 0.0 <= self.group_value_min_target_gap <= 2.0:
            raise ValueError(
                "group_value_min_target_gap must be between 0 and 2"
            )
        if self.group_value_pairs_per_task < 0:
            raise ValueError("group_value_pairs_per_task must be non-negative")
        if (
            self.group_value_ranking_loss_scale > 0.0
            and self.group_value_pairs_per_task == 0
        ):
            raise ValueError(
                "group_value_ranking_loss_scale requires at least one pair per task"
            )
        if (
            self.group_value_calibration_loss_scale > 0.0
            or self.group_value_ranking_loss_scale > 0.0
        ) and not self.configuration_value_head_enabled:
            raise ValueError(
                "group value losses require configuration_value_head_enabled"
            )
        if (
            self.group_value_calibration_loss_scale > 0.0
            or self.group_value_ranking_loss_scale > 0.0
        ) and not self.group_value_head_only_updates:
            raise ValueError(
                "group value losses currently require head-only updates"
            )
        if self.group_value_head_only_updates and not (
            self.group_value_calibration_loss_scale > 0.0
            or self.group_value_ranking_loss_scale > 0.0
        ):
            raise ValueError(
                "group_value_head_only_updates requires an active group value loss"
            )
        if self.checkpoint_objective not in {
            "validation_objective",
            "grouped_configuration_value_brier",
        }:
            raise ValueError(
                "checkpoint_objective must be validation_objective or "
                "grouped_configuration_value_brier"
            )
        if (
            self.checkpoint_objective == "grouped_configuration_value_brier"
            and not self.configuration_value_head_enabled
        ):
            raise ValueError(
                "grouped_configuration_value_brier checkpointing requires "
                "the configuration value head"
            )
        if self.ranking_pairs_per_batch < 0:
            raise ValueError("ranking_pairs_per_batch must be non-negative")


def _grouped_ranking_order(
    sequences: list[dict[str, Any]], rng: np.random.Generator
) -> np.ndarray:
    """Keep opposite-utility trajectories from the same task close in a batch."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        grouped[str(sequence["group_key"])].append(index)
    group_keys = list(grouped)
    rng.shuffle(group_keys)
    order: list[int] = []
    for key in group_keys:
        indices = grouped[key]
        positive = [
            index
            for index in indices
            if float(sequences[index]["final_binary_utility"]) > 0.5
        ]
        negative = [
            index
            for index in indices
            if float(sequences[index]["final_binary_utility"]) <= 0.5
        ]
        rng.shuffle(positive)
        rng.shuffle(negative)
        while positive or negative:
            if positive:
                order.append(positive.pop())
            if negative:
                order.append(negative.pop())
    return np.asarray(order, dtype=np.int64)


def _build_ranking_pairs(
    sequences: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return same-task positive/negative sequence index pairs."""

    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    for index, sequence in enumerate(sequences):
        label = (
            "positive"
            if float(sequence["final_binary_utility"]) > 0.5
            else "negative"
        )
        grouped[str(sequence["group_key"])][label].append(index)
    return [
        (positive, negative)
        for group in grouped.values()
        for positive in group["positive"]
        for negative in group["negative"]
    ]


def _inject_ranking_pairs(
    base_indices: np.ndarray,
    ranking_pairs: list[tuple[int, int]],
    pairs_per_batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a small tail of an IID batch with explicit ranking pairs."""

    selected = [int(index) for index in base_indices]
    pair_count = min(pairs_per_batch, len(selected) // 2)
    if pair_count == 0 or not ranking_pairs:
        return np.asarray(selected, dtype=np.int64)
    prefix_length = len(selected) - 2 * pair_count
    prefix = selected[:prefix_length]
    blocked = set(prefix)
    injected: list[int] = []
    for _ in range(pair_count):
        candidates = [
            pair
            for pair in ranking_pairs
            if pair[0] not in blocked
            and pair[1] not in blocked
            and pair[0] not in injected
            and pair[1] not in injected
        ]
        pool = candidates or ranking_pairs
        pair = pool[int(rng.integers(0, len(pool)))]
        injected.extend(pair)
    return np.asarray(prefix + injected, dtype=np.int64)


def _multiseed_group_batches(
    sequences: list[dict[str, Any]],
    batch_size: int,
    rng: np.random.Generator,
    *,
    group_key_field: str = "risk_group_key",
    expected_size_field: str = "risk_group_expected_size",
    target_field: str = "final_soft_risk",
    group_label: str = "risk",
    task_aware: bool = False,
) -> list[np.ndarray]:
    """Pack complete repeated-configuration groups without splitting them."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    singleton_units: list[list[int]] = []
    for index, sequence in enumerate(sequences):
        key = sequence.get(group_key_field)
        if key is None:
            singleton_units.append([index])
        else:
            grouped[str(key)].append(index)

    units: list[list[int]] = []
    units_by_task: dict[str, list[list[int]]] = defaultdict(list)
    for key, indices in grouped.items():
        expected_sizes = {
            int(sequences[index][expected_size_field])
            for index in indices
        }
        if len(expected_sizes) != 1:
            raise ValueError(
                f"Inconsistent expected multi-seed size for {group_label} group {key}"
            )
        expected_size = next(iter(expected_sizes))
        if len(indices) != expected_size:
            raise ValueError(
                f"Incomplete {group_label} group {key}: found {len(indices)}, "
                f"expected {expected_size}"
            )
        targets = [float(sequences[index][target_field]) for index in indices]
        if not np.allclose(targets, targets[0], rtol=0.0, atol=1e-7):
            raise ValueError(
                f"Inconsistent posterior target for {group_label} group {key}"
            )
        if len(indices) > batch_size:
            raise ValueError(
                f"{group_label.title()} group {key} has {len(indices)} trajectories, exceeding "
                f"batch size {batch_size}"
            )
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if task_aware:
            task_keys = {str(sequences[index]["group_key"]) for index in indices}
            if len(task_keys) != 1:
                raise ValueError(f"{group_label.title()} group {key} spans tasks")
            units_by_task[next(iter(task_keys))].append(shuffled)
        else:
            units.append(shuffled)

    batches: list[np.ndarray] = []

    def pack(pack_units: list[list[int]]) -> None:
        current: list[int] = []
        for unit in pack_units:
            if current and len(current) + len(unit) > batch_size:
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
            current.extend(unit)
        if current:
            batches.append(np.asarray(current, dtype=np.int64))

    if task_aware:
        task_keys = list(units_by_task)
        rng.shuffle(task_keys)
        for task_key in task_keys:
            task_units = units_by_task[task_key]
            rng.shuffle(task_units)
            pack(task_units)
        rng.shuffle(singleton_units)
        pack(singleton_units)
    else:
        units.extend(singleton_units)
        rng.shuffle(units)
        pack(units)
    return batches


def _continuous_group_pair_indices(
    task_ids: list[int],
    targets: list[float],
    *,
    min_target_gap: float,
    max_pairs_per_task: int,
) -> list[tuple[int, int, float]]:
    """Select deterministic high/low group pairs with the largest target gaps."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        grouped[int(task_id)].append(index)
    selected: list[tuple[int, int, float]] = []
    for task_id in sorted(grouped):
        candidates: list[tuple[int, int, float]] = []
        indices = grouped[task_id]
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                difference = float(targets[left]) - float(targets[right])
                if abs(difference) + 1e-12 < min_target_gap:
                    continue
                high, low = (left, right) if difference > 0.0 else (right, left)
                candidates.append((high, low, abs(difference)))
        candidates.sort(key=lambda row: (-row[2], row[0], row[1]))
        selected.extend(candidates[:max_pairs_per_task])
    return selected


def _append_ranking_pairs(
    base_indices: np.ndarray,
    ranking_pairs: list[tuple[int, int]],
    pairs_per_batch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Append ranking examples while excluding them from group calibration."""

    selected = [int(index) for index in base_indices]
    calibration_members = [True] * len(selected)
    if pairs_per_batch <= 0 or not ranking_pairs:
        return (
            np.asarray(selected, dtype=np.int64),
            np.asarray(calibration_members, dtype=np.bool_),
        )
    blocked = set(selected)
    appended: list[int] = []
    for _ in range(pairs_per_batch):
        candidates = [
            pair
            for pair in ranking_pairs
            if pair[0] not in blocked
            and pair[1] not in blocked
            and pair[0] not in appended
            and pair[1] not in appended
        ]
        pool = candidates or ranking_pairs
        pair = pool[int(rng.integers(0, len(pool)))]
        appended.extend(pair)
    return (
        np.asarray(selected + appended, dtype=np.int64),
        np.asarray(calibration_members + [False] * len(appended), dtype=np.bool_),
    )


def _require_full_sheeprl():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.distributions import Independent, OneHotCategoricalStraightThrough
        from torch.distributions.kl import kl_divergence

        from sheeprl.algos.dreamer_v3.agent import (
            Actor,
            MLPDecoder,
            MLPEncoder,
            RSSM,
            RecurrentModel,
        )
        from sheeprl.algos.dreamer_v3.utils import (
            compute_lambda_values,
            init_weights,
            uniform_init_weights,
        )
        from sheeprl.models.models import LayerNorm, MLP
        from sheeprl.utils.distribution import (
            SymlogDistribution,
            TwoHotEncodingDistribution,
        )
    except Exception as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "FullSheepRLDreamerV3 requires torch and SheepRL 0.5.8.dev. "
            "Run it in the configured sheeprl_env."
        ) from exc
    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Independent": Independent,
        "OneHotCategoricalStraightThrough": OneHotCategoricalStraightThrough,
        "kl_divergence": kl_divergence,
        "Actor": Actor,
        "MLPDecoder": MLPDecoder,
        "MLPEncoder": MLPEncoder,
        "RSSM": RSSM,
        "RecurrentModel": RecurrentModel,
        "compute_lambda_values": compute_lambda_values,
        "init_weights": init_weights,
        "uniform_init_weights": uniform_init_weights,
        "LayerNorm": LayerNorm,
        "MLP": MLP,
        "SymlogDistribution": SymlogDistribution,
        "TwoHotEncodingDistribution": TwoHotEncodingDistribution,
    }


def _grouped_probability_metrics(
    steps: list[StepRecord],
    predictions: dict[str, Any],
    *,
    decision_step: str = "final",
) -> dict[str, Any]:
    """Evaluate one probability prediction per repeated attack configuration.

    Each trajectory first contributes only its requested decision step, so trajectories with
    more tool calls do not receive more weight.  Predictions are then averaged
    across victim-model seeds sharing ``multiseed_group_id``.  Clean groups are
    excluded because their attack probability target is undefined.
    """

    if decision_step not in {"first", "final"}:
        raise ValueError("decision_step must be 'first' or 'final'")
    decision_by_trajectory: dict[str, int] = {}
    for index, step in enumerate(steps):
        previous = decision_by_trajectory.get(step.trajectory_id)
        if previous is None:
            decision_by_trajectory[step.trajectory_id] = index
        elif decision_step == "first" and steps[previous].step_id > step.step_id:
            decision_by_trajectory[step.trajectory_id] = index
        elif decision_step == "final" and steps[previous].step_id < step.step_id:
            decision_by_trajectory[step.trajectory_id] = index

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index in decision_by_trajectory.values():
        step = steps[index]
        if (
            step.multiseed_group_id is not None
            and step.attack_probability_target is not None
        ):
            grouped_indices[step.multiseed_group_id].append(index)

    if not grouped_indices:
        return {
            "grouped_configuration_count": 0,
            "grouped_trajectory_count": 0,
            "grouped_risk_probability_brier_score": None,
            "grouped_risk_probability_mae": None,
            "grouped_utility_probability_brier_score": None,
            "grouped_utility_probability_mae": None,
            "grouped_configuration_value_normalized_brier_score": None,
            "grouped_configuration_value_mae": None,
            "grouped_preservation_configuration_count": 0,
            "grouped_preservation_probability_brier_score": None,
            "grouped_preservation_probability_mae": None,
        }

    risk_predictions = np.asarray(predictions["risk_score"], dtype=np.float32)
    utility_predictions = np.asarray(
        predictions["utility_score"], dtype=np.float32
    )
    preservation_predictions = np.asarray(
        predictions["preservation_score"], dtype=np.float32
    )
    configuration_value_predictions = (
        np.asarray(predictions["configuration_value_score"], dtype=np.float32)
        if "configuration_value_score" in predictions
        else None
    )
    group_risk_target = []
    group_risk_prediction = []
    group_utility_target = []
    group_utility_prediction = []
    group_configuration_value_target = []
    group_configuration_value_prediction = []
    group_preservation_target = []
    group_preservation_prediction = []
    trajectory_count = 0
    for indices in grouped_indices.values():
        trajectory_count += len(indices)
        risk_targets = [steps[index].attack_probability_target for index in indices]
        utility_targets = [
            steps[index].utility_probability_target for index in indices
        ]
        if any(target is None for target in risk_targets + utility_targets):
            raise ValueError("Repeated attack group has missing probability targets")
        group_risk_target.append(float(np.mean(risk_targets)))
        group_risk_prediction.append(float(np.mean(risk_predictions[indices])))
        group_utility_target.append(float(np.mean(utility_targets)))
        group_utility_prediction.append(float(np.mean(utility_predictions[indices])))
        if configuration_value_predictions is not None:
            group_configuration_value_target.append(
                float(np.mean(risk_targets) + np.mean(utility_targets))
            )
            group_configuration_value_prediction.append(
                float(np.mean(configuration_value_predictions[indices]))
            )
        preservation_targets = [
            steps[index].preservation_probability_target for index in indices
        ]
        if all(
            target is not None and steps[index].preservation_trainable
            for index, target in zip(indices, preservation_targets, strict=True)
        ):
            group_preservation_target.append(float(np.mean(preservation_targets)))
            group_preservation_prediction.append(
                float(np.mean(preservation_predictions[indices]))
            )

    risk_target = np.asarray(group_risk_target, dtype=np.float32)
    risk_prediction = np.asarray(group_risk_prediction, dtype=np.float32)
    utility_target = np.asarray(group_utility_target, dtype=np.float32)
    utility_prediction = np.asarray(group_utility_prediction, dtype=np.float32)
    result = {
        "grouped_configuration_count": len(grouped_indices),
        "grouped_trajectory_count": trajectory_count,
        "grouped_risk_probability_brier_score": float(
            np.mean((risk_target - risk_prediction) ** 2)
        ),
        "grouped_risk_probability_mae": float(
            np.mean(np.abs(risk_target - risk_prediction))
        ),
        "grouped_utility_probability_brier_score": float(
            np.mean((utility_target - utility_prediction) ** 2)
        ),
        "grouped_utility_probability_mae": float(
            np.mean(np.abs(utility_target - utility_prediction))
        ),
        "grouped_preservation_configuration_count": len(
            group_preservation_target
        ),
    }
    if group_configuration_value_target:
        value_target = np.asarray(
            group_configuration_value_target, dtype=np.float32
        )
        value_prediction = np.asarray(
            group_configuration_value_prediction, dtype=np.float32
        )
        result["grouped_configuration_value_normalized_brier_score"] = float(
            np.mean(((value_target - value_prediction) / 2.0) ** 2)
        )
        result["grouped_configuration_value_mae"] = float(
            np.mean(np.abs(value_target - value_prediction))
        )
    else:
        result["grouped_configuration_value_normalized_brier_score"] = None
        result["grouped_configuration_value_mae"] = None
    if group_preservation_target:
        preservation_target = np.asarray(
            group_preservation_target, dtype=np.float32
        )
        preservation_prediction = np.asarray(
            group_preservation_prediction, dtype=np.float32
        )
        result["grouped_preservation_probability_brier_score"] = float(
            np.mean((preservation_target - preservation_prediction) ** 2)
        )
        result["grouped_preservation_probability_mae"] = float(
            np.mean(np.abs(preservation_target - preservation_prediction))
        )
    else:
        result["grouped_preservation_probability_brier_score"] = None
        result["grouped_preservation_probability_mae"] = None
    return result


def evaluate_full_dreamer_predictions(
    steps: list[StepRecord],
    predictions: dict[str, Any],
    *,
    validation_risk_mode: str = "binary",
    validation_utility_mode: str = "continuous",
    validation_aggregation: str = "step",
    validation_group_step: str = "final",
) -> dict[str, Any]:
    if validation_risk_mode not in {"continuous", "binary"}:
        raise ValueError(
            "validation_risk_mode must be 'continuous' or 'binary'"
        )
    if validation_utility_mode not in {"continuous", "binary"}:
        raise ValueError(
            "validation_utility_mode must be 'continuous' or 'binary'"
        )
    if validation_aggregation not in {"step", "multiseed_group"}:
        raise ValueError(
            "validation_aggregation must be 'step' or 'multiseed_group'"
        )
    if validation_group_step not in {"first", "final"}:
        raise ValueError("validation_group_step must be 'first' or 'final'")
    skill_true = np.asarray([step.selected_skill for step in steps])
    binary_risk = np.asarray([step.attack_success for step in steps], dtype=np.float32)
    risk_target = np.asarray(
        [
            step.attack_probability_target
            if step.attack_probability_target is not None
            else float(step.attack_success)
            for step in steps
        ],
        dtype=np.float32,
    )
    binary_utility = np.asarray([step.task_success for step in steps], dtype=np.float32)
    utility_target = np.asarray(
        [
            step.utility_probability_target
            if step.utility_probability_target is not None
            else float(step.task_success)
            for step in steps
        ],
        dtype=np.float32,
    )
    preservation_mask = np.asarray(
        [step.preservation_probability_target is not None for step in steps], dtype=bool
    )
    preservation_target = np.asarray(
        [step.preservation_probability_target or 0.0 for step in steps], dtype=np.float32
    )
    skill_proba = np.asarray(predictions["next_skill_proba"])
    skill_classes = np.asarray(predictions["skill_classes"])
    risk = np.asarray(predictions["risk_score"], dtype=np.float32)
    utility = np.asarray(predictions["utility_score"], dtype=np.float32)
    preservation = np.asarray(predictions["preservation_score"], dtype=np.float32)
    top_k = min(3, skill_proba.shape[1])
    top_indices = np.argsort(skill_proba, axis=1)[:, -top_k:]
    metrics = {
        "next_skill_accuracy": float(
            np.mean(skill_true == skill_classes[np.argmax(skill_proba, axis=1)])
        ),
        "next_skill_top3_accuracy": float(
            np.mean(
                [
                    truth in skill_classes[indices]
                    for truth, indices in zip(skill_true, top_indices, strict=True)
                ]
            )
        ),
        "risk_auc": _binary_auc(binary_risk.astype(int), risk),
        "risk_brier_score": float(np.mean((binary_risk - risk) ** 2)),
        "risk_probability_brier_score": float(
            np.mean((risk_target - risk) ** 2)
        ),
        "risk_probability_mae": float(np.mean(np.abs(risk_target - risk))),
        "binary_utility_auc": _binary_auc(binary_utility.astype(int), utility),
        "binary_utility_brier_score": float(
            np.mean((binary_utility - utility) ** 2)
        ),
        "utility_probability_brier_score": float(
            np.mean((utility_target - utility) ** 2)
        ),
        "utility_probability_mae": float(np.mean(np.abs(utility_target - utility))),
        "preservation_eval_count": int(preservation_mask.sum()),
    }
    if preservation_mask.any():
        metrics["preservation_probability_brier_score"] = float(
            np.mean(
                (preservation_target[preservation_mask] - preservation[preservation_mask])
                ** 2
            )
        )
        metrics["preservation_probability_mae"] = float(
            np.mean(
                np.abs(
                    preservation_target[preservation_mask]
                    - preservation[preservation_mask]
                )
            )
        )
    else:
        metrics["preservation_probability_brier_score"] = None
        metrics["preservation_probability_mae"] = None
    risk_validation_brier = metrics[
        "risk_probability_brier_score"
        if validation_risk_mode == "continuous"
        else "risk_brier_score"
    ]
    metrics["validation_objective_continuous"] = (
        risk_validation_brier
        + metrics["utility_probability_brier_score"]
        + (metrics["preservation_probability_brier_score"] or 0.0)
        + 0.25 * (1.0 - metrics["next_skill_accuracy"])
    )
    metrics["validation_objective_binary"] = (
        risk_validation_brier
        + metrics["binary_utility_brier_score"]
        + (metrics["preservation_probability_brier_score"] or 0.0)
        + 0.25 * (1.0 - metrics["next_skill_accuracy"])
    )
    grouped = _grouped_probability_metrics(
        steps, predictions, decision_step=validation_group_step
    )
    metrics.update(grouped)
    if grouped["grouped_configuration_count"]:
        metrics["validation_objective_multiseed_group"] = (
            grouped["grouped_risk_probability_brier_score"]
            + grouped["grouped_utility_probability_brier_score"]
            + (grouped["grouped_preservation_probability_brier_score"] or 0.0)
            + 0.25 * (1.0 - metrics["next_skill_accuracy"])
        )
    else:
        metrics["validation_objective_multiseed_group"] = None
    if validation_aggregation == "multiseed_group":
        if metrics["validation_objective_multiseed_group"] is None:
            raise ValueError(
                "multiseed_group validation requested but no grouped labels exist"
            )
        metrics["validation_objective"] = metrics[
            "validation_objective_multiseed_group"
        ]
    else:
        metrics["validation_objective"] = metrics[
            f"validation_objective_{validation_utility_mode}"
        ]
    metrics["validation_risk_mode"] = validation_risk_mode
    metrics["validation_utility_mode"] = validation_utility_mode
    metrics["validation_aggregation"] = validation_aggregation
    metrics["validation_group_step"] = validation_group_step
    return metrics


class FullSheepRLDreamerV3:
    """Offline full DreamerV3 learner specialized for AgentDojo skill actions."""

    def __init__(
        self,
        config: FullDreamerV3Config | None = None,
        skill_classes: list[str] | None = None,
    ) -> None:
        self.config = config or FullDreamerV3Config()
        self.skill_classes = skill_classes or []
        self.skill_to_id = {
            skill: index for index, skill in enumerate(self.skill_classes)
        }
        self._module = None
        self._observation_feature_matrix: np.ndarray | None = None
        self._observation_feature_index: dict[str, int] | None = None
        self.training_history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None

    def _device_name(self) -> str:
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _make_module(self):
        deps = _require_full_sheeprl()
        torch, nn = deps["torch"], deps["nn"]
        Actor = deps["Actor"]
        MLPEncoder, MLPDecoder = deps["MLPEncoder"], deps["MLPDecoder"]
        RSSM, RecurrentModel = deps["RSSM"], deps["RecurrentModel"]
        MLP, LayerNorm = deps["MLP"], deps["LayerNorm"]
        init_weights = deps["init_weights"]
        uniform_init_weights = deps["uniform_init_weights"]
        cfg = self.config
        num_actions = len(self.skill_classes)
        if num_actions <= 1:
            raise ValueError("The full DreamerV3 backend needs at least two skill classes")
        stochastic_state_size = cfg.stochastic_size * cfg.discrete_size
        latent_size = stochastic_state_size + cfg.recurrent_state_size

        def mlp(
            input_dim: int,
            output_dim: int,
            *,
            layers: int,
            units: int | None = None,
        ):
            hidden = units or cfg.dense_units
            return MLP(
                input_dims=input_dim,
                output_dim=output_dim,
                hidden_sizes=[hidden] * layers,
                activation=nn.SiLU,
                flatten_dim=None,
                layer_args={"bias": False},
                norm_layer=LayerNorm,
                norm_args={"normalized_shape": hidden, "eps": 1e-3},
            )

        class _AgentDojoFullDreamer(nn.Module):
            def __init__(self):
                super().__init__()
                self.stochastic_size = cfg.stochastic_size
                self.discrete_size = cfg.discrete_size
                self.stochastic_state_size = stochastic_state_size
                self.latent_size = latent_size
                self.num_actions = num_actions
                self.encoder = MLPEncoder(
                    keys=["obs"],
                    input_dims=[cfg.obs_dim],
                    mlp_layers=cfg.encoder_layers,
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                recurrent = RecurrentModel(
                    input_size=stochastic_state_size + num_actions,
                    recurrent_state_size=cfg.recurrent_state_size,
                    dense_units=cfg.dense_units,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                representation = mlp(
                    cfg.recurrent_state_size + cfg.dense_units,
                    stochastic_state_size,
                    layers=1,
                )
                transition = mlp(
                    cfg.recurrent_state_size,
                    stochastic_state_size,
                    layers=1,
                )
                self.rssm = RSSM(
                    recurrent_model=recurrent,
                    representation_model=representation,
                    transition_model=transition,
                    distribution_cfg={"type": "discrete"},
                    discrete=cfg.discrete_size,
                    unimix=cfg.unimix,
                    learnable_initial_recurrent_state=True,
                )
                self.observation_model = MLPDecoder(
                    keys=["obs"],
                    output_dims=[cfg.obs_dim],
                    latent_state_size=latent_size,
                    mlp_layers=cfg.decoder_layers,
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                reward_input_size = latent_size + num_actions
                self.reward_model = mlp(
                    reward_input_size, cfg.reward_bins, layers=cfg.head_layers
                )
                self.continue_model = mlp(
                    reward_input_size, 1, layers=cfg.head_layers
                )
                self.skill_head = mlp(latent_size, num_actions, layers=cfg.head_layers)
                self.candidate_head = mlp(
                    latent_size, num_actions, layers=cfg.head_layers
                )
                self.risk_head = mlp(latent_size, 1, layers=cfg.head_layers)
                self.utility_head = mlp(latent_size, 1, layers=cfg.head_layers)
                self.preservation_head = mlp(
                    latent_size, 1, layers=cfg.head_layers
                )
                self.actor = Actor(
                    latent_state_size=latent_size,
                    actions_dim=[num_actions],
                    is_continuous=False,
                    distribution_cfg={"type": "discrete"},
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    mlp_layers=cfg.actor_layers,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                    unimix=cfg.unimix,
                )
                self.critic = mlp(
                    latent_size, cfg.reward_bins, layers=cfg.critic_layers
                )
                self.apply(init_weights)
                self.actor.mlp_heads.apply(uniform_init_weights(1.0))
                self.critic.model[-1].apply(uniform_init_weights(0.0))
                self.reward_model.model[-1].apply(uniform_init_weights(0.0))
                self.continue_model.model[-1].apply(uniform_init_weights(1.0))
                self.rssm.transition_model.model[-1].apply(
                    uniform_init_weights(1.0)
                )
                self.rssm.representation_model.model[-1].apply(
                    uniform_init_weights(1.0)
                )
                self.observation_model.heads.apply(uniform_init_weights(1.0))
                self.target_critic = copy.deepcopy(self.critic)
                self.target_critic.requires_grad_(False)
                self.register_buffer("return_low", torch.zeros(()))
                self.register_buffer("return_high", torch.zeros(()))
                self.configuration_value_head = None
                if cfg.configuration_value_head_enabled:
                    # Isolate initialization so adding this optional head does not
                    # perturb the baseline actor/world-model RNG stream.
                    with torch.random.fork_rng():
                        torch.manual_seed(cfg.seed + 104729)
                        self.configuration_value_head = mlp(
                            latent_size, 1, layers=cfg.head_layers
                        )
                        self.configuration_value_head.apply(init_weights)

            def world_parameters(self):
                modules = [
                    self.encoder,
                    self.rssm,
                    self.observation_model,
                    self.reward_model,
                    self.continue_model,
                    self.skill_head,
                    self.candidate_head,
                    self.risk_head,
                    self.utility_head,
                    self.preservation_head,
                ]
                if self.configuration_value_head is not None:
                    modules.append(self.configuration_value_head)
                for module in modules:
                    yield from module.parameters()

            def observe(self, obs, action_ids):
                batch, steps, _ = obs.shape
                encoded = self.encoder(
                    {"obs": obs.reshape(batch * steps, -1)}
                ).reshape(batch, steps, -1)
                previous_actions = torch.zeros(
                    batch, steps, num_actions, device=obs.device
                )
                if steps > 1:
                    previous_ids = action_ids[:, :-1].clamp_min(0)
                    previous_actions[:, 1:, :].scatter_(
                        2, previous_ids.unsqueeze(-1), 1.0
                    )
                recurrent_state, posterior = self.rssm.get_initial_states((1, batch))
                latent_states = []
                posterior_states = []
                recurrent_states = []
                posterior_logits = []
                prior_logits = []
                for index in range(steps):
                    is_first = torch.zeros(1, batch, 1, device=obs.device)
                    if index == 0:
                        is_first.fill_(1.0)
                    recurrent_state, posterior, _, post_logits, prior_logits_t = (
                        self.rssm.dynamic(
                            posterior,
                            recurrent_state,
                            previous_actions[:, index, :].unsqueeze(0),
                            encoded[:, index, :].unsqueeze(0),
                            is_first,
                        )
                    )
                    posterior_flat = posterior.reshape(1, batch, -1)
                    latent = torch.cat((posterior_flat, recurrent_state), dim=-1)
                    latent_states.append(latent.squeeze(0))
                    posterior_states.append(posterior_flat.squeeze(0))
                    recurrent_states.append(recurrent_state.squeeze(0))
                    posterior_logits.append(post_logits.squeeze(0))
                    prior_logits.append(prior_logits_t.squeeze(0))
                latent = torch.stack(latent_states, dim=1)
                current_actions = torch.zeros(
                    batch, steps, num_actions, device=obs.device
                )
                current_actions.scatter_(
                    2, action_ids.clamp_min(0).unsqueeze(-1), 1.0
                )
                reward_features = torch.cat((latent, current_actions), dim=-1)
                decoded = self.observation_model(
                    latent.reshape(batch * steps, -1)
                )["obs"].reshape(batch, steps, -1)
                output = {
                    "latent": latent,
                    "posterior_states": torch.stack(posterior_states, dim=1),
                    "recurrent_states": torch.stack(recurrent_states, dim=1),
                    "posterior_logits": torch.stack(posterior_logits, dim=1),
                    "prior_logits": torch.stack(prior_logits, dim=1),
                    "reconstruction": decoded,
                    "reward_logits": self.reward_model(reward_features),
                    "continue_logits": self.continue_model(reward_features).squeeze(-1),
                    "skill_logits": self.skill_head(latent),
                    "candidate_logits": self.candidate_head(latent),
                    "risk_logits": self.risk_head(latent).squeeze(-1),
                    "utility_logits": self.utility_head(latent).squeeze(-1),
                    "preservation_logits": self.preservation_head(latent).squeeze(-1),
                }
                if self.configuration_value_head is not None:
                    output["configuration_value_logits"] = (
                        self.configuration_value_head(latent).squeeze(-1)
                    )
                return output

        return _AgentDojoFullDreamer()

    def _ensure_module(self):
        if self._module is None:
            self._module = self._make_module().to(self._device_name())
        return self._module

    @contextmanager
    def _deterministic_inference(self, module):
        """Seed stochastic RSSM sampling without altering the training RNG."""

        torch = _require_full_sheeprl()["torch"]
        device = next(module.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed)
            yield

    def _vectorize_step(
        self, step: StepRecord | dict, attack_action: str | None = None
    ) -> np.ndarray:
        if self.config.observation_feature_mode == "precomputed":
            if self._observation_feature_index is None:
                self._load_precomputed_observation_features()
            assert self._observation_feature_index is not None
            assert self._observation_feature_matrix is not None
            key = observation_cache_key(step, attack_action)
            index = self._observation_feature_index.get(key)
            if index is None:
                raise KeyError(
                    "Observation is missing from the precomputed semantic cache: "
                    f"{key}. Rebuild the cache with every train/validation/test "
                    "record and any counterfactual attack actions used at inference."
                )
            return self._observation_feature_matrix[index]
        return hash_text_features(
            step_to_dreamer_text(step, attack_action), self.config.obs_dim
        )

    def _load_precomputed_observation_features(self) -> None:
        path_value = self.config.observation_feature_path
        if path_value is None:
            raise ValueError("Missing precomputed observation feature path")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Observation feature cache does not exist: {path}")
        with np.load(path, allow_pickle=False) as payload:
            keys = [str(value) for value in payload["keys"].tolist()]
            features = np.asarray(payload["features"], dtype=np.float32).copy()
        if features.ndim != 2 or features.shape[1] != self.config.obs_dim:
            raise ValueError(
                "Observation feature cache dimension mismatch: "
                f"expected {self.config.obs_dim}, got {features.shape}"
            )
        if len(keys) != len(features) or len(set(keys)) != len(keys):
            raise ValueError("Observation feature cache keys are missing or duplicated")
        self._observation_feature_matrix = features
        self._observation_feature_index = {
            key: index for index, key in enumerate(keys)
        }

    def _sequence_payload(self, sequence: list[StepRecord]) -> dict[str, Any]:
        length = len(sequence)
        obs = np.stack([self._vectorize_step(step) for step in sequence])
        actions = np.asarray(
            [self.skill_to_id[step.selected_skill] for step in sequence],
            dtype=np.int64,
        )
        candidate_mask = np.zeros((length, len(self.skill_classes)), dtype=np.float32)
        for index, step in enumerate(sequence):
            for skill in step.candidate_skills:
                if skill in self.skill_to_id:
                    candidate_mask[index, self.skill_to_id[skill]] = 1.0
            candidate_mask[index, actions[index]] = 1.0
        binary_risk = np.asarray(
            [float(step.attack_success) for step in sequence], dtype=np.float32
        )
        soft_risk = np.asarray(
            [
                step.attack_probability_target
                if step.attack_probability_target is not None
                else float(step.attack_success)
                for step in sequence
            ],
            dtype=np.float32,
        )
        risk_confidence = np.asarray(
            [
                max(
                    self.config.probability_confidence_floor,
                    step.attack_probability_confidence,
                )
                if step.attack_probability_target is not None
                else 1.0
                for step in sequence
            ],
            dtype=np.float32,
        )
        binary_utility = np.asarray(
            [float(step.task_success) for step in sequence], dtype=np.float32
        )
        soft_utility = np.asarray(
            [
                step.utility_probability_target
                if step.utility_probability_target is not None
                else float(step.task_success)
                for step in sequence
            ],
            dtype=np.float32,
        )
        preservation = np.asarray(
            [step.preservation_probability_target or 0.0 for step in sequence],
            dtype=np.float32,
        )
        preservation_mask = np.asarray(
            [step.preservation_probability_target is not None for step in sequence],
            dtype=np.float32,
        )
        preservation_weight = np.asarray(
            [
                step.preservation_weight
                if step.preservation_probability_target is not None
                else 0.0
                for step in sequence
            ],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [
                max(
                    self.config.probability_confidence_floor,
                    step.probability_label_confidence,
                )
                if step.utility_probability_target is not None
                else 1.0
                for step in sequence
            ],
            dtype=np.float32,
        )
        terminated = np.zeros(length, dtype=np.float32)
        terminated[-1] = 1.0
        reward = np.zeros(length, dtype=np.float32)
        final = sequence[-1]
        risk_group_key = (
            final.multiseed_group_id
            if final.attack_probability_target is not None
            else None
        )
        utility_group_key = (
            final.multiseed_group_id
            if final.attack_probability_target is not None
            and final.utility_probability_target is not None
            else None
        )
        utility_reward = (
            self.config.utility_reward_binary_mix * binary_utility[-1]
            + (1.0 - self.config.utility_reward_binary_mix) * soft_utility[-1]
        )
        risk_reward = (
            self.config.risk_reward_binary_mix * binary_risk[-1]
            + (1.0 - self.config.risk_reward_binary_mix) * soft_risk[-1]
        )
        reward[-1] = (
            self.config.risk_reward_scale * float(risk_reward)
            + self.config.utility_reward_scale * float(utility_reward)
            + self.config.target_skill_reward_scale * float(final.target_skill_success)
        )
        return {
            "obs": obs,
            "actions": actions,
            "candidate_mask": candidate_mask,
            "binary_risk": binary_risk,
            "soft_risk": soft_risk,
            "risk_confidence": risk_confidence,
            "binary_utility": binary_utility,
            "soft_utility": soft_utility,
            "preservation": preservation,
            "preservation_mask": preservation_mask,
            "preservation_weight": preservation_weight,
            "confidence": confidence,
            "reward": reward,
            "terminated": terminated,
            "length": length,
            "group_key": f"{sequence[0].domain}|{sequence[0].task_id}",
            "final_binary_utility": float(binary_utility[-1]),
            "risk_group_key": risk_group_key,
            "risk_group_expected_size": (
                int(final.multiseed_trials or 1) if risk_group_key is not None else 0
            ),
            "final_soft_risk": float(soft_risk[-1]),
            "final_risk_confidence": float(risk_confidence[-1]),
            "utility_group_key": utility_group_key,
            "utility_group_expected_size": (
                int(final.multiseed_trials or 1)
                if utility_group_key is not None
                else 0
            ),
            "final_soft_utility": float(soft_utility[-1]),
            "final_utility_confidence": float(confidence[-1]),
            "final_soft_value": float(soft_risk[-1] + soft_utility[-1]),
            "final_value_confidence": float(
                np.sqrt(risk_confidence[-1] * confidence[-1])
            ),
            "records": sequence,
        }

    def _prepare_sequences(self, steps: list[StepRecord]) -> list[dict[str, Any]]:
        return [self._sequence_payload(sequence) for sequence in _group_steps(steps)]

    def _collate(
        self,
        sequences: list[dict[str, Any]],
        device: str,
        calibration_members: np.ndarray | None = None,
    ):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        batch = len(sequences)
        max_len = max(sequence["length"] for sequence in sequences)
        actions_n = len(self.skill_classes)
        arrays = {
            "obs": np.zeros((batch, max_len, self.config.obs_dim), dtype=np.float32),
            "actions": np.zeros((batch, max_len), dtype=np.int64),
            "candidate_mask": np.zeros((batch, max_len, actions_n), dtype=np.float32),
            "binary_risk": np.zeros((batch, max_len), dtype=np.float32),
            "soft_risk": np.zeros((batch, max_len), dtype=np.float32),
            "risk_confidence": np.zeros((batch, max_len), dtype=np.float32),
            "binary_utility": np.zeros((batch, max_len), dtype=np.float32),
            "soft_utility": np.zeros((batch, max_len), dtype=np.float32),
            "preservation": np.zeros((batch, max_len), dtype=np.float32),
            "preservation_mask": np.zeros((batch, max_len), dtype=np.float32),
            "preservation_weight": np.zeros(
                (batch, max_len), dtype=np.float32
            ),
            "confidence": np.zeros((batch, max_len), dtype=np.float32),
            "reward": np.zeros((batch, max_len), dtype=np.float32),
            "terminated": np.ones((batch, max_len), dtype=np.float32),
            "mask": np.zeros((batch, max_len), dtype=np.float32),
        }
        group_to_id: dict[str, int] = {}
        group_ids = np.zeros(batch, dtype=np.int64)
        final_binary_utility = np.zeros(batch, dtype=np.float32)
        risk_group_to_id: dict[str, int] = {}
        risk_group_ids = np.full(batch, -1, dtype=np.int64)
        risk_group_expected_size = np.zeros(batch, dtype=np.int64)
        final_soft_risk = np.zeros(batch, dtype=np.float32)
        final_risk_confidence = np.zeros(batch, dtype=np.float32)
        risk_group_eligible = np.zeros(batch, dtype=np.float32)
        utility_group_to_id: dict[str, int] = {}
        utility_group_ids = np.full(batch, -1, dtype=np.int64)
        utility_group_expected_size = np.zeros(batch, dtype=np.int64)
        final_soft_utility = np.zeros(batch, dtype=np.float32)
        final_utility_confidence = np.zeros(batch, dtype=np.float32)
        final_soft_value = np.zeros(batch, dtype=np.float32)
        final_value_confidence = np.zeros(batch, dtype=np.float32)
        utility_group_eligible = np.zeros(batch, dtype=np.float32)
        if calibration_members is None:
            calibration_members = np.ones(batch, dtype=np.bool_)
        if len(calibration_members) != batch:
            raise ValueError("calibration_members must match the sequence batch")
        for row, sequence in enumerate(sequences):
            length = sequence["length"]
            for key in arrays:
                if key == "mask":
                    arrays[key][row, :length] = 1.0
                else:
                    arrays[key][row, :length] = sequence[key]
            group_key = str(sequence["group_key"])
            group_to_id.setdefault(group_key, len(group_to_id))
            group_ids[row] = group_to_id[group_key]
            final_binary_utility[row] = sequence["final_binary_utility"]
            risk_group_key = sequence.get("risk_group_key")
            if risk_group_key is not None:
                risk_group_key = str(risk_group_key)
                risk_group_to_id.setdefault(
                    risk_group_key, len(risk_group_to_id)
                )
                risk_group_ids[row] = risk_group_to_id[risk_group_key]
                risk_group_expected_size[row] = int(
                    sequence["risk_group_expected_size"]
                )
                final_soft_risk[row] = float(sequence["final_soft_risk"])
                final_risk_confidence[row] = float(
                    sequence["final_risk_confidence"]
                )
                risk_group_eligible[row] = 1.0
            utility_group_key = sequence.get("utility_group_key")
            if utility_group_key is not None:
                utility_group_key = str(utility_group_key)
                utility_group_to_id.setdefault(
                    utility_group_key, len(utility_group_to_id)
                )
                utility_group_ids[row] = utility_group_to_id[utility_group_key]
                utility_group_expected_size[row] = int(
                    sequence["utility_group_expected_size"]
                )
                final_soft_utility[row] = float(sequence["final_soft_utility"])
                final_utility_confidence[row] = float(
                    sequence["final_utility_confidence"]
                )
                final_soft_value[row] = float(sequence["final_soft_value"])
                final_value_confidence[row] = float(
                    sequence["final_value_confidence"]
                )
                utility_group_eligible[row] = 1.0
        output = {
            key: torch.from_numpy(value).to(device)
            for key, value in arrays.items()
        }
        output["group_ids"] = torch.from_numpy(group_ids).to(device)
        output["final_binary_utility"] = torch.from_numpy(
            final_binary_utility
        ).to(device)
        output["risk_group_ids"] = torch.from_numpy(risk_group_ids).to(device)
        output["risk_group_expected_size"] = torch.from_numpy(
            risk_group_expected_size
        ).to(device)
        output["final_soft_risk"] = torch.from_numpy(final_soft_risk).to(device)
        output["final_risk_confidence"] = torch.from_numpy(
            final_risk_confidence
        ).to(device)
        output["risk_group_eligible"] = torch.from_numpy(
            risk_group_eligible
        ).to(device)
        output["risk_calibration_member"] = torch.from_numpy(
            calibration_members.astype(np.float32)
        ).to(device)
        output["utility_group_ids"] = torch.from_numpy(utility_group_ids).to(device)
        output["utility_group_expected_size"] = torch.from_numpy(
            utility_group_expected_size
        ).to(device)
        output["final_soft_utility"] = torch.from_numpy(final_soft_utility).to(device)
        output["final_utility_confidence"] = torch.from_numpy(
            final_utility_confidence
        ).to(device)
        output["final_soft_value"] = torch.from_numpy(final_soft_value).to(device)
        output["final_value_confidence"] = torch.from_numpy(
            final_value_confidence
        ).to(device)
        output["utility_group_eligible"] = torch.from_numpy(
            utility_group_eligible
        ).to(device)
        output["utility_calibration_member"] = torch.from_numpy(
            calibration_members.astype(np.float32)
        ).to(device)
        return output

    @staticmethod
    def _weighted_mean(value, weight):
        return (value * weight).sum() / weight.sum().clamp_min(1e-6)

    def _world_losses(
        self,
        module,
        batch,
        out,
        *,
        include_group_utility: bool = True,
        include_group_value: bool = True,
    ):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        Independent = deps["Independent"]
        OneHot = deps["OneHotCategoricalStraightThrough"]
        kl_divergence = deps["kl_divergence"]
        SymlogDistribution = deps["SymlogDistribution"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        mask = batch["mask"]
        observation_loss = -SymlogDistribution(
            out["reconstruction"], dims=1, agg="mean"
        ).log_prob(batch["obs"])
        observation_loss = self._weighted_mean(observation_loss, mask)
        reward_loss = -TwoHot(out["reward_logits"], dims=1).log_prob(
            batch["reward"].unsqueeze(-1)
        )
        reward_loss = self._weighted_mean(reward_loss, mask)
        continue_target = 1.0 - batch["terminated"]
        continue_loss = F.binary_cross_entropy_with_logits(
            out["continue_logits"], continue_target, reduction="none"
        )
        continue_loss = self._weighted_mean(continue_loss, mask)

        post = out["posterior_logits"].reshape(
            *out["posterior_logits"].shape[:2],
            self.config.stochastic_size,
            self.config.discrete_size,
        )
        prior = out["prior_logits"].reshape_as(post)
        dynamic = kl_divergence(
            Independent(OneHot(logits=post.detach()), 1),
            Independent(OneHot(logits=prior), 1),
        )
        representation = kl_divergence(
            Independent(OneHot(logits=post), 1),
            Independent(OneHot(logits=prior.detach()), 1),
        )
        free_nats = torch.full_like(dynamic, self.config.kl_free_nats)
        kl_loss = (
            self.config.kl_dynamic_scale * torch.maximum(dynamic, free_nats)
            + self.config.kl_representation_scale
            * torch.maximum(representation, free_nats)
        )
        kl_loss = self._weighted_mean(kl_loss, mask)

        flat_mask = mask.reshape(-1) > 0
        skill_loss = F.cross_entropy(
            out["skill_logits"].reshape(-1, len(self.skill_classes))[flat_mask],
            batch["actions"].reshape(-1)[flat_mask],
        )
        candidate_loss = F.binary_cross_entropy_with_logits(
            out["candidate_logits"], batch["candidate_mask"], reduction="none"
        ).mean(dim=-1)
        candidate_loss = self._weighted_mean(candidate_loss, mask)
        sequence_lengths = mask.sum(dim=1).long().clamp_min(1)
        final_indices = sequence_lengths - 1
        batch_indices = torch.arange(mask.shape[0], device=mask.device)
        if self.config.risk_final_step_only:
            risk_logits_for_loss = out["risk_logits"][batch_indices, final_indices]
            binary_risk_for_loss = batch["binary_risk"][
                batch_indices, final_indices
            ]
            soft_risk_for_loss = batch["soft_risk"][batch_indices, final_indices]
            risk_probability_weight = batch["risk_confidence"][
                batch_indices, final_indices
            ]
            binary_risk_loss = F.binary_cross_entropy_with_logits(
                risk_logits_for_loss, binary_risk_for_loss
            )
            soft_risk_loss_raw = F.binary_cross_entropy_with_logits(
                risk_logits_for_loss, soft_risk_for_loss, reduction="none"
            )
            soft_risk_loss = self._weighted_mean(
                soft_risk_loss_raw, risk_probability_weight
            )
        else:
            binary_risk_loss = F.binary_cross_entropy_with_logits(
                out["risk_logits"], batch["binary_risk"], reduction="none"
            )
            binary_risk_loss = self._weighted_mean(binary_risk_loss, mask)
            risk_probability_weight = mask * batch["risk_confidence"]
            soft_risk_loss = F.binary_cross_entropy_with_logits(
                out["risk_logits"], batch["soft_risk"], reduction="none"
            )
            soft_risk_loss = self._weighted_mean(
                soft_risk_loss, risk_probability_weight
            )

        if self.config.group_risk_calibration_loss_scale > 0.0:
            calibration_logits = (
                module.risk_head(out["latent"].detach()).squeeze(-1)
                if self.config.group_risk_calibration_detach_latent
                else out["risk_logits"]
            )
            final_risk_probability = torch.sigmoid(
                calibration_logits[batch_indices, final_indices]
            )
            eligible = (
                (batch["risk_group_eligible"] > 0)
                & (batch["risk_calibration_member"] > 0)
                & (batch["risk_group_ids"] >= 0)
            )
            calibration_losses = []
            calibration_weights = []
            for group_id in torch.unique(batch["risk_group_ids"][eligible]):
                members = eligible & (batch["risk_group_ids"] == group_id)
                member_count = int(members.sum().item())
                expected_sizes = torch.unique(
                    batch["risk_group_expected_size"][members]
                )
                if len(expected_sizes) != 1:
                    raise ValueError("Inconsistent risk group size inside batch")
                expected_size = int(expected_sizes[0].item())
                if member_count != expected_size:
                    raise ValueError(
                        f"Incomplete risk calibration group: found {member_count}, "
                        f"expected {expected_size}"
                    )
                target_values = batch["final_soft_risk"][members]
                if not torch.allclose(
                    target_values,
                    target_values[:1].expand_as(target_values),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise ValueError("Inconsistent posterior target inside batch")
                group_prediction = final_risk_probability[members].mean()
                group_target = target_values.mean()
                calibration_losses.append((group_prediction - group_target).square())
                calibration_weights.append(
                    batch["final_risk_confidence"][members].mean()
                )
            if calibration_losses:
                group_risk_calibration_loss = self._weighted_mean(
                    torch.stack(calibration_losses),
                    torch.stack(calibration_weights),
                )
                group_risk_calibration_count = torch.as_tensor(
                    float(len(calibration_losses)), device=mask.device
                )
            else:
                # Clean-only batches have no attack-probability target.
                group_risk_calibration_loss = out["risk_logits"].sum() * 0.0
                group_risk_calibration_count = out["risk_logits"].sum() * 0.0
        else:
            group_risk_calibration_loss = out["risk_logits"].sum() * 0.0
            group_risk_calibration_count = out["risk_logits"].sum() * 0.0
        risk_loss = (
            self.config.binary_risk_loss_scale * binary_risk_loss
            + self.config.soft_risk_loss_scale * soft_risk_loss
            + self.config.group_risk_calibration_loss_scale
            * group_risk_calibration_loss
        )
        binary_utility_loss = F.binary_cross_entropy_with_logits(
            out["utility_logits"], batch["binary_utility"], reduction="none"
        )
        binary_utility_loss = self._weighted_mean(binary_utility_loss, mask)
        probability_weight = mask * batch["confidence"]
        soft_utility_loss = F.binary_cross_entropy_with_logits(
            out["utility_logits"], batch["soft_utility"], reduction="none"
        )
        soft_utility_loss = self._weighted_mean(
            soft_utility_loss, probability_weight
        )

        ranking_utility_logits = (
            module.utility_head(out["latent"].detach()).squeeze(-1)
            if self.config.utility_ranking_detach_latent
            else out["utility_logits"]
        )
        final_utility_logits = ranking_utility_logits[
            batch_indices, final_indices
        ]
        final_binary_utility = batch["final_binary_utility"]
        positive_pairs = (
            (batch["group_ids"][:, None] == batch["group_ids"][None, :])
            & (final_binary_utility[:, None] > final_binary_utility[None, :])
        )
        if positive_pairs.any():
            positive_index, negative_index = positive_pairs.nonzero(as_tuple=True)
            utility_ranking_loss = F.margin_ranking_loss(
                final_utility_logits[positive_index],
                final_utility_logits[negative_index],
                torch.ones_like(final_utility_logits[positive_index]),
                margin=self.config.utility_ranking_margin,
            )
            utility_ranking_pair_count = positive_pairs.sum().float()
        else:
            utility_ranking_loss = final_utility_logits.sum() * 0.0
            utility_ranking_pair_count = final_utility_logits.sum() * 0.0

        group_utility_calibration_loss = out["utility_logits"].sum() * 0.0
        group_utility_calibration_count = out["utility_logits"].sum() * 0.0
        group_utility_ranking_loss = out["utility_logits"].sum() * 0.0
        group_utility_ranking_pair_count = out["utility_logits"].sum() * 0.0
        if include_group_utility and (
            self.config.group_utility_calibration_loss_scale > 0.0
            or self.config.group_utility_ranking_loss_scale > 0.0
        ):
            eligible = (
                (batch["utility_group_eligible"] > 0)
                & (batch["utility_calibration_member"] > 0)
                & (batch["utility_group_ids"] >= 0)
            )
            group_members = []
            group_targets = []
            group_confidences = []
            group_task_ids: list[int] = []
            for group_id in torch.unique(batch["utility_group_ids"][eligible]):
                members = eligible & (batch["utility_group_ids"] == group_id)
                member_count = int(members.sum().item())
                expected_sizes = torch.unique(
                    batch["utility_group_expected_size"][members]
                )
                if len(expected_sizes) != 1:
                    raise ValueError("Inconsistent utility group size inside batch")
                expected_size = int(expected_sizes[0].item())
                if member_count != expected_size:
                    raise ValueError(
                        f"Incomplete utility calibration group: found {member_count}, "
                        f"expected {expected_size}"
                    )
                target_values = batch["final_soft_utility"][members]
                if not torch.allclose(
                    target_values,
                    target_values[:1].expand_as(target_values),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Inconsistent utility posterior target inside batch"
                    )
                task_values = torch.unique(batch["group_ids"][members])
                if len(task_values) != 1:
                    raise ValueError("Utility group spans multiple user tasks")
                group_members.append(members)
                group_targets.append(target_values.mean())
                group_confidences.append(
                    batch["final_utility_confidence"][members].mean()
                )
                group_task_ids.append(int(task_values[0].item()))

            if group_members:
                if self.config.group_utility_calibration_loss_scale > 0.0:
                    calibration_logits = (
                        module.utility_head(out["latent"].detach()).squeeze(-1)
                        if self.config.group_utility_calibration_detach_latent
                        else out["utility_logits"]
                    )
                    first_probabilities = torch.sigmoid(calibration_logits[:, 0])
                    group_predictions = torch.stack(
                        [first_probabilities[members].mean() for members in group_members]
                    )
                    target_tensor = torch.stack(group_targets)
                    confidence_tensor = torch.stack(group_confidences)
                    group_utility_calibration_loss = self._weighted_mean(
                        (group_predictions - target_tensor).square(),
                        confidence_tensor,
                    )
                    group_utility_calibration_count = torch.as_tensor(
                        float(len(group_members)), device=mask.device
                    )

                if self.config.group_utility_ranking_loss_scale > 0.0:
                    group_ranking_logits = (
                        module.utility_head(out["latent"].detach()).squeeze(-1)
                        if self.config.group_utility_ranking_detach_latent
                        else out["utility_logits"]
                    )
                    first_ranking_probabilities = torch.sigmoid(
                        group_ranking_logits[:, 0]
                    )
                    group_ranking_probabilities = torch.stack(
                        [
                            first_ranking_probabilities[members].mean()
                            for members in group_members
                        ]
                    )
                    group_ranking_scores = torch.logit(
                        group_ranking_probabilities.clamp(1e-5, 1.0 - 1e-5)
                    )
                    target_values = [
                        float(target.detach().cpu()) for target in group_targets
                    ]
                    ranking_pairs = _continuous_group_pair_indices(
                        group_task_ids,
                        target_values,
                        min_target_gap=self.config.group_utility_min_target_gap,
                        max_pairs_per_task=self.config.group_utility_pairs_per_task,
                    )
                    if ranking_pairs:
                        ranking_losses = []
                        ranking_weights = []
                        for high, low, target_gap in ranking_pairs:
                            ranking_losses.append(
                                F.softplus(
                                    -(
                                        group_ranking_scores[high]
                                        - group_ranking_scores[low]
                                    )
                                )
                            )
                            ranking_weights.append(
                                torch.as_tensor(target_gap, device=mask.device)
                                * torch.sqrt(
                                    group_confidences[high]
                                    * group_confidences[low]
                                )
                            )
                        group_utility_ranking_loss = self._weighted_mean(
                            torch.stack(ranking_losses),
                            torch.stack(ranking_weights),
                        )
                        group_utility_ranking_pair_count = torch.as_tensor(
                            float(len(ranking_pairs)), device=mask.device
                        )

        value_anchor = out.get(
            "configuration_value_logits", out["risk_logits"]
        )
        group_value_calibration_loss = value_anchor.sum() * 0.0
        group_value_calibration_count = value_anchor.sum() * 0.0
        group_value_ranking_loss = value_anchor.sum() * 0.0
        group_value_ranking_pair_count = value_anchor.sum() * 0.0
        if include_group_value and (
            self.config.group_value_calibration_loss_scale > 0.0
            or self.config.group_value_ranking_loss_scale > 0.0
        ):
            if module.configuration_value_head is None:
                raise RuntimeError("Configuration value head is not enabled")
            eligible = (
                (batch["utility_group_eligible"] > 0)
                & (batch["utility_calibration_member"] > 0)
                & (batch["utility_group_ids"] >= 0)
            )
            value_group_members = []
            value_group_targets = []
            value_group_confidences = []
            value_group_task_ids: list[int] = []
            for group_id in torch.unique(batch["utility_group_ids"][eligible]):
                members = eligible & (batch["utility_group_ids"] == group_id)
                member_count = int(members.sum().item())
                expected_sizes = torch.unique(
                    batch["utility_group_expected_size"][members]
                )
                if len(expected_sizes) != 1:
                    raise ValueError("Inconsistent value group size inside batch")
                expected_size = int(expected_sizes[0].item())
                if member_count != expected_size:
                    raise ValueError(
                        f"Incomplete value group: found {member_count}, "
                        f"expected {expected_size}"
                    )
                target_values = batch["final_soft_value"][members]
                if not torch.allclose(
                    target_values,
                    target_values[:1].expand_as(target_values),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Inconsistent configuration value target inside batch"
                    )
                task_values = torch.unique(batch["group_ids"][members])
                if len(task_values) != 1:
                    raise ValueError("Value group spans multiple user tasks")
                value_group_members.append(members)
                value_group_targets.append(target_values.mean())
                value_group_confidences.append(
                    batch["final_value_confidence"][members].mean()
                )
                value_group_task_ids.append(int(task_values[0].item()))

            if value_group_members:
                value_logits = module.configuration_value_head(
                    out["latent"].detach()
                ).squeeze(-1)
                first_value_probabilities = torch.sigmoid(value_logits[:, 0])
                group_value_probabilities = torch.stack(
                    [
                        first_value_probabilities[members].mean()
                        for members in value_group_members
                    ]
                )
                normalized_value_targets = torch.stack(
                    value_group_targets
                ) / 2.0
                confidence_tensor = torch.stack(value_group_confidences)
                if self.config.group_value_calibration_loss_scale > 0.0:
                    group_value_calibration_loss = self._weighted_mean(
                        (
                            group_value_probabilities
                            - normalized_value_targets
                        ).square(),
                        confidence_tensor,
                    )
                    group_value_calibration_count = torch.as_tensor(
                        float(len(value_group_members)), device=mask.device
                    )

                if self.config.group_value_ranking_loss_scale > 0.0:
                    ranking_scores = torch.logit(
                        group_value_probabilities.clamp(1e-5, 1.0 - 1e-5)
                    )
                    raw_targets = [
                        float(target.detach().cpu())
                        for target in value_group_targets
                    ]
                    ranking_pairs = _continuous_group_pair_indices(
                        value_group_task_ids,
                        raw_targets,
                        min_target_gap=self.config.group_value_min_target_gap,
                        max_pairs_per_task=self.config.group_value_pairs_per_task,
                    )
                    if ranking_pairs:
                        ranking_losses = []
                        ranking_weights = []
                        for high, low, target_gap in ranking_pairs:
                            ranking_losses.append(
                                F.softplus(
                                    -(ranking_scores[high] - ranking_scores[low])
                                )
                            )
                            ranking_weights.append(
                                torch.as_tensor(
                                    target_gap / 2.0, device=mask.device
                                )
                                * torch.sqrt(
                                    value_group_confidences[high]
                                    * value_group_confidences[low]
                                )
                            )
                        group_value_ranking_loss = self._weighted_mean(
                            torch.stack(ranking_losses),
                            torch.stack(ranking_weights),
                        )
                        group_value_ranking_pair_count = torch.as_tensor(
                            float(len(ranking_pairs)), device=mask.device
                        )
        utility_loss = (
            self.config.binary_utility_loss_scale * binary_utility_loss
            + self.config.soft_utility_loss_scale * soft_utility_loss
            + self.config.utility_ranking_loss_scale * utility_ranking_loss
            + self.config.group_utility_calibration_loss_scale
            * group_utility_calibration_loss
            + self.config.group_utility_ranking_loss_scale
            * group_utility_ranking_loss
        )
        preservation_weight = (
            mask
            * batch["preservation_mask"]
            * batch["preservation_weight"]
            * batch["confidence"]
        )
        preservation_loss_raw = F.binary_cross_entropy_with_logits(
            out["preservation_logits"],
            batch["preservation"],
            reduction="none",
        )
        if preservation_weight.sum() > 0:
            preservation_loss = self._weighted_mean(
                preservation_loss_raw, preservation_weight
            )
        else:
            preservation_loss = out["preservation_logits"].sum() * 0.0

        total = (
            self.config.observation_loss_scale * observation_loss
            + self.config.reward_loss_scale * reward_loss
            + self.config.continue_loss_scale * continue_loss
            + self.config.kl_loss_scale * kl_loss
            + self.config.skill_loss_scale * skill_loss
            + self.config.candidate_loss_scale * candidate_loss
            + self.config.risk_loss_scale * risk_loss
            + self.config.utility_loss_scale * utility_loss
            + self.config.preservation_loss_scale * preservation_loss
            + self.config.group_value_calibration_loss_scale
            * group_value_calibration_loss
            + self.config.group_value_ranking_loss_scale
            * group_value_ranking_loss
        )
        return {
            "world": total,
            "observation": observation_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl_loss,
            "skill": skill_loss,
            "candidate": candidate_loss,
            "risk": risk_loss,
            "binary_risk": binary_risk_loss,
            "soft_risk": soft_risk_loss,
            "group_risk_calibration": group_risk_calibration_loss,
            "group_risk_calibration_count": group_risk_calibration_count,
            "utility": utility_loss,
            "binary_utility": binary_utility_loss,
            "soft_utility": soft_utility_loss,
            "utility_ranking": utility_ranking_loss,
            "utility_ranking_pair_count": utility_ranking_pair_count,
            "group_utility_calibration": group_utility_calibration_loss,
            "group_utility_calibration_count": group_utility_calibration_count,
            "group_utility_ranking": group_utility_ranking_loss,
            "group_utility_ranking_pair_count": (
                group_utility_ranking_pair_count
            ),
            "group_value_calibration": group_value_calibration_loss,
            "group_value_calibration_count": group_value_calibration_count,
            "group_value_ranking": group_value_ranking_loss,
            "group_value_ranking_pair_count": group_value_ranking_pair_count,
            "preservation": preservation_loss,
        }

    def _valid_candidate_mask(self, module, latent, fallback_mask=None):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        probabilities = torch.sigmoid(module.candidate_head(latent))
        mask = probabilities >= self.config.candidate_threshold
        if fallback_mask is not None:
            mask = mask | fallback_mask.bool()
        empty = ~mask.any(dim=-1)
        if empty.any():
            best = probabilities[empty].argmax(dim=-1)
            mask[empty] = False
            mask[empty, best] = True
        finish_index = self.skill_to_id.get("finish")
        if finish_index is not None:
            mask[..., finish_index] = True
        return mask

    def _imagine(self, module, start, candidate_mask, actual_actions):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        compute_lambda_values = deps["compute_lambda_values"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        horizon = self.config.imagination_horizon
        stochastic = start[:, : module.stochastic_state_size]
        recurrent = start[:, module.stochastic_state_size :]
        state = start
        states = [state]
        imagined_actions = []
        log_probabilities = []
        entropies = []
        current_mask = candidate_mask.bool()
        for _ in range(horizon):
            actions, distributions = module.actor(
                state.detach(), mask={"mask_action_type": current_mask}
            )
            action = actions[0]
            imagined_actions.append(action.detach())
            distribution = distributions[0]
            log_probabilities.append(distribution.log_prob(action.detach()))
            entropies.append(distribution.entropy())
            with torch.no_grad():
                stochastic_next, recurrent_next = module.rssm.imagination(
                    stochastic.unsqueeze(0),
                    recurrent.unsqueeze(0),
                    action.detach().unsqueeze(0),
                )
                stochastic = stochastic_next.reshape(len(start), -1)
                recurrent = recurrent_next.squeeze(0)
                state = torch.cat((stochastic, recurrent), dim=-1)
                current_mask = self._valid_candidate_mask(module, state)
            states.append(state)
        imagined_states = torch.stack(states, dim=0)
        imagined_actions = torch.stack(imagined_actions, dim=0)
        log_probabilities = torch.stack(log_probabilities, dim=0)
        entropies = torch.stack(entropies, dim=0)
        with torch.no_grad():
            reward_features = torch.cat(
                (imagined_states[:-1], imagined_actions), dim=-1
            )
            rewards = TwoHot(
                module.reward_model(reward_features), dims=1
            ).mean
            continues = torch.sigmoid(
                module.continue_model(reward_features)
            )
            target_values = TwoHot(
                module.target_critic(imagined_states), dims=1
            ).mean
            lambda_values = compute_lambda_values(
                rewards,
                target_values[1:],
                continues * self.config.gamma,
                lmbda=self.config.lmbda,
            )
            baseline = TwoHot(module.critic(imagined_states[:-1]), dims=1).mean
            low = torch.quantile(lambda_values.float(), 0.05)
            high = torch.quantile(lambda_values.float(), 0.95)
            module.return_low.mul_(0.99).add_(low, alpha=0.01)
            module.return_high.mul_(0.99).add_(high, alpha=0.01)
            scale = torch.maximum(
                torch.ones_like(module.return_high),
                module.return_high - module.return_low,
            )
            advantage = (
                (lambda_values - module.return_low) / scale
                - (baseline - module.return_low) / scale
            )
            discounts = torch.cumprod(
                torch.cat(
                    (
                        torch.ones_like(continues[:1]),
                        continues[:-1] * self.config.gamma,
                    ),
                    dim=0,
                ),
                dim=0,
            )
        policy_loss = -(
            discounts.squeeze(-1)
            * (
                log_probabilities * advantage.detach().squeeze(-1)
                + self.config.entropy_scale * entropies
            )
        ).mean()
        actual_onehot = F.one_hot(
            actual_actions, num_classes=len(self.skill_classes)
        ).float()
        _, bc_distributions = module.actor(
            start.detach(), mask={"mask_action_type": candidate_mask.bool()}
        )
        behavior_cloning = -bc_distributions[0].log_prob(actual_onehot).mean()
        actor_loss = (
            policy_loss + self.config.behavior_cloning_scale * behavior_cloning
        )

        q_values = TwoHot(module.critic(imagined_states[:-1].detach()), dims=1)
        with torch.no_grad():
            target_prediction = TwoHot(
                module.target_critic(imagined_states[:-1]), dims=1
            ).mean
        critic_raw = -q_values.log_prob(lambda_values.detach())
        critic_raw = critic_raw - q_values.log_prob(target_prediction.detach())
        critic_loss = (
            critic_raw * discounts.squeeze(-1).detach()
        ).mean()
        return actor_loss, critic_loss, {
            "policy": policy_loss,
            "behavior_cloning": behavior_cloning,
            "imagined_return": lambda_values.mean(),
        }

    def _behavior_batch(self, out, batch):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        valid = batch["mask"].reshape(-1) > 0
        latent = out["latent"].reshape(-1, out["latent"].shape[-1])[valid]
        candidates = batch["candidate_mask"].reshape(
            -1, len(self.skill_classes)
        )[valid]
        actions = batch["actions"].reshape(-1)[valid]
        limit = self.config.imagination_batch_size
        if len(latent) > limit:
            indices = torch.randperm(len(latent), device=latent.device)[:limit]
            latent = latent[indices]
            candidates = candidates[indices]
            actions = actions[indices]
        return latent.detach(), candidates.detach(), actions.detach()

    @staticmethod
    def _update_target_critic(module, tau: float):
        with _require_full_sheeprl()["torch"].no_grad():
            for target, source in zip(
                module.target_critic.parameters(),
                module.critic.parameters(),
                strict=True,
            ):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def fit(
        self,
        train_steps: list[StepRecord],
        *,
        val_steps: list[StepRecord] | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        if not train_steps:
            raise ValueError("Cannot train the full DreamerV3 model with zero steps")
        vocabulary_steps = train_steps + (val_steps or [])
        discovered_skills = set(_build_vocab(vocabulary_steps))
        discovered_skills.update(self.skill_classes)
        self.skill_classes = sorted(discovered_skills)
        self.skill_to_id = {
            skill: index for index, skill in enumerate(self.skill_classes)
        }
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        module = self._make_module().to(self._device_name())
        self._module = module
        device = next(module.parameters()).device
        world_optimizer = torch.optim.AdamW(
            list(module.world_parameters()),
            lr=self.config.world_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        actor_optimizer = torch.optim.AdamW(
            module.actor.parameters(),
            lr=self.config.actor_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        critic_optimizer = torch.optim.AdamW(
            module.critic.parameters(),
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        sequences = self._prepare_sequences(train_steps)
        ranking_pairs = _build_ranking_pairs(sequences)
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        self.training_history = []
        best_state = None
        best_objective = float("inf")
        ranking_rng = np.random.default_rng(self.config.seed)
        head_only_group_updates = (
            self.config.group_utility_head_only_updates
            or self.config.group_value_head_only_updates
        )
        for epoch in range(1, epochs + 1):
            module.train()
            auxiliary_group_batches: list[np.ndarray] = []
            if head_only_group_updates:
                auxiliary_target_field = (
                    "final_soft_value"
                    if self.config.group_value_head_only_updates
                    else "final_soft_utility"
                )
                auxiliary_group_batches = [
                    indices
                    for indices in _multiseed_group_batches(
                        sequences,
                        batch_size,
                        ranking_rng,
                        group_key_field="utility_group_key",
                        expected_size_field="utility_group_expected_size",
                        target_field=auxiliary_target_field,
                        group_label=(
                            "value"
                            if self.config.group_value_head_only_updates
                            else "utility"
                        ),
                        task_aware=(
                            self.config.group_utility_ranking_loss_scale > 0.0
                            or self.config.group_value_ranking_loss_scale > 0.0
                        ),
                    )
                    if any(
                        sequences[int(index)].get("utility_group_key") is not None
                        for index in indices
                    )
                ]
                if not auxiliary_group_batches:
                    raise ValueError(
                        "Head-only grouped updates found no complete groups"
                    )
            if (
                self.config.grouped_utility_batches
                and not self.config.group_utility_head_only_updates
            ):
                epoch_batches = _multiseed_group_batches(
                    sequences,
                    batch_size,
                    ranking_rng,
                    group_key_field="utility_group_key",
                    expected_size_field="utility_group_expected_size",
                    target_field="final_soft_utility",
                    group_label="utility",
                    task_aware=self.config.group_utility_ranking_loss_scale > 0.0,
                )
            elif self.config.grouped_risk_calibration_batches:
                epoch_batches = _multiseed_group_batches(
                    sequences, batch_size, ranking_rng
                )
            elif (
                self.config.grouped_ranking_batches
                and self.config.utility_ranking_loss_scale > 0.0
            ):
                order = _grouped_ranking_order(sequences, ranking_rng)
                epoch_batches = [
                    order[start_index : start_index + batch_size]
                    for start_index in range(0, len(order), batch_size)
                ]
            else:
                order = np.random.permutation(len(sequences))
                epoch_batches = [
                    order[start_index : start_index + batch_size]
                    for start_index in range(0, len(order), batch_size)
                ]
            totals: dict[str, float] = {}
            updates = 0
            for base_indices in epoch_batches:
                calibration_members = None
                if (
                    self.config.utility_ranking_loss_scale > 0.0
                    and self.config.ranking_pairs_per_batch > 0
                ):
                    if (
                        self.config.grouped_risk_calibration_batches
                        or (
                            self.config.grouped_utility_batches
                            and not self.config.group_utility_head_only_updates
                        )
                    ):
                        batch_indices, calibration_members = _append_ranking_pairs(
                            base_indices,
                            ranking_pairs,
                            self.config.ranking_pairs_per_batch,
                            ranking_rng,
                        )
                    else:
                        batch_indices = _inject_ranking_pairs(
                            base_indices,
                            ranking_pairs,
                            self.config.ranking_pairs_per_batch,
                            ranking_rng,
                        )
                else:
                    batch_indices = base_indices
                selected = [
                    sequences[index]
                    for index in batch_indices
                ]
                batch = self._collate(
                    selected,
                    str(device),
                    calibration_members=calibration_members,
                )
                out = module.observe(batch["obs"], batch["actions"])
                losses = self._world_losses(
                    module,
                    batch,
                    out,
                    include_group_utility=(
                        not self.config.group_utility_head_only_updates
                    ),
                    include_group_value=(
                        not self.config.group_value_head_only_updates
                    ),
                )
                if head_only_group_updates:
                    auxiliary_indices = auxiliary_group_batches[
                        updates % len(auxiliary_group_batches)
                    ]
                    auxiliary_selected = [
                        sequences[int(index)] for index in auxiliary_indices
                    ]
                    auxiliary_batch = self._collate(
                        auxiliary_selected, str(device)
                    )
                    with self._deterministic_inference(module), torch.no_grad():
                        auxiliary_out = module.observe(
                            auxiliary_batch["obs"], auxiliary_batch["actions"]
                        )
                    auxiliary_losses = self._world_losses(
                        module,
                        auxiliary_batch,
                        auxiliary_out,
                        include_group_utility=True,
                        include_group_value=True,
                    )
                    auxiliary_loss = (
                        auxiliary_losses["group_value_calibration"] * 0.0
                        if self.config.group_value_head_only_updates
                        else auxiliary_losses["group_utility_calibration"] * 0.0
                    )
                    if self.config.group_utility_head_only_updates:
                        auxiliary_loss = auxiliary_loss + (
                            self.config.utility_loss_scale
                            * (
                                self.config.group_utility_calibration_loss_scale
                                * auxiliary_losses["group_utility_calibration"]
                                + self.config.group_utility_ranking_loss_scale
                                * auxiliary_losses["group_utility_ranking"]
                            )
                        )
                    if self.config.group_value_head_only_updates:
                        auxiliary_loss = auxiliary_loss + (
                            self.config.group_value_calibration_loss_scale
                            * auxiliary_losses["group_value_calibration"]
                            + self.config.group_value_ranking_loss_scale
                            * auxiliary_losses["group_value_ranking"]
                        )
                    losses["world"] = losses["world"] + auxiliary_loss
                    for key in (
                        "group_utility_calibration",
                        "group_utility_calibration_count",
                        "group_utility_ranking",
                        "group_utility_ranking_pair_count",
                        "group_value_calibration",
                        "group_value_calibration_count",
                        "group_value_ranking",
                        "group_value_ranking_pair_count",
                    ):
                        losses[key] = auxiliary_losses[key]
                    losses["group_head_only_auxiliary"] = auxiliary_loss
                    if self.config.group_utility_head_only_updates:
                        losses["group_utility_head_only_auxiliary"] = (
                            auxiliary_loss
                        )
                    if self.config.group_value_head_only_updates:
                        losses["group_value_head_only_auxiliary"] = auxiliary_loss
                world_optimizer.zero_grad(set_to_none=True)
                losses["world"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(module.world_parameters()), self.config.world_gradient_clip
                )
                world_optimizer.step()

                with torch.no_grad():
                    refreshed = module.observe(batch["obs"], batch["actions"])
                latent, candidates, actual_actions = self._behavior_batch(
                    refreshed, batch
                )
                actor_loss, critic_loss, behavior = self._imagine(
                    module, latent, candidates, actual_actions
                )
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.actor.parameters(), self.config.actor_gradient_clip
                )
                actor_optimizer.step()
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.critic.parameters(), self.config.critic_gradient_clip
                )
                critic_optimizer.step()
                self._update_target_critic(module, self.config.target_critic_tau)

                scalars = {
                    **{key: value for key, value in losses.items()},
                    "actor": actor_loss,
                    "critic": critic_loss,
                    **behavior,
                }
                for key, value in scalars.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                updates += 1
            history: dict[str, Any] = {
                "epoch": epoch,
                **{key: value / max(updates, 1) for key, value in totals.items()},
            }
            if val_steps:
                val_metrics = evaluate_full_dreamer_predictions(
                    val_steps,
                    self.predict(val_steps),
                    validation_risk_mode=self.config.validation_risk_mode,
                    validation_utility_mode=self.config.validation_utility_mode,
                    validation_aggregation=self.config.validation_aggregation,
                    validation_group_step=self.config.validation_group_step,
                )
                history["validation"] = val_metrics
                if (
                    self.config.checkpoint_objective
                    == "grouped_configuration_value_brier"
                ):
                    objective = val_metrics[
                        "grouped_configuration_value_normalized_brier_score"
                    ]
                    if objective is None:
                        raise ValueError(
                            "Configuration-value checkpointing requested but "
                            "validation has no grouped value metric"
                        )
                else:
                    objective = val_metrics["validation_objective"]
            else:
                objective = history["world"]
            self.training_history.append(history)
            if objective < best_objective:
                best_objective = objective
                self.best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        return self

    def _predict_sequence_batches(self, steps: list[StepRecord]):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        module = self._ensure_module()
        device = next(module.parameters()).device
        sequences = self._prepare_sequences(steps)
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        module.eval()
        with self._deterministic_inference(module), torch.no_grad():
            for start in range(0, len(sequences), 64):
                selected = sequences[start : start + 64]
                batch = self._collate(selected, str(device))
                out = module.observe(batch["obs"], batch["actions"])
                for index, sequence in enumerate(selected):
                    length = sequence["length"]
                    latent = out["latent"][index, :length]
                    candidates = batch["candidate_mask"][index, :length].bool()
                    _, distributions = module.actor(
                        latent, mask={"mask_action_type": candidates}
                    )
                    skill_proba = distributions[0].probs
                    value = TwoHot(module.critic(latent), dims=1).mean.squeeze(-1)
                    reward = TwoHot(
                        out["reward_logits"][index, :length], dims=1
                    ).mean.squeeze(-1)
                    for offset, record in enumerate(sequence["records"]):
                        rows[(record.trajectory_id, record.step_id)] = {
                            "skill_proba": skill_proba[offset].cpu().numpy(),
                            "risk": float(
                                torch.sigmoid(out["risk_logits"][index, offset]).cpu()
                            ),
                            "utility": float(
                                torch.sigmoid(out["utility_logits"][index, offset]).cpu()
                            ),
                            "preservation": float(
                                torch.sigmoid(
                                    out["preservation_logits"][index, offset]
                                ).cpu()
                            ),
                            "value": float(value[offset].cpu()),
                            "reward": float(reward[offset].cpu()),
                        }
                        if "configuration_value_logits" in out:
                            rows[(record.trajectory_id, record.step_id)][
                                "configuration_value"
                            ] = float(
                                2.0
                                * torch.sigmoid(
                                    out["configuration_value_logits"][
                                        index, offset
                                    ]
                                ).cpu()
                            )
        return rows

    def predict(self, steps: list[StepRecord | dict]) -> dict[str, Any]:
        records = [
            step if isinstance(step, StepRecord) else StepRecord.model_validate(step)
            for step in steps
        ]
        rows = self._predict_sequence_batches(records)
        ordered = [rows[(step.trajectory_id, step.step_id)] for step in records]
        probabilities = np.stack([row["skill_proba"] for row in ordered])
        classes = np.asarray(self.skill_classes)
        result = {
            "next_skill": classes[np.argmax(probabilities, axis=1)],
            "next_skill_proba": probabilities,
            "skill_classes": classes,
            "risk_score": np.asarray([row["risk"] for row in ordered]),
            "utility_score": np.asarray([row["utility"] for row in ordered]),
            "preservation_score": np.asarray(
                [row["preservation"] for row in ordered]
            ),
            "value_score": np.asarray([row["value"] for row in ordered]),
            "reward_score": np.asarray([row["reward"] for row in ordered]),
        }
        if all("configuration_value" in row for row in ordered):
            result["configuration_value_score"] = np.asarray(
                [row["configuration_value"] for row in ordered]
            )
        return result

    def score_actions(
        self, step: StepRecord | dict, actions: list[str]
    ) -> dict[str, Any]:
        repeated = []
        base = step.model_dump(mode="json") if isinstance(step, StepRecord) else dict(step)
        for index, action in enumerate(actions):
            row = dict(base)
            row["attack_action"] = action
            row["trajectory_id"] = f"{base['trajectory_id']}::candidate::{index}"
            repeated.append(StepRecord.model_validate(row))
        return self.predict(repeated)

    def rollout_score_step(
        self, step: StepRecord | dict, *, horizon: int = 5
    ) -> dict[str, Any]:
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        module = self._ensure_module()
        device = next(module.parameters()).device
        record = step if isinstance(step, StepRecord) else StepRecord.model_validate(step)
        payload = self._sequence_payload([record])
        batch = self._collate([payload], str(device))
        branch_skills = [
            skill for skill in record.candidate_skills if skill in self.skill_to_id
        ]
        if not branch_skills:
            branch_skills = ["finish"] if "finish" in self.skill_to_id else [self.skill_classes[0]]
        target_index = self.skill_to_id.get(record.target_skill) if record.target_skill else None
        summaries = []
        module.eval()
        with self._deterministic_inference(module), torch.no_grad():
            observed = module.observe(batch["obs"], batch["actions"])
            base_stochastic = observed["posterior_states"][:, -1]
            base_recurrent = observed["recurrent_states"][:, -1]
            for first_skill in branch_skills:
                stochastic = base_stochastic.clone()
                recurrent = base_recurrent.clone()
                latent = torch.cat((stochastic, recurrent), dim=-1)
                action_id = self.skill_to_id[first_skill]
                imagined_skills = []
                risks, utilities, preservations, values, rewards, target_probs = (
                    [], [], [], [], [], []
                )
                for rollout_index in range(max(1, horizon)):
                    action = torch.zeros(1, 1, len(self.skill_classes), device=device)
                    action[:, :, action_id] = 1.0
                    reward_features = torch.cat(
                        (latent, action.squeeze(0)), dim=-1
                    )
                    predicted_reward = TwoHot(
                        module.reward_model(reward_features), dims=1
                    ).mean
                    stochastic_next, recurrent_next = module.rssm.imagination(
                        stochastic.unsqueeze(0), recurrent.unsqueeze(0), action
                    )
                    stochastic = stochastic_next.reshape(1, -1)
                    recurrent = recurrent_next.squeeze(0)
                    latent = torch.cat((stochastic, recurrent), dim=-1)
                    candidate_mask = self._valid_candidate_mask(module, latent)
                    _, distributions = module.actor(
                        latent, mask={"mask_action_type": candidate_mask}
                    )
                    probabilities = distributions[0].probs
                    next_id = int(probabilities.argmax(dim=-1).item())
                    imagined_skills.append(
                        first_skill if rollout_index == 0 else self.skill_classes[next_id]
                    )
                    risks.append(float(torch.sigmoid(module.risk_head(latent)).item()))
                    utilities.append(float(torch.sigmoid(module.utility_head(latent)).item()))
                    preservations.append(
                        float(torch.sigmoid(module.preservation_head(latent)).item())
                    )
                    values.append(float(TwoHot(module.critic(latent), dims=1).mean.item()))
                    rewards.append(float(predicted_reward.item()))
                    target_probs.append(
                        float(probabilities[0, target_index].item())
                        if target_index is not None
                        else 0.0
                    )
                    action_id = next_id
                selection_score = (
                    values[0]
                    + max(risks)
                    + float(np.mean(preservations))
                    + 0.25 * max(target_probs)
                )
                summaries.append(
                    {
                        "branch_first_skill": first_skill,
                        "risk_score": max(risks),
                        "utility_score": float(np.mean(utilities)),
                        "preservation_score": float(np.mean(preservations)),
                        "min_utility_score": min(utilities),
                        "final_utility_score": utilities[-1],
                        "value_score": values[0],
                        "reward_score": float(np.mean(rewards)),
                        "target_skill_probability": max(target_probs),
                        "selection_score": selection_score,
                        "rollout_imagined_skills": imagined_skills,
                        "rollout_target_reached": float(
                            record.target_skill in imagined_skills
                        )
                        if record.target_skill
                        else 0.0,
                    }
                )
        best = max(summaries, key=lambda row: row["selection_score"])
        return {
            **best,
            "rollout_backend": "sheeprl_full_dreamer_v3_actor_critic",
            "rollout_branch_count": len(summaries),
            "rollout_top_branch_summaries": sorted(
                summaries, key=lambda row: row["selection_score"], reverse=True
            )[:3],
        }

    def model_info(self) -> dict[str, Any]:
        module = self._ensure_module()
        components: dict[str, int] = {}
        for name, parameter in module.named_parameters():
            key = name.split(".")[0]
            components[key] = components.get(key, 0) + parameter.numel()
        return {
            "backend": "sheeprl_full_dreamer_v3_offline",
            "parameter_count": sum(p.numel() for p in module.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
            "parameters_by_component": components,
            "latent_size": (
                self.config.recurrent_state_size
                + self.config.stochastic_size * self.config.discrete_size
            ),
            "skill_class_count": len(self.skill_classes),
            "configuration_value_head_enabled": (
                self.config.configuration_value_head_enabled
            ),
            "checkpoint_objective": self.config.checkpoint_objective,
            "observation_feature_mode": self.config.observation_feature_mode,
            "observation_feature_path": self.config.observation_feature_path,
            "sheeprl_components": [
                "MLPEncoder",
                "MLPDecoder",
                "RSSM",
                "RecurrentModel",
                "Actor",
                "TwoHotEncodingDistribution",
                "lambda_return",
            ],
        }

    def save(self, path: str | Path) -> None:
        torch = _require_full_sheeprl()["torch"]
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        module = self._ensure_module()
        torch.save(module.state_dict(), path / "model.pt")
        metadata = {
            "config": asdict(self.config),
            "skill_classes": self.skill_classes,
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
            **self.model_info(),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path):
        torch = _require_full_sheeprl()["torch"]
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(
            config=FullDreamerV3Config(**metadata["config"]),
            skill_classes=metadata["skill_classes"],
        )
        model.training_history = metadata.get("training_history", [])
        model.best_epoch = metadata.get("best_epoch")
        module = model._ensure_module()
        state = torch.load(path / "model.pt", map_location=model._device_name())
        module.load_state_dict(state)
        module.eval()
        return model
