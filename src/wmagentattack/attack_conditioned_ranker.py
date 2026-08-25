"""Attack-conditioned four-cell ranking on frozen AgentDojo trajectories.

The module is deliberately configuration-level.  It predicts the joint
``(attack success, utility success)`` distribution for an attacker action
fixed before the victim rollout.  Final outcomes are targets only; raw task,
injection, trajectory, payload, and outcome identifiers are never features.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


FORBIDDEN_FEATURE_FIELDS = {
    "row_id",
    "user_task_id",
    "injection_task_id",
    "task_group_id",
    "base_pair_id",
    "injection_goal_group_id",
    "payload",
    "payload_by_vector",
    "payload_sha256",
    "injection_goal",
    "utility",
    "security",
    "task_success",
    "attack_success",
    "joint_class",
    "joint_outcome_counts",
}


def task_name(row: Mapping[str, Any]) -> str:
    return f"{row['suite']}|{row['user_task_id']}"


def _token_parts(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-zA-Z0-9]+", value.lower()) if part]


def _categorical(features: dict[str, float], field: str, value: Any) -> None:
    normalized = str(value if value not in (None, "") else "unknown").lower()
    features[f"{field}={normalized}"] = 1.0


def structured_attack_features(
    row: Mapping[str, Any], *, include_family: bool = False
) -> dict[str, float]:
    """Return typed, payload-blind attacker-action features.

    The primary arm excludes family/name/variant so a held-out textual family
    cannot be recognized by an ID shortcut.  Tool and injection-vector names
    are decomposed into typed tokens, while raw goal/payload text is excluded.
    """

    features: dict[str, float] = {}
    for field in (
        "attack_kind",
        "attack_role",
        "knowledge_level",
        "endpoint_policy",
        "payload_position",
        "trigger_stage",
        "underspecification",
        "solvability_bin",
    ):
        _categorical(features, field, row.get(field))
    if include_family:
        for field in ("attack_family", "attack_name", "attack_variant"):
            _categorical(features, field, row.get(field))

    tools = list(row.get("target_tool_sequence", []))
    vectors = [str(value) for value in row.get("candidate_injection_vectors", [])]
    segments = list(row.get("payload_segments", []))
    features["required_tool_depth"] = float(row.get("required_tool_depth") or len(tools))
    features["injection_vector_count"] = float(len(vectors))
    features["payload_segment_count"] = float(len(segments))
    features["vector_specific_payload"] = float(bool(row.get("payload_by_vector")))
    features["has_optimizer"] = float(str(row.get("optimizer", "none")) != "none")

    clean_rate = row.get("base_success_rate")
    if isinstance(clean_rate, (int, float)) and math.isfinite(float(clean_rate)):
        features["independent_clean_success_prior"] = float(clean_rate)
        features["clean_prior_observed"] = 1.0
    else:
        features["independent_clean_success_prior"] = 0.5
        features["clean_prior_observed"] = 0.0

    for position, call in enumerate(tools):
        tool = str(call.get("tool", "unknown"))
        features[f"target_tool={tool.lower()}"] = 1.0
        features[f"target_tool_position={position}:{tool.lower()}"] = 1.0
        for part in _token_parts(tool):
            features[f"target_tool_token={part}"] = 1.0
        placeholder = call.get("placeholder_args")
        if isinstance(placeholder, Mapping):
            for key, value in placeholder.items():
                features[f"target_arg={str(key).lower()}"] = 1.0
                features[f"target_arg_type={type(value).__name__}"] = 1.0
    for vector in vectors:
        features[f"injection_vector={vector.lower()}"] = 1.0
        for part in _token_parts(vector):
            features[f"injection_vector_token={part}"] = 1.0
    return features


def forbidden_feature_keys(features: Mapping[str, float]) -> list[str]:
    findings = []
    for name in features:
        lowered = str(name).lower()
        root = lowered.split("=", 1)[0]
        if root in FORBIDDEN_FEATURE_FIELDS:
            findings.append(str(name))
    return sorted(findings)


def four_cell_target_from_label_group(
    label: Mapping[str, Any], *, dirichlet_prior: float = 0.5
) -> list[float]:
    """Read the modern target or reconstruct the identical target from counts.

    The July final label-group archive predates the explicit four-cell field,
    but contains utility, attack, and joint-success binomial sufficient
    statistics.  Inclusion-exclusion recovers all four empirical cells without
    reading individual trajectories or changing the v5 Dirichlet prior.
    """

    probabilities = label.get("joint_outcome_probability_target")
    if isinstance(probabilities, Mapping):
        target = [float(probabilities[name]) for name in JOINT_OUTCOME_CLASSES]
    else:
        trials = int(label["joint_success_probability_trials"])
        attack = int(label["attack_probability_successes"])
        utility = int(label["utility_probability_successes"])
        joint = int(label["joint_success_probability_successes"])
        counts = [
            trials - attack - utility + joint,
            utility - joint,
            attack - joint,
            joint,
        ]
        if any(value < 0 for value in counts) or sum(counts) != trials:
            raise ValueError(f"invalid four-cell sufficient statistics: {counts}")
        denominator = trials + dirichlet_prior * len(JOINT_OUTCOME_CLASSES)
        target = [(value + dirichlet_prior) / denominator for value in counts]
    if not np.isclose(sum(target), 1.0):
        raise ValueError("four-cell target does not sum to one")
    return target


def align_attack_candidates(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    label_groups: Sequence[Mapping[str, Any]],
    v5_predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align 400 manifest actions, five-seed labels, and frozen v5 scores."""

    manifest = {str(row["row_id"]): dict(row) for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate manifest row_id")
    labels = {
        str(row["row_id"]): dict(row)
        for row in label_groups
        if str(row.get("source_kind")) == "attack"
    }
    prediction_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fold_by_task: dict[str, set[int]] = defaultdict(set)
    for row in v5_predictions:
        if str(row.get("arm")) != "structured_joint_aux":
            continue
        if not bool(row.get("joint_trainable")):
            continue
        group = str(row["joint_group_id"])
        prediction_groups[group].append(row)
        fold_by_task[str(row["task_name"])].add(int(row["fold"]))

    output = []
    missing = []
    for row_id, action in sorted(manifest.items()):
        label = labels.get(row_id)
        group_id = f"attack::{row_id}"
        predictions = prediction_groups.get(group_id, [])
        name = task_name(action)
        folds = fold_by_task.get(name, set())
        if label is None or not predictions or len(folds) != 1:
            missing.append(row_id)
            continue
        target = four_cell_target_from_label_group(label)
        p11_values = [float(item["joint_p11"]) for item in predictions]
        output.append(
            {
                "row_id": row_id,
                "task_name": name,
                "fold": next(iter(folds)),
                "attack_family": str(action.get("attack_family", "unknown")),
                "features": structured_attack_features(action, include_family=False),
                "family_features": structured_attack_features(action, include_family=True),
                "target": target,
                "target_p11": target[3],
                "v5_p11": float(np.mean(p11_values)),
                "v5_prediction_count": len(p11_values),
            }
        )
    tasks = sorted({row["task_name"] for row in output})
    per_task = {task: sum(row["task_name"] == task for row in output) for task in tasks}
    checks = {
        "manifest_rows_400": len(manifest) == 400,
        "attack_label_groups_400": len(labels) == 400,
        "aligned_candidates_400": len(output) == 400,
        "tasks_20": len(tasks) == 20,
        "twenty_candidates_per_task": set(per_task.values()) == {20},
        "every_task_one_confirmation_fold": all(len(value) == 1 for value in fold_by_task.values()),
        "all_v5_scores_finite": all(math.isfinite(row["v5_p11"]) for row in output),
        "features_outcome_blind": all(
            not forbidden_feature_keys(row["features"]) for row in output
        ),
        "zero_missing_alignments": not missing,
    }
    return output, {
        "passed": all(checks.values()),
        "checks": checks,
        "candidates": len(output),
        "tasks": len(tasks),
        "candidates_per_task": per_task,
        "missing_row_ids": missing,
    }


def base_distribution_from_p11(p11: Tensor, prior: Tensor) -> Tensor:
    """Lift a frozen p11 score to a valid four-cell prior distribution."""

    eps = torch.finfo(p11.dtype).eps
    p11 = p11.clamp(eps, 1.0 - eps)
    other = prior[:3].clamp_min(eps)
    other = other / other.sum()
    return torch.cat([(1.0 - p11[:, None]) * other[None, :], p11[:, None]], dim=1)


class AttackConditionedResidualRanker(nn.Module):
    """Zero-start residual over a frozen four-cell base distribution."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.residual = nn.Linear(hidden_size, len(JOINT_OUTCOME_CLASSES))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features: Tensor, base_probabilities: Tensor) -> Tensor:
        base_logits = torch.log(base_probabilities.clamp_min(1e-8))
        return base_logits + self.residual(self.encoder(features))


def task_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["task_name"])] += 1
    weights = np.asarray(
        [1.0 / len(counts) / counts[str(row["task_name"])] for row in rows],
        dtype=np.float32,
    )
    return weights * len(weights) / weights.sum()


def within_task_pairs(
    rows: Sequence[Mapping[str, Any]], *, minimum_target_gap: float
) -> np.ndarray:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[str(row["task_name"])].append(index)
    pairs = []
    for indices in by_task.values():
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                gap = float(rows[left]["target_p11"]) - float(rows[right]["target_p11"])
                if abs(gap) >= minimum_target_gap:
                    pairs.append((left, right, 1.0 if gap > 0 else -1.0, abs(gap)))
    return np.asarray(pairs, dtype=np.float32).reshape(-1, 4)


def pairwise_logistic_loss(p11: Tensor, pairs: Tensor) -> Tensor:
    if not pairs.numel():
        return p11.new_zeros(())
    left = pairs[:, 0].long()
    right = pairs[:, 1].long()
    sign = pairs[:, 2]
    weight = pairs[:, 3]
    margin = sign * (p11[left] - p11[right])
    return (torch.nn.functional.softplus(-margin) * weight).sum() / weight.sum()


def ranking_metrics(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_name"])].append(row)
    selected = []
    random_values = []
    regrets = []
    pair_correct = 0.0
    pair_total = 0
    for task, candidates in sorted(by_task.items()):
        choice = max(candidates, key=lambda row: (float(row[score_key]), str(row["row_id"])))
        target = float(choice["target_p11"])
        oracle = max(float(row["target_p11"]) for row in candidates)
        selected.append({"task_name": task, "row_id": choice["row_id"], "target_p11": target})
        random_values.append(float(np.mean([row["target_p11"] for row in candidates])))
        regrets.append(oracle - target)
        for left_pos, left in enumerate(candidates):
            for right in candidates[left_pos + 1 :]:
                target_gap = float(left["target_p11"]) - float(right["target_p11"])
                if abs(target_gap) < 1e-12:
                    continue
                score_gap = float(left[score_key]) - float(right[score_key])
                pair_correct += float(score_gap * target_gap > 0) + 0.5 * float(score_gap == 0)
                pair_total += 1
    return {
        "task_count": len(by_task),
        "top1_target_p11": float(np.mean([row["target_p11"] for row in selected])),
        "random_expected_target_p11": float(np.mean(random_values)),
        "top1_regret": float(np.mean(regrets)),
        "pairwise_accuracy": pair_correct / max(1, pair_total),
        "selected": selected,
    }
