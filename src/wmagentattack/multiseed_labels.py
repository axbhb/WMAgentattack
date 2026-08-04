"""Configuration-level probability labels from repeated AgentDojo runs.

Every attacked manifest row is repeated with the same fixed set of victim-model
sampling seeds.  This module validates that rectangular design and replaces a
single Bernoulli label with a Jeffreys-posterior mean.  Clean repetitions are
kept separate: they estimate task solvability and define the conditional
preservation target, but they are never treated as attacked observations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


JOINT_OUTCOME_KEYS = (
    "attack0_utility0",
    "attack0_utility1",
    "attack1_utility0",
    "attack1_utility1",
)


@dataclass(frozen=True)
class BetaPosterior:
    successes: int
    trials: int
    alpha: float
    beta: float
    mean: float
    variance: float
    confidence: float


def jeffreys_posterior(successes: int, trials: int) -> BetaPosterior:
    """Return a Beta(successes + 1/2, failures + 1/2) posterior."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    alpha = successes + 0.5
    beta = trials - successes + 0.5
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    # A Beta(1, 1) distribution has variance 1/12.  The score is zero at
    # that uncertainty and approaches one as repeated evidence accumulates.
    confidence = max(0.0, min(1.0, 1.0 - 12.0 * variance))
    return BetaPosterior(
        successes=successes,
        trials=trials,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
        confidence=confidence,
    )


def _single_value(rows: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Inconsistent {key} within repeated group: {sorted(values)!r}")
    return next(iter(values))


def _posterior_fields(prefix: str, posterior: BetaPosterior) -> dict[str, Any]:
    return {
        f"{prefix}_successes": posterior.successes,
        f"{prefix}_trials": posterior.trials,
        f"{prefix}_alpha": posterior.alpha,
        f"{prefix}_beta": posterior.beta,
        f"{prefix}_target": posterior.mean,
        f"{prefix}_variance": posterior.variance,
        f"{prefix}_confidence": posterior.confidence,
    }


def _safe_logit(probability: float, epsilon: float = 1e-6) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, probability))
    return math.log(clipped / (1.0 - clipped))


def joint_outcome_posterior(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    """Return four-cell counts and a Jeffreys Dirichlet posterior.

    Bit order is always attack then utility.  A symmetric one-half prior is
    the multinomial analogue of the Jeffreys Beta prior used for the marginal
    labels in this module.
    """

    counts = {key: 0 for key in JOINT_OUTCOME_KEYS}
    for row in rows:
        key = (
            f"attack{int(bool(row.get('security')))}_"
            f"utility{int(bool(row.get('utility')))}"
        )
        counts[key] += 1
    alpha = {key: counts[key] + 0.5 for key in JOINT_OUTCOME_KEYS}
    total = sum(alpha.values())
    probabilities = {key: alpha[key] / total for key in JOINT_OUTCOME_KEYS}
    return counts, alpha, probabilities


def build_multiseed_labels(
    metadata: Iterable[dict[str, Any]],
    *,
    expected_attack_seeds: Iterable[int],
    min_clean_seeds: int = 3,
    min_base_success_rate: float = 0.5,
    preservation_weight_floor: float = 0.05,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate repetitions and return per-trajectory and per-group labels.

    The returned annotation map is keyed by normalized trajectory id.  Group
    rows are suitable for probability-level evaluation without incorrectly
    treating repeated trajectories as independent configurations.
    """

    expected_seeds = {int(seed) for seed in expected_attack_seeds}
    if not expected_seeds:
        raise ValueError("At least one expected attack seed is required")
    if min_clean_seeds <= 0:
        raise ValueError("min_clean_seeds must be positive")
    if not 0.0 <= min_base_success_rate <= 1.0:
        raise ValueError("min_base_success_rate must lie in [0, 1]")
    if not 0.0 <= preservation_weight_floor <= 1.0:
        raise ValueError("preservation_weight_floor must lie in [0, 1]")

    attack_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    clean_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_trajectory_ids: set[str] = set()
    for row in metadata:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in seen_trajectory_ids:
            raise ValueError(f"Duplicate trajectory id: {trajectory_id}")
        seen_trajectory_ids.add(trajectory_id)
        source_kind = row.get("source_kind")
        if source_kind == "attack":
            row_id = row.get("row_id")
            if not row_id:
                raise ValueError("Attack metadata row is missing row_id")
            attack_groups[str(row_id)].append(row)
        elif source_kind == "clean":
            clean_groups[(str(row["suite"]), str(row["user_task_id"]))].append(row)
        else:
            raise ValueError(f"Unsupported source_kind: {source_kind!r}")

    if not attack_groups:
        raise ValueError("No attack groups found")
    if not clean_groups:
        raise ValueError("No clean groups found")

    clean_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(clean_groups.items()):
        run_seeds = [row.get("run_seed") for row in rows]
        if any(seed is None for seed in run_seeds):
            raise ValueError(f"Clean group {key} has a missing run seed")
        if len(set(run_seeds)) != len(run_seeds):
            raise ValueError(f"Clean group {key} has duplicate run seeds: {run_seeds}")
        if len(rows) < min_clean_seeds:
            raise ValueError(
                f"Clean group {key} has {len(rows)} seeds; need at least {min_clean_seeds}"
            )
        successes = sum(bool(row.get("utility")) for row in rows)
        posterior = jeffreys_posterior(successes, len(rows))
        clean_rate = successes / len(rows)
        group_id = f"clean::{key[0]}::{key[1]}"
        summary = {
            "multiseed_group_id": group_id,
            "source_kind": "clean",
            "suite": key[0],
            "user_task_id": key[1],
            "task_split": _single_value(rows, "task_split"),
            "run_seeds": sorted(int(seed) for seed in run_seeds),
            "base_task_success_rate": clean_rate,
            **_posterior_fields("utility_probability", posterior),
        }
        clean_summaries[key] = summary
        group_rows.append(summary)
        for row in rows:
            annotations[str(row["trajectory_id"])] = {
                "multiseed_group_id": group_id,
                "multiseed_trials": len(rows),
                "base_task_success_rate": clean_rate,
                "preservation_trainable": False,
                "preservation_weight": 0.0,
                "utility_probability_target": posterior.mean,
                "preservation_probability_target": None,
                "attack_probability_target": None,
                "joint_success_probability_target": None,
                "probability_label_alpha": posterior.alpha,
                "probability_label_beta": posterior.beta,
                "probability_label_variance": posterior.variance,
                "probability_label_confidence": posterior.confidence,
                "attack_probability_confidence": 0.0,
                "joint_success_probability_confidence": 0.0,
                "probability_label_source": (
                    f"jeffreys_clean_multiseed_{len(rows)}"
                ),
            }

    for row_id, rows in sorted(attack_groups.items()):
        run_seeds = [int(row["run_seed"]) for row in rows]
        seed_set = set(run_seeds)
        if len(seed_set) != len(run_seeds):
            raise ValueError(f"Attack group {row_id} has duplicate seeds: {run_seeds}")
        if seed_set != expected_seeds:
            missing = sorted(expected_seeds - seed_set)
            extra = sorted(seed_set - expected_seeds)
            raise ValueError(
                f"Attack group {row_id} seed mismatch: missing={missing}, extra={extra}"
            )
        suite = str(_single_value(rows, "suite"))
        user_task_id = str(_single_value(rows, "user_task_id"))
        clean = clean_summaries.get((suite, user_task_id))
        if clean is None:
            raise ValueError(f"Attack group {row_id} has no clean evidence")

        utility_successes = sum(bool(row.get("utility")) for row in rows)
        attack_successes = sum(bool(row.get("security")) for row in rows)
        joint_successes = sum(
            bool(row.get("utility")) and bool(row.get("security")) for row in rows
        )
        utility = jeffreys_posterior(utility_successes, len(rows))
        attack = jeffreys_posterior(attack_successes, len(rows))
        joint = jeffreys_posterior(joint_successes, len(rows))
        clean_probability = float(clean["utility_probability_target"])
        preservation_probability = min(1.0, utility.mean / clean_probability)
        clean_logit_prior = _safe_logit(clean_probability)
        utility_logit_residual = _safe_logit(utility.mean) - clean_logit_prior
        joint_counts, joint_alpha, joint_probabilities = joint_outcome_posterior(rows)
        base_rate = float(clean["base_task_success_rate"])
        trainable = base_rate >= min_base_success_rate
        preservation_weight = max(preservation_weight_floor, base_rate)
        group_id = f"attack::{row_id}"
        summary = {
            "multiseed_group_id": group_id,
            "source_kind": "attack",
            "row_id": row_id,
            "suite": suite,
            "user_task_id": user_task_id,
            "injection_task_id": _single_value(rows, "injection_task_id"),
            "task_split": _single_value(rows, "task_split"),
            "attack_name": _single_value(rows, "attack_name"),
            "attack_family": _single_value(rows, "attack_family"),
            "attack_role": _single_value(rows, "attack_role"),
            "run_seeds": sorted(run_seeds),
            "base_task_success_rate": base_rate,
            "clean_probability_target": clean_probability,
            "preservation_probability_target": preservation_probability,
            "clean_utility_logit_prior": clean_logit_prior,
            "attack_utility_logit_residual_target": utility_logit_residual,
            "joint_outcome_counts": joint_counts,
            "joint_outcome_dirichlet_alpha": joint_alpha,
            "joint_outcome_probability_target": joint_probabilities,
            "joint_outcome_trials": len(rows),
            "preservation_trainable": trainable,
            "preservation_weight": preservation_weight,
            **_posterior_fields("utility_probability", utility),
            **_posterior_fields("attack_probability", attack),
            **_posterior_fields("joint_success_probability", joint),
        }
        group_rows.append(summary)
        for row in rows:
            annotations[str(row["trajectory_id"])] = {
                "multiseed_group_id": group_id,
                "multiseed_trials": len(rows),
                "base_task_success_rate": base_rate,
                "preservation_trainable": trainable,
                "preservation_weight": preservation_weight,
                "utility_probability_target": utility.mean,
                "preservation_probability_target": preservation_probability,
                "attack_probability_target": attack.mean,
                "joint_success_probability_target": joint.mean,
                "joint_outcome_counts": joint_counts,
                "joint_outcome_dirichlet_alpha": joint_alpha,
                "joint_outcome_probability_target": joint_probabilities,
                "joint_outcome_trials": len(rows),
                "clean_utility_logit_prior": clean_logit_prior,
                "attack_utility_logit_residual_target": utility_logit_residual,
                "probability_label_alpha": utility.alpha,
                "probability_label_beta": utility.beta,
                "probability_label_variance": utility.variance,
                "probability_label_confidence": utility.confidence,
                "attack_probability_confidence": attack.confidence,
                "joint_success_probability_confidence": joint.confidence,
                "probability_label_source": (
                    f"jeffreys_attack_config_multiseed_{len(rows)}"
                    "+clean_multiseed_ratio+joint_dirichlet_residual_v1"
                ),
            }

    if set(annotations) != seen_trajectory_ids:
        missing = sorted(seen_trajectory_ids - set(annotations))
        raise RuntimeError(f"Missing trajectory annotations: {missing[:5]}")

    attack_rows = [row for row in group_rows if row["source_kind"] == "attack"]
    audit = {
        "expected_attack_seeds": sorted(expected_seeds),
        "attack_groups": len(attack_rows),
        "attack_trajectories": sum(
            row["utility_probability_trials"] for row in attack_rows
        ),
        "clean_groups": len(clean_summaries),
        "clean_trajectories": sum(
            row["utility_probability_trials"]
            for row in group_rows
            if row["source_kind"] == "clean"
        ),
        "mixed_utility_groups": sum(
            0 < row["utility_probability_successes"] < row["utility_probability_trials"]
            for row in attack_rows
        ),
        "mixed_attack_groups": sum(
            0 < row["attack_probability_successes"] < row["attack_probability_trials"]
            for row in attack_rows
        ),
        "multi_cell_joint_groups": sum(
            sum(value > 0 for value in row["joint_outcome_counts"].values()) > 1
            for row in attack_rows
        ),
        "preservation_trainable_groups": sum(
            bool(row["preservation_trainable"]) for row in attack_rows
        ),
        "utility_probability_confidence_mean": sum(
            row["utility_probability_confidence"] for row in attack_rows
        )
        / len(attack_rows),
        "attack_probability_confidence_mean": sum(
            row["attack_probability_confidence"] for row in attack_rows
        )
        / len(attack_rows),
    }
    return annotations, group_rows, audit


def posterior_as_dict(posterior: BetaPosterior) -> dict[str, Any]:
    """Public serialization helper used by diagnostics and tests."""

    return asdict(posterior)
