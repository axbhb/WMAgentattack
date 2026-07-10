"""Leakage-resistant continuous utility labels for AgentDojo trajectories.

The original AgentDojo utility is a single Bernoulli observation.  This module
turns it into a posterior predictive probability without pretending that the
three clean seeds are attacked repetitions.  Clean solvability supplies the
task prior, while attack-location evidence is estimated from the training split
only.  Each training row uses leave-one-out context statistics before its own
outcome is added, and validation/test rows never contribute to context priors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


TaskKey = tuple[str, str]
ContextKey = tuple[str, str]


@dataclass(frozen=True)
class BetaEvidence:
    successes: float
    attempts: float

    @property
    def failures(self) -> float:
        return self.attempts - self.successes


@dataclass(frozen=True)
class ProbabilityLabel:
    utility_probability: float
    preservation_probability: float | None
    alpha: float
    beta: float
    variance: float
    confidence: float
    source: str


def load_clean_evidence(payload: dict[str, Any]) -> dict[TaskKey, BetaEvidence]:
    evidence: dict[TaskKey, BetaEvidence] = {}
    for row in payload.get("tasks", []):
        key = (str(row["suite"]), str(row["user_task_id"]))
        attempts = float(row.get("attempts", 0))
        successes = float(row.get("successes", 0))
        if attempts <= 0 or not 0 <= successes <= attempts:
            continue
        evidence[key] = BetaEvidence(successes=successes, attempts=attempts)
    return evidence


def trajectory_context(row: dict[str, Any]) -> ContextKey | None:
    steps = row.get("steps") or []
    if not steps or steps[0].get("attack_action") is None:
        return None
    location = str(steps[0].get("attack_location") or "UNKNOWN")
    return str(row["domain"]), location


def build_training_context_evidence(
    trajectories: Iterable[dict[str, Any]],
) -> dict[ContextKey, BetaEvidence]:
    grouped: dict[ContextKey, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in trajectories:
        context = trajectory_context(row)
        if context is None:
            continue
        grouped[context][0] += float(bool(row["final_task_success"]))
        grouped[context][1] += 1.0
    return {
        key: BetaEvidence(successes=values[0], attempts=values[1])
        for key, values in grouped.items()
    }


def build_training_global_evidence(
    trajectories: Iterable[dict[str, Any]],
) -> BetaEvidence:
    successes = 0.0
    attempts = 0.0
    for row in trajectories:
        if trajectory_context(row) is None:
            continue
        successes += float(bool(row["final_task_success"]))
        attempts += 1.0
    if attempts <= 0:
        raise ValueError("Training split has no attacked trajectories")
    return BetaEvidence(successes=successes, attempts=attempts)


def beta_variance(alpha: float, beta: float) -> float:
    total = alpha + beta
    return alpha * beta / (total * total * (total + 1.0))


def _confidence_from_variance(variance: float) -> float:
    # A Beta(1, 1) prior has variance 1/12.  Map that uncertainty to zero and
    # approach one as the posterior tightens.
    return max(0.0, min(1.0, 1.0 - 12.0 * variance))


def _capped_context_counts(
    evidence: BetaEvidence | None,
    *,
    max_strength: float,
) -> tuple[float, float, float]:
    if evidence is None or evidence.attempts <= 0 or max_strength <= 0:
        return 0.0, 0.0, 0.0
    strength = min(max_strength, evidence.attempts)
    rate = evidence.successes / evidence.attempts
    return rate * strength, (1.0 - rate) * strength, strength


def estimate_probability_label(
    *,
    clean: BetaEvidence,
    global_attack: BetaEvidence | None,
    attacked: bool,
    observed_success: bool,
    context: BetaEvidence | None,
    split: str,
    global_prior_strength: float = 1.0,
    clean_prior_strength: float = 2.0,
    context_max_strength: float = 4.0,
    observation_strength: float = 1.0,
    min_clean_probability: float = 1e-6,
) -> ProbabilityLabel:
    """Estimate a continuous utility target from clean and attack evidence.

    For attacked rows, ``context`` must already be leave-one-out for a training
    row or training-only for a held-out row.  The row's own Bernoulli outcome is
    then added exactly once with ``observation_strength``.
    """

    if global_prior_strength <= 0 or clean_prior_strength < 0:
        raise ValueError("Prior strengths must be valid")
    if observation_strength < 0 or context_max_strength < 0:
        raise ValueError("Evidence strengths must be non-negative")

    clean_alpha = 0.5 + clean.successes
    clean_beta = 0.5 + clean.failures
    clean_probability = clean_alpha / (clean_alpha + clean_beta)

    if not attacked:
        alpha = clean_alpha
        beta = clean_beta
        source_parts = ["jeffreys_clean"]
    else:
        if global_attack is None or global_attack.attempts <= 0:
            raise ValueError("Attacked labels require training-global attack evidence")
        global_probability = global_attack.successes / global_attack.attempts
        alpha = global_prior_strength * global_probability
        beta = global_prior_strength * (1.0 - global_probability)
        alpha += clean_prior_strength * clean_probability
        beta += clean_prior_strength * (1.0 - clean_probability)
        source_parts = ["train_global", "weak_clean"]
        context_successes, context_failures, context_strength = _capped_context_counts(
            context,
            max_strength=context_max_strength,
        )
        alpha += context_successes
        beta += context_failures
        if context_strength > 0:
            source_parts.append(f"train_attack_location_{split}")
        alpha += observation_strength * float(observed_success)
        beta += observation_strength * float(not observed_success)
        source_parts.append("observed_attack_outcome")

    utility_probability = alpha / (alpha + beta)
    preservation_probability = None
    if attacked:
        preservation_probability = min(
            1.0,
            utility_probability / max(clean_probability, min_clean_probability),
        )
    variance = beta_variance(alpha, beta)
    return ProbabilityLabel(
        utility_probability=utility_probability,
        preservation_probability=preservation_probability,
        alpha=alpha,
        beta=beta,
        variance=variance,
        confidence=_confidence_from_variance(variance),
        source="+".join(source_parts),
    )


def context_for_row(
    row: dict[str, Any],
    *,
    training_context: dict[ContextKey, BetaEvidence],
    split: str,
) -> BetaEvidence | None:
    key = trajectory_context(row)
    if key is None:
        return None
    evidence = training_context.get(key)
    if evidence is None:
        return None
    if split != "train":
        return evidence

    # Leave the current training outcome out of the context prior.  Its outcome
    # is added once by ``estimate_probability_label``.
    remaining_attempts = evidence.attempts - 1.0
    remaining_successes = evidence.successes - float(bool(row["final_task_success"]))
    if remaining_attempts <= 0:
        return None
    return BetaEvidence(successes=remaining_successes, attempts=remaining_attempts)


def global_for_row(
    row: dict[str, Any],
    *,
    training_global: BetaEvidence,
    split: str,
) -> BetaEvidence:
    if split != "train" or trajectory_context(row) is None:
        return training_global
    attempts = training_global.attempts - 1.0
    successes = training_global.successes - float(bool(row["final_task_success"]))
    if attempts <= 0:
        return training_global
    return BetaEvidence(successes=successes, attempts=attempts)
