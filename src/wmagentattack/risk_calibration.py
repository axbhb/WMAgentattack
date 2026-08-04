"""Frozen monotonic calibration for multi-seed AgentDojo risk predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class GroupedRiskData:
    group_ids: tuple[str, ...]
    score_groups: tuple[np.ndarray, ...]
    targets: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def trajectory_count(self) -> int:
        return sum(len(scores) for scores in self.score_groups)


@dataclass(frozen=True)
class MonotonicAffineRiskCalibrator:
    scale: float = 1.0
    bias: float = 0.0
    regularization: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and strictly positive")
        if not np.isfinite(self.bias):
            raise ValueError("bias must be finite")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")

    def transform(self, probabilities: Iterable[float]) -> np.ndarray:
        return transform_risk_probabilities(
            probabilities, scale=self.scale, bias=self.bias
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        return cls(
            scale=float(payload["scale"]),
            bias=float(payload["bias"]),
            regularization=float(payload.get("regularization", 0.0)),
        )


def transform_risk_probabilities(
    probabilities: Iterable[float],
    *,
    scale: float,
    bias: float,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Apply a strictly increasing affine transform in logit space."""

    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and strictly positive")
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    logits = np.log(clipped) - np.log1p(-clipped)
    transformed_logits = np.clip(scale * logits + bias, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-transformed_logits))


def build_grouped_risk_data(
    steps: list[Any], risk_scores: Iterable[float]
) -> GroupedRiskData:
    """Use one final prediction per trajectory and preserve complete seed groups."""

    scores = np.asarray(list(risk_scores), dtype=np.float64)
    if len(scores) != len(steps):
        raise ValueError("risk_scores must align one-to-one with steps")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("risk scores must be finite values in [0, 1]")

    final_by_trajectory: dict[str, int] = {}
    for index, step in enumerate(steps):
        previous = final_by_trajectory.get(step.trajectory_id)
        if previous is None or steps[previous].step_id < step.step_id:
            final_by_trajectory[step.trajectory_id] = index

    grouped: dict[str, list[int]] = {}
    for index in final_by_trajectory.values():
        step = steps[index]
        if (
            step.multiseed_group_id is None
            or step.attack_probability_target is None
        ):
            continue
        grouped.setdefault(str(step.multiseed_group_id), []).append(index)
    if not grouped:
        raise ValueError("No repeated attack groups with probability targets found")

    group_ids = []
    score_groups = []
    targets = []
    for group_id in sorted(grouped):
        indices = grouped[group_id]
        target_values = np.asarray(
            [steps[index].attack_probability_target for index in indices],
            dtype=np.float64,
        )
        if not np.allclose(target_values, target_values[0], rtol=0.0, atol=1e-7):
            raise ValueError(f"Inconsistent attack target in group {group_id}")
        expected_sizes = {
            int(steps[index].multiseed_trials)
            for index in indices
            if steps[index].multiseed_trials is not None
        }
        if len(expected_sizes) > 1:
            raise ValueError(f"Inconsistent expected size in group {group_id}")
        if expected_sizes and len(indices) != next(iter(expected_sizes)):
            raise ValueError(
                f"Incomplete group {group_id}: found {len(indices)}, "
                f"expected {next(iter(expected_sizes))}"
            )
        group_ids.append(group_id)
        score_groups.append(scores[indices].copy())
        targets.append(float(target_values[0]))
    return GroupedRiskData(
        group_ids=tuple(group_ids),
        score_groups=tuple(score_groups),
        targets=np.asarray(targets, dtype=np.float64),
    )


def grouped_risk_brier(
    data: GroupedRiskData,
    calibrator: MonotonicAffineRiskCalibrator | None = None,
) -> float:
    calibrator = calibrator or MonotonicAffineRiskCalibrator()
    predictions = np.asarray(
        [calibrator.transform(scores).mean() for scores in data.score_groups],
        dtype=np.float64,
    )
    return float(np.mean((predictions - data.targets) ** 2))


def fit_monotonic_affine_group_calibrator(
    data: GroupedRiskData,
    *,
    regularization: float,
    initial_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    max_iter: int = 300,
) -> tuple[MonotonicAffineRiskCalibrator, dict[str, Any]]:
    """Fit two frozen-head calibration parameters with group-level Brier loss."""

    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if not initial_scales or any(scale <= 0.0 for scale in initial_scales):
        raise ValueError("initial_scales must contain positive values")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - deployment environment
        raise RuntimeError("Calibration fitting requires torch") from exc

    dtype = torch.float64
    epsilon = 1e-7
    flattened = np.concatenate(data.score_groups)
    clipped = np.clip(flattened, epsilon, 1.0 - epsilon)
    logits = torch.as_tensor(
        np.log(clipped) - np.log1p(-clipped), dtype=dtype
    )
    group_slices = []
    start = 0
    for values in data.score_groups:
        group_slices.append(slice(start, start + len(values)))
        start += len(values)
    targets = torch.as_tensor(data.targets, dtype=dtype)
    minimum_scale = 1e-8
    candidates: list[dict[str, Any]] = []

    for initial_scale in initial_scales:
        initial_raw = np.log(np.expm1(max(initial_scale - minimum_scale, 1e-8)))
        raw_scale = torch.tensor(initial_raw, dtype=dtype, requires_grad=True)
        bias = torch.tensor(0.0, dtype=dtype, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [raw_scale, bias],
            lr=0.5,
            max_iter=max_iter,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def objective():
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad()
            scale = F.softplus(raw_scale) + minimum_scale
            probabilities = torch.sigmoid(scale * logits + bias)
            group_predictions = torch.stack(
                [probabilities[group_slice].mean() for group_slice in group_slices]
            )
            brier = (group_predictions - targets).square().mean()
            identity_penalty = (scale - 1.0).square() + bias.square()
            loss = brier + regularization * identity_penalty
            loss.backward()
            return loss

        optimizer.step(objective)
        with torch.no_grad():
            scale_value = float((F.softplus(raw_scale) + minimum_scale).item())
            bias_value = float(bias.item())
        calibrator = MonotonicAffineRiskCalibrator(
            scale=scale_value,
            bias=bias_value,
            regularization=regularization,
        )
        brier = grouped_risk_brier(data, calibrator)
        objective_value = brier + regularization * (
            (scale_value - 1.0) ** 2 + bias_value**2
        )
        candidates.append(
            {
                "calibrator": calibrator,
                "objective": objective_value,
                "brier": brier,
                "initial_scale": initial_scale,
                "closure_calls": closure_calls,
            }
        )

    best = min(candidates, key=lambda row: row["objective"])
    calibrator = best["calibrator"]
    stats = {
        "group_count": data.group_count,
        "trajectory_count": data.trajectory_count,
        "regularization": regularization,
        "train_group_brier_before": grouped_risk_brier(data),
        "train_group_brier_after": best["brier"],
        "regularized_objective": best["objective"],
        "selected_initial_scale": best["initial_scale"],
        "closure_calls": best["closure_calls"],
        "candidate_objectives": [
            {
                key: value
                for key, value in row.items()
                if key != "calibrator"
            }
            for row in candidates
        ],
    }
    return calibrator, stats
