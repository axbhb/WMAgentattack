import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wmagentattack.risk_calibration import (
    MonotonicAffineRiskCalibrator,
    build_grouped_risk_data,
    fit_monotonic_affine_group_calibrator,
    grouped_risk_brier,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frozen_risk_calibration_summary",
    ROOT / "scripts" / "68_summarize_frozen_risk_calibration.py",
)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def _step(trajectory, step_id, group, target, trials=2):
    return SimpleNamespace(
        trajectory_id=trajectory,
        step_id=step_id,
        multiseed_group_id=group,
        multiseed_trials=trials,
        attack_probability_target=target,
    )


def test_group_data_uses_final_steps_and_monotonic_transform_preserves_order():
    steps = [
        _step("a", 0, "g1", 0.25),
        _step("a", 1, "g1", 0.25),
        _step("b", 0, "g1", 0.25),
        _step("c", 0, "g2", 0.75),
        _step("d", 0, "g2", 0.75),
    ]
    scores = np.asarray([0.99, 0.4, 0.5, 0.6, 0.7])
    data = build_grouped_risk_data(steps, scores)
    assert data.group_count == 2
    assert data.trajectory_count == 4
    assert data.score_groups[0].tolist() == [0.4, 0.5]

    calibrator = MonotonicAffineRiskCalibrator(scale=1.7, bias=-0.2)
    transformed = calibrator.transform(scores)
    assert np.array_equal(np.argsort(scores), np.argsort(transformed))
    assert np.allclose(MonotonicAffineRiskCalibrator().transform(scores), scores)


def test_fit_improves_synthetic_group_brier():
    pytest.importorskip("torch")
    steps = []
    scores = []
    for group, target, values in (
        ("low", 0.1, [0.35, 0.4]),
        ("mid", 0.5, [0.48, 0.52]),
        ("high", 0.9, [0.6, 0.65]),
    ):
        for index, score in enumerate(values):
            steps.append(_step(f"{group}-{index}", 0, group, target))
            scores.append(score)
    data = build_grouped_risk_data(steps, scores)
    calibrator, stats = fit_monotonic_affine_group_calibrator(
        data, regularization=0.0
    )
    assert calibrator.scale > 0.0
    assert grouped_risk_brier(data, calibrator) < grouped_risk_brier(data)
    assert stats["group_count"] == 3


def _write_validation(root, seed, candidate, objective):
    if candidate == "identity":
        path = root / f"seed{seed}" / "identity_val_metrics.json"
    else:
        path = root / f"seed{seed}" / candidate / "val_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "validation_objective": objective,
                    "grouped_risk_probability_brier_score": objective,
                    "risk_auc": 0.8,
                    "next_skill_accuracy": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )


def test_selection_includes_identity_and_does_not_read_test(tmp_path):
    seeds = [7, 13, 21]
    for candidate in SUMMARY.CANDIDATES:
        value = 0.2 if candidate == "reg1e-2" else 0.4
        for seed in seeds:
            _write_validation(tmp_path, seed, candidate, value)
            bad_test = tmp_path / f"seed{seed}" / candidate / "test_result.json"
            bad_test.parent.mkdir(parents=True, exist_ok=True)
            bad_test.write_text("not-json", encoding="utf-8")
    selection = SUMMARY.select(tmp_path, seeds)
    assert selection["selected_candidate_by_validation"] == "reg1e-2"
