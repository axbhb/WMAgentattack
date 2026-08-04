import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "82_summarize_v2_group_utility_seedrep.py"
)
SPEC = importlib.util.spec_from_file_location("group_utility_seedrep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(utility_brier: float, risk_auc: float) -> dict:
    return {
        "metrics": {
            "validation_objective": utility_brier + 0.2,
            "grouped_risk_probability_brier_score": 0.04,
            "grouped_utility_probability_brier_score": utility_brier,
            "grouped_preservation_probability_brier_score": 0.05,
            "binary_utility_auc": 0.75,
            "risk_auc": risk_auc,
        }
    }


def _training() -> dict:
    return {
        "best_epoch": 1,
        "training_history": [
            {
                "world": 1.0,
                "group_utility_calibration": 0.1,
                "group_utility_ranking": 0.5,
                "group_utility_calibration_count": 2.0,
                "group_utility_ranking_pair_count": 1.0,
            }
        ],
    }


def _downstream(score: float, bup: float) -> dict:
    return {
        "test": {
            budget: {
                "calibrated": {
                    "frozen_validation_recipe": "risk_plus_utility",
                    "ensemble": {
                        "ASR": score - bup,
                        "BUP": bup,
                        "ASR_plus_BUP": score,
                    },
                    "per_seed": {
                        str(seed): {
                            "ASR": score - bup,
                            "BUP": bup,
                            "ASR_plus_BUP": score,
                        }
                        for seed in (7, 13, 21)
                    },
                }
            }
            for budget in ("1", "2", "4")
        }
    }


def test_seedrep_gate_requires_replicated_utility_and_joint_improvement(tmp_path):
    variant_briers = {7: 0.08, 13: 0.09, 21: 0.11}
    for seed in (7, 13, 21):
        baseline = tmp_path / "baseline" / "fold1" / f"seed{seed}"
        variant = (
            tmp_path
            / "models"
            / "fold1"
            / "group_utility_head_only"
            / f"seed{seed}"
        )
        for split in ("val", "test"):
            _write(baseline / f"{split}_metrics.json", _metrics(0.10, 0.90))
            _write(
                variant / f"{split}_metrics.json",
                _metrics(variant_briers[seed], 0.87),
            )
        _write(variant / "training_stdout.json", _training())
    _write(tmp_path / "baseline_downstream.json", _downstream(0.45, 0.30))
    _write(tmp_path / "variant_downstream.json", _downstream(0.55, 0.32))

    result = MODULE.summarize(
        tmp_path,
        fold=1,
        seeds=(7, 13, 21),
        variant="group_utility_head_only",
    )

    assert result["replication_gate"]["proceed_to_broader_folds"]
    assert result["replication_gate"]["test_utility_brier_improved_seed_count"] == 2
    assert result["downstream_contrast"]["1"][
        "ensemble_variant_minus_baseline"
    ]["ASR_plus_BUP"] == pytest.approx(0.10)
