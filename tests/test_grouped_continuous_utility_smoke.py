import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "81_compare_v2_group_utility_smoke.py"
SPEC = importlib.util.spec_from_file_location("group_utility_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(utility_brier: float) -> dict:
    return {
        "metrics": {
            "validation_objective": utility_brier + 0.2,
            "grouped_risk_probability_brier_score": 0.04,
            "grouped_utility_probability_brier_score": utility_brier,
            "grouped_preservation_probability_brier_score": 0.03,
            "binary_utility_auc": 0.7,
            "risk_auc": 0.8,
        }
    }


def _downstream(score: float) -> dict:
    return {
        "test": {
            budget: {
                "calibrated": {
                    "frozen_validation_recipe": "risk_plus_utility",
                    "ensemble": {
                        "ASR": score - 0.5,
                        "BUP": 0.5,
                        "ASR_plus_BUP": score,
                    },
                }
            }
            for budget in ("1", "2", "4")
        }
    }


def _training(pair_count: float) -> dict:
    epoch = {
        "world": 1.0,
        "group_utility_calibration": 0.1,
        "group_utility_calibration_count": 2.0,
        "group_utility_ranking": 0.6,
        "group_utility_ranking_pair_count": pair_count,
    }
    return {"best_epoch": 1, "training_history": [epoch, epoch]}


def test_smoke_summary_selects_by_validation_and_checks_loss_activation(tmp_path):
    _write(tmp_path / "baseline" / "fold0" / "val_metrics.json", _metrics(0.10))
    _write(tmp_path / "baseline" / "fold0" / "test_metrics.json", _metrics(0.11))
    _write(tmp_path / "baseline_downstream.json", _downstream(0.70))
    settings = {
        "group_utility_detached": (0.08, 0.09, 0.75, 4.0),
        "group_utility_end_to_end": (0.12, 0.10, 0.68, 4.0),
    }
    for variant, (val_brier, test_brier, score, pair_count) in settings.items():
        run = tmp_path / "models" / "fold0" / variant / "seed7"
        _write(run / "training_stdout.json", _training(pair_count))
        _write(run / "val_metrics.json", _metrics(val_brier))
        _write(run / "test_metrics.json", _metrics(test_brier))
        _write(tmp_path / f"{variant}_downstream.json", _downstream(score))

    result = MODULE.summarize(tmp_path, fold=0, seed=7)

    assert result["smoke_gate"]["activation_valid"]
    assert (
        result["smoke_gate"]["validation_selected_variant"]
        == "group_utility_detached"
    )
    assert result["smoke_gate"]["proceed_to_formal_5fold"]
    assert result["variants"]["group_utility_detached"]["downstream"]["1"][
        "ASR_plus_BUP_delta_vs_baseline"
    ] == pytest.approx(0.05)

    single = MODULE.summarize(
        tmp_path,
        fold=0,
        seed=7,
        variants=("group_utility_detached",),
    )
    assert single["protocol"]["variants"] == ["group_utility_detached"]
    assert set(single["variants"]) == {"group_utility_detached"}
