import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "residual_diagnostic_test",
    ROOT / "scripts" / "93_diagnose_v2_residual_preservation.py",
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAGNOSTIC)


def _row(task, group, attack, utility, std, target_asr, target_bup):
    return {
        "task_key": task,
        "group_id": group,
        "rank_score": attack + utility - std,
        "prediction": attack + utility,
        "attack_prediction": attack,
        "utility_prediction": utility,
        "utility_standard_deviation": std,
        "target": target_asr + target_bup,
        "target_asr": target_asr,
        "target_bup": target_bup,
        "observed": target_asr + target_bup,
        "observed_asr": target_asr,
        "observed_bup": target_bup,
        "clean_probability": 0.5,
        "clean_successes": 1,
        "clean_trials": 3,
        "utility_successes": 2,
        "utility_trials": 5,
    }


def test_penalty_ablation_reuses_predictions_and_marks_posthoc(tmp_path):
    root = tmp_path / "run"
    fold = root / "frozen" / "direct_probability" / "fold0"
    fold.mkdir(parents=True)
    candidate = {
        "target_kind": "direct_probability",
        "alpha": 1.0,
        "utility_weight": 1.0,
        "uncertainty_penalty": 1.0,
    }
    rows = [
        _row("banking|task0", "a", 0.7, 0.8, 0.7, 0.8, 0.8),
        _row("banking|task0", "b", 0.6, 0.7, 0.0, 0.6, 0.7),
    ]
    (fold / "result.json").write_text(
        json.dumps(
            {
                "protocol": {"frozen_candidate": candidate},
                "test_candidate_scores": rows,
            }
        ),
        encoding="utf-8",
    )

    result = DIAGNOSTIC.diagnose(root)
    family = result["families"]["direct_probability"]

    assert "Held-out labels" in result["claim_boundary"]
    assert family["penalty_ablation"]["0.0"]["selected_group_by_task"][
        "banking|task0"
    ] == "a"
    assert family["penalty_ablation"]["1.0"]["selected_group_by_task"][
        "banking|task0"
    ] == "b"
    assert family["selection_switches_vs_formal_penalty"]["0.0"][
        "task_count"
    ] == 1
