import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "144_summarize_custom_panel_v2_architecture_ablation.py"
SPEC = importlib.util.spec_from_file_location("architecture_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _protocol():
    return {
        "protocol_id": "fixture",
        "claim_boundary": "clean probe only",
        "uncertainty": {
            "paired_task_bootstrap_draws": 100,
            "bootstrap_seed": 11,
        },
        "frozen_gates": {
            "minimum_action_nll_gain": 0.02,
            "minimum_evidence_nll_gain": 0.02,
            "maximum_cross_head_nll_regression": 0.02,
            "minimum_positive_training_seeds": 2,
            "minimum_confirmation_positive_tasks": 2,
        },
    }


def _run_metrics():
    values = {
        "semantic_markov": (1.00, 1.00, 0.20),
        "observable_execution": (0.90, 0.90, 0.18),
        "observable_execution_ledger_v2": (0.84, 0.75, 0.14),
    }
    runs = []
    for variant, (action, evidence, brier) in values.items():
        for seed in (7, 17, 29):
            split = {
                name: {
                    "task_macro_action_nll": action,
                    "task_macro_action_accuracy": 1.0 - action / 2,
                    "task_macro_stop_accuracy": 0.8,
                    "task_macro_argument_key_f1": 0.5,
                    "task_macro_error_recovery_action_nll": action,
                    "task_macro_evidence_status_nll": evidence,
                    "task_macro_evidence_status_accuracy": 1.0 - evidence / 2,
                    "task_macro_supported_brier": brier,
                }
                for name in ("training", "calibration", "confirmation")
            }
            runs.append(
                {"variant": variant, "training_seed": seed, "metrics": split}
            )
    return {
        "variants": list(values),
        "training_seeds": [7, 17, 29],
        "runs": runs,
    }


def _predictions():
    values = {
        "semantic_markov": (1.00, 1.00, 0.20),
        "observable_execution": (0.90, 0.90, 0.18),
        "observable_execution_ledger_v2": (0.84, 0.75, 0.14),
    }
    rows = []
    for variant, (action, evidence, brier) in values.items():
        for seed in (7, 17, 29):
            for split in ("calibration", "confirmation"):
                for task in ("task-a", "task-b"):
                    rows.append(
                        {
                            "variant": variant,
                            "training_seed": seed,
                            "split": split,
                            "task_id": task,
                            "prediction_type": "dynamics",
                            "action_nll": action,
                            "action_correct": 1.0,
                        }
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "training_seed": seed,
                            "split": split,
                            "task_id": task,
                            "prediction_type": "evidence",
                            "status_nll": evidence,
                            "supported_brier": brier,
                        }
                    )
    return rows


def test_positive_nested_representation_signals_pass_independent_gates():
    report = MODULE.summarize(_protocol(), _run_metrics(), _predictions())
    assert report["decision"] == "CUSTOM_PANEL_V2_ARCHITECTURE_INCREMENT_PROVISIONAL_GO"
    assert all(row["passed"] for row in report["gates"].values())
    assert report["accepted_heads"] == {
        "victim_dynamics": "observable_execution_ledger_v2",
        "evidence_progress": "observable_execution_ledger_v2",
    }
    assert report["permissions"]["attack_data"] is False
    assert report["permissions"]["dreamer_training"] is False
