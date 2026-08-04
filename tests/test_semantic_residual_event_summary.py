import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_residual_summary",
    ROOT / "scripts" / "114_summarize_semantic_residual_round.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _prediction(trajectory_id, generated):
    return {
        "trajectory_id": trajectory_id,
        "generated_skill_path": generated,
        "free_joint_probability": {
            "attack0_utility0": 0.1,
            "attack0_utility1": 0.2,
            "attack1_utility0": 0.3,
            "attack1_utility1": 0.4,
        },
    }


def test_summary_applies_direct_markov_free_running_and_clean_gates(tmp_path, monkeypatch):
    for seed, accuracy in ((7, 0.61), (17, 0.62), (29, 0.60)):
        directory = tmp_path / "model" / f"seed{seed}"
        directory.mkdir(parents=True)
        teacher_val = {
            "next_skill_nll": 0.90,
            "next_skill_accuracy": accuracy,
            "static_joint_count_nll": 1.0,
            "dynamic_joint_count_nll": 0.90,
            "dynamic_minus_static_joint_nll": -0.10,
        }
        teacher_test = {
            "next_skill_nll": 0.95,
            "next_skill_accuracy": 0.59,
            "static_joint_count_nll": 1.0,
            "dynamic_joint_count_nll": 0.92,
            "dynamic_minus_static_joint_nll": -0.08,
        }
        payload = {
            "data": {
                "vocabulary_source": "training candidate_skills only",
                "task_group_overlap": {
                    "train_validation": [],
                    "train_test": [],
                    "validation_test": [],
                },
                "dataset_audit": {
                    split: {"selected_skill_oov_events": 0}
                    for split in ("train", "validation", "test")
                },
            },
            "metrics": {
                "validation": {
                    "teacher": teacher_val,
                    "free": {
                        "normalized_edit_distance": 0.30,
                        "exact_sequence_accuracy": 0.20,
                        "free_dynamic_joint_count_nll": 0.90,
                        "conservative_truncation_fraction": 0.10,
                    },
                },
                "test": {
                    "teacher": teacher_test,
                    "free": {
                        "normalized_edit_distance": 0.35,
                        "exact_sequence_accuracy": 0.18,
                        "free_dynamic_joint_count_nll": 0.92,
                        "conservative_truncation_fraction": 0.12,
                    },
                },
            },
            "baselines": {
                "candidate_hierarchical_markov": {
                    "metrics": {
                        "validation": {
                            "teacher": {
                                "next_skill_nll": 1.0,
                                "next_skill_accuracy": 0.50,
                            },
                            "free": {
                                "normalized_edit_distance": 0.40,
                                "exact_sequence_accuracy": 0.10,
                            },
                        },
                        "test": {
                            "teacher": {
                                "next_skill_nll": 1.0,
                                "next_skill_accuracy": 0.49,
                            },
                            "free": {
                                "normalized_edit_distance": 0.40,
                                "exact_sequence_accuracy": 0.09,
                            },
                        },
                    }
                }
            },
        }
        (directory / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        for split in ("validation", "test"):
            (directory / f"{split}_free_predictions.jsonl").write_text(
                json.dumps(_prediction("trajectory-1", ["record_read", "finish"]))
                + "\n",
                encoding="utf-8",
            )

    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary", "--root", str(tmp_path), "--output", str(output)],
    )
    MODULE.main()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["architecture_signal"] is True
    assert summary["gates"]["zero_selected_skill_oov"] is True
    assert summary["gates"]["free_validation_edit_beats_candidate_markov"] is True
    assert summary["gates"]["clean_eligibility_gate"] is False
    assert summary["decision"] == "ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED"
