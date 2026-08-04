import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task_macro_dynamics_summary",
    ROOT / "scripts" / "118_summarize_task_macro_dynamics_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


TASKS = [f"domain|task_{index}" for index in range(4)]


def _teacher(nll):
    per_task = {
        task: {
            "event_count": 10,
            "next_skill_nll": nll,
            "next_skill_accuracy": 0.6,
            "static_joint_count_nll": 1.2,
            "dynamic_joint_count_nll": 1.0,
            "static_joint_log_score": 0.8,
            "dynamic_joint_log_score": 0.7,
            "dynamic_minus_static_joint_nll": -0.2,
        }
        for task in TASKS
    }
    return {
        "task_count": 4,
        "task_macro": {
            "next_skill_nll": nll,
            "next_skill_accuracy": 0.6,
            "static_joint_count_nll": 1.2,
            "dynamic_joint_count_nll": 1.0,
            "static_joint_log_score": 0.8,
            "dynamic_joint_log_score": 0.7,
            "dynamic_minus_static_joint_nll": -0.2,
        },
        "micro": {
            "event_count": 40,
            "next_skill_nll": nll + 0.05,
            "next_skill_accuracy": 0.59,
        },
        "per_task": per_task,
    }


def _free(edit):
    per_task = {
        task: {
            "trajectory_count": 10,
            "exact_sequence_accuracy": 0.5,
            "normalized_edit_distance": edit,
        }
        for task in TASKS
    }
    return {
        "micro": {
            "trajectory_count": 40,
            "exact_sequence_accuracy": 0.5,
            "normalized_edit_distance": edit,
        },
        "task_count": 4,
        "task_macro": {
            "exact_sequence_accuracy": 0.5,
            "normalized_edit_distance": edit,
        },
        "per_task": per_task,
    }


def _baseline():
    return {
        "teacher": _teacher(1.2),
        "free": _free(0.5),
    }


def _control(nll):
    return {
        "task_macro_joint_count_nll": nll,
        "task_macro_joint_probability_log_score": nll / 2,
        "per_task": {},
    }


def _payload(variant, seed):
    nll = {
        "length_semantic": 1.1,
        "event_no_attack_semantics": 1.0,
        "semantic_event": 0.9,
    }[variant]
    prefix = None
    if variant == "semantic_event":
        prefix = {
            split: {
                "static_semantic": _control(1.10),
                "observed": _control(1.00),
                "shuffled": _control(1.05),
                "length_only": _control(1.20),
                "random_length_matched": _control(1.20),
                "markov_length_matched": _control(1.15),
                "semantic_markov_length_matched": _control(1.14),
            }
            for split in ("validation", "test")
        }
    return {
        "variant": variant,
        "clean_eligibility_gate": False,
        "training": {"seed": seed},
        "data": {
            "task_group_overlap": {
                "train_validation": [],
                "train_test": [],
                "validation_test": [],
            },
            "dataset_audit": {
                "train": {"task_count": 12},
                "validation": {"task_count": 4},
                "test": {"task_count": 4},
            },
        },
        "metrics": {
            "validation": {"teacher": _teacher(nll), "free": _free(0.3)},
            "test": {"teacher": _teacher(nll), "free": _free(0.3)},
        },
        "baselines": {
            split: {
                "candidate_uniform": _baseline(),
                "candidate_hierarchical_markov": _baseline(),
                "candidate_factorized_semantic_markov": _baseline(),
            }
            for split in ("validation", "test")
        },
        "prefix_value_controls": prefix,
    }


def test_summary_applies_task_macro_component_and_prefix_gates(tmp_path, monkeypatch):
    protocol = json.loads(
        (ROOT / "configs" / "0723_task_macro_dynamics_ablation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    audit = {
        "ontology_fingerprint": protocol["frozen_representation"][
            "event_ontology_fingerprint"
        ],
        "splits": {
            split: {"forbidden_outcome_fields_seen": []}
            for split in ("train", "validation", "test")
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    for variant in MODULE.VARIANTS:
        for seed in (7, 17, 29):
            directory = tmp_path / "model" / variant / f"seed{seed}"
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(
                json.dumps(_payload(variant, seed)), encoding="utf-8"
            )
            if variant == "semantic_event":
                predictions = []
                for index in range(4):
                    joint_success = 0.10 + 0.05 * index
                    predictions.append(
                        {
                            "trajectory_id": f"trajectory-{index}",
                            "generated_skill_path": ["record_read", "finish"],
                            "free_joint_probability": {
                                "attack0_utility0": 0.20,
                                "attack0_utility1": 0.60 - joint_success,
                                "attack1_utility0": 0.20,
                                "attack1_utility1": joint_success,
                            },
                        }
                    )
                for split in ("validation", "test"):
                    (directory / f"{split}_free_predictions.jsonl").write_text(
                        "\n".join(json.dumps(row) for row in predictions) + "\n",
                        encoding="utf-8",
                    )

    output = tmp_path / "summary.json"
    markdown = tmp_path / "summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--root",
            str(tmp_path),
            "--protocol",
            str(protocol_path),
            "--ontology-audit",
            str(audit_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ],
    )
    MODULE.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision"] == "TASK_MACRO_MECHANISM_SIGNAL_CLEAN_GATE_BLOCKED"
    assert result["gates"]["task_macro_predictive_and_free_vs_markov"] is True
    assert result["gates"]["prefix_order_beyond_shuffled_multiset"] is True
    assert result["gates"]["clean_eligibility_gate"] is False
    assert markdown.exists()
