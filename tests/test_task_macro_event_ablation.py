import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task_macro_event_ablation",
    ROOT / "scripts" / "117_train_task_macro_event_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_unbiased_task_macro_gives_each_task_equal_weight():
    values = torch.tensor([[2.0, 0.0, 0.0], [3.0, 6.0, 9.0]])
    weights = torch.tensor([[0.5, 0.0, 0.0], [1.0 / 6.0] * 3])
    batch = {
        "trajectory_task_weights": torch.tensor([0.5, 0.5]),
        "sampling_population_size": 2,
    }
    objective = MODULE._unbiased_task_macro(values, weights, batch)
    assert float(objective) == pytest.approx((2.0 + 6.0) / 2.0)


def test_variant_inputs_isolate_length_and_attack_semantics():
    batch = {
        "skill_ids": torch.tensor([[2, 4, 5, 0], [2, 6, 0, 0]]),
        "semantic_ids": torch.tensor([[3, 4, 5], [6, 7, 8]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
    }
    unknown = torch.tensor([0, 1, 2])

    length_skills, length_semantics = MODULE._variant_inputs(
        batch, "length_semantic", bos_id=2, semantic_unknown_ids=unknown
    )
    assert length_skills.tolist() == [[2, 2, 2, 0], [2, 2, 0, 0]]
    assert torch.equal(length_semantics, batch["semantic_ids"])

    event_skills, event_semantics = MODULE._variant_inputs(
        batch,
        "event_no_attack_semantics",
        bos_id=2,
        semantic_unknown_ids=unknown,
    )
    assert torch.equal(event_skills, batch["skill_ids"])
    assert event_semantics.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_prefix_controls_preserve_length_and_declared_information():
    batch = {
        "trajectory_ids": ["trajectory-a", "trajectory-b"],
        "skill_ids": torch.tensor([[2, 4, 5, 6, 0], [2, 7, 8, 0, 0]]),
        "attention_mask": torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, True, False, False],
            ]
        ),
        "initial_candidate_mask": torch.tensor(
            [
                [False, False, False, True, True, True, True, False, False],
                [False, False, False, False, False, False, False, True, True],
            ]
        ),
    }
    kwargs = {
        "bos_id": 2,
        "trajectory_rows": {},
        "markov_prefixes": {
            "trajectory-a": [3, 4, 5],
            "trajectory-b": [7, 7],
        },
        "semantic_markov_prefixes": {
            "trajectory-a": [6, 5, 4],
            "trajectory-b": [8, 8],
        },
    }

    shuffled = MODULE._controlled_prefix(batch, "shuffled", **kwargs)
    assert sorted(shuffled[0, 1:4].tolist()) == [4, 5, 6]
    assert sorted(shuffled[1, 1:3].tolist()) == [7, 8]
    assert shuffled[:, 0].tolist() == [2, 2]

    length_only = MODULE._controlled_prefix(batch, "length_only", **kwargs)
    assert length_only.tolist() == [[2, 2, 2, 2, 0], [2, 2, 2, 0, 0]]

    random_prefix = MODULE._controlled_prefix(
        batch, "random_length_matched", **kwargs
    )
    assert set(random_prefix[0, 1:4].tolist()) <= {3, 4, 5, 6}
    assert set(random_prefix[1, 1:3].tolist()) <= {7, 8}

    markov = MODULE._controlled_prefix(batch, "markov_length_matched", **kwargs)
    assert markov.tolist() == [[2, 3, 4, 5, 0], [2, 7, 7, 0, 0]]
    semantic_markov = MODULE._controlled_prefix(
        batch, "semantic_markov_length_matched", **kwargs
    )
    assert semantic_markov.tolist() == [[2, 6, 5, 4, 0], [2, 8, 8, 0, 0]]


def test_dirichlet_multinomial_rows_ignore_zero_support_rows():
    concentration = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    counts = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
    nll, valid = MODULE._dm_nll_rows(concentration, counts)
    assert torch.isfinite(nll).all()
    assert valid.tolist() == [False, True]


def _trajectory_payload(split, task_index):
    trajectory_id = f"{split}-trajectory-{task_index}"
    path = ["record_read", "finish"] if task_index == 0 else ["finish"]
    steps = []
    for step_id, selected in enumerate(path):
        steps.append(
            {
                "trajectory_id": trajectory_id,
                "dataset": split,
                "domain": "workspace",
                "task_id": f"{split}_task_{task_index}",
                "victim_model": "synthetic",
                "agent_architecture": "function-tags",
                "defense": "none",
                "step_id": step_id,
                "trusted_instruction": "read one record",
                "user_goal": "read one record",
                "agent_history": "",
                "current_observation": "synthetic state",
                "untrusted_content": "synthetic sandbox payload",
                "candidate_tools": ["record_read"],
                "candidate_skills": ["finish", "record_read"],
                "candidate_skill_descriptions": {
                    "finish": "finish the task",
                    "record_read": "read a record",
                },
                "previous_skills": path[:step_id],
                "selected_tool": None if selected == "finish" else "record_read",
                "selected_skill": selected,
                "skill_arguments": {} if selected == "finish" else {"id": "one"},
                "skill_output": "synthetic output",
                "attack_action": "synthetic payload",
                "attack_location": "tool output",
                "target_skill": "record_read",
                "task_success": True,
                "attack_success": False,
                "target_skill_success": False,
                "policy_violation": False,
                "risk_level": 0,
                "base_task_success_rate": 0.75,
                "joint_outcome_counts": (
                    {
                        "attack0_utility0": 0,
                        "attack0_utility1": 3,
                        "attack1_utility0": 0,
                        "attack1_utility1": 0,
                    }
                    if step_id == 0
                    else None
                ),
                "joint_outcome_trials": 3 if step_id == 0 else None,
                "multiseed_trials": 3 if step_id == 0 else None,
            }
        )
    return {
        "trajectory_id": trajectory_id,
        "dataset": split,
        "domain": "workspace",
        "task_id": f"{split}_task_{task_index}",
        "victim_model": "synthetic",
        "steps": steps,
        "final_task_success": True,
        "final_attack_success": False,
        "final_policy_violation": False,
        "final_skill_path": path,
    }


def test_one_epoch_cpu_smoke_exercises_full_training_and_controls(tmp_path, monkeypatch):
    paths = {}
    for split, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        trajectory_path = tmp_path / f"{short}_trajectories.jsonl"
        metadata_path = tmp_path / f"{short}_metadata.jsonl"
        trajectories = [_trajectory_payload(split, index) for index in range(2)]
        trajectory_path.write_text(
            "\n".join(json.dumps(row) for row in trajectories) + "\n",
            encoding="utf-8",
        )
        metadata = [
            {
                "trajectory_id": row["trajectory_id"],
                "attack_family": "direct",
                "attack_role": "tool",
                "trigger_stage": "observation",
                "payload_position": "suffix",
                "knowledge_level": "none",
                "endpoint_policy": "sandbox",
                "required_tool_depth": 1,
            }
            for row in trajectories
        ]
        metadata_path.write_text(
            "\n".join(json.dumps(row) for row in metadata) + "\n",
            encoding="utf-8",
        )
        paths[split] = (trajectory_path, metadata_path)

    output = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--train",
            str(paths["train"][0]),
            "--validation",
            str(paths["validation"][0]),
            "--test",
            str(paths["test"][0]),
            "--train-metadata",
            str(paths["train"][1]),
            "--validation-metadata",
            str(paths["validation"][1]),
            "--test-metadata",
            str(paths["test"][1]),
            "--output-dir",
            str(output),
            "--variant",
            "semantic_event",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-size",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--max-generation-steps",
            "3",
            "--device",
            "cpu",
            "--run-prefix-controls",
        ],
    )
    MODULE.main()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data"]["dataset_audit"]["validation"]["task_count"] == 2
    assert metrics["prefix_value_controls"]["validation"]["shuffled"]
    assert metrics["prefix_value_controls"]["validation"][
        "semantic_markov_length_matched"
    ]
    assert "candidate_factorized_semantic_markov" in metrics["baselines"][
        "validation"
    ]
    assert (output / "task_macro_event_model.pt").exists()
