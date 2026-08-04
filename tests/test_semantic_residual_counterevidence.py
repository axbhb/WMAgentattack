import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_residual_counterevidence",
    ROOT / "scripts" / "115_analyze_semantic_residual_counterevidence.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cluster_bootstrap_is_task_equal_weighted_and_directional():
    rows = [
        {"task_group": "a", "delta": -0.4},
        {"task_group": "a", "delta": -0.2},
        {"task_group": "b", "delta": -0.1},
    ]
    result = MODULE.cluster_bootstrap(rows, "delta", seed=7, draws=1000)
    assert result["task_count"] == 2
    assert result["task_equal_weight_point"] == pytest.approx(-0.2)
    assert result["ci95"][1] <= -0.1


def test_candidate_markov_never_generates_outside_initial_candidates():
    names = ["<PAD>", "<UNK>", "<BOS>", "record_read", "finish", "email_read"]
    model = MODULE.CandidateMarkov(names)
    train = [
        {
            "domain": "workspace",
            "steps": [
                {"selected_skill": "email_read"},
                {"selected_skill": "finish"},
            ],
        }
    ]
    model.fit(train)
    trajectory = {
        "domain": "workspace",
        "steps": [
            {
                "candidate_skills": ["record_read", "finish"],
                "selected_skill": "finish",
            }
        ],
    }
    generated = model.generate(trajectory, max_steps=3)
    assert set(generated) <= {"record_read", "finish"}
