import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "large_train", ROOT / "scripts" / "298_train_large_hybrid_world_model_v45.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trajectory_windows_are_contiguous_and_task_filtered():
    module = load_script()
    events = [
        {"task_name": "a", "trajectory_id": "x", "step_id": i}
        for i in range(4)
    ] + [{"task_name": "b", "trajectory_id": "y", "step_id": 0}]
    arrays = {"target": np.asarray([1, 2, 3, -1, 0])}
    windows = module.trajectory_windows(events, arrays, {"a"}, max_horizon=3)
    assert all(events[row["start"]]["task_name"] == "a" for row in windows)
    assert {row["horizon"] for row in windows} == {1, 2, 3}
    assert all(len(row["future_rows"]) == row["horizon"] - 1 for row in windows)


def test_task_weights_balance_tasks():
    module = load_script()
    events = [{"task_name": "a"}, {"task_name": "a"}, {"task_name": "b"}]
    weights = module.task_weights(events, np.asarray([0, 1, 2]))
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights.mean(), 1.0)


def test_frozen_protocol_builds_requested_large_architecture():
    module = load_script()
    protocol = json.loads(
        (ROOT / "configs" / "0902_large_hybrid_world_model_v45_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    config = module.model_config(protocol)
    assert config.semantic_size == 768
    assert config.hidden_size == 768
    assert config.state_layers == 8
    assert config.action_layers == 6
    assert config.residual_layers == 6
    assert config.attention_heads == 12
    assert config.memory_tokens == 8
    assert protocol["submission"]["submitted"] is False


def test_requested_architecture_is_over_one_hundred_million_trainable_parameters():
    module = load_script()
    protocol = json.loads(
        (ROOT / "configs" / "0902_large_hybrid_world_model_v45_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    model = module.LargeHybridWorldModel(module.model_config(protocol))
    breakdown = module.parameter_breakdown(model)
    assert breakdown["structured_state_encoder"] > 50_000_000
    assert breakdown["victim_action_dynamics"] > 40_000_000
    assert breakdown["multi_step_residual_dynamics"] > 40_000_000
    assert breakdown["total"] > 140_000_000
