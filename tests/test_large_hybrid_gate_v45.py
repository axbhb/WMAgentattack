import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "large_gate", ROOT / "scripts" / "299_gate_large_hybrid_world_model_v45.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_gain_requires_identical_surface():
    module = load_script()
    baseline = {(7, 2, "task-a"): 2.0}
    candidate = {(7, 2, "task-a"): 1.5}
    assert module.paired_gain(baseline, candidate) == {(7, 2, "task-a"): 0.5}


def test_horizon_weighted_gain_is_unit_balanced():
    module = load_script()
    gains = {
        (7, 2, "a"): 1.0,
        (7, 3, "a"): 0.0,
        (17, 2, "a"): 0.5,
        (17, 3, "a"): 0.5,
    }
    value = module.weighted_multistep_gain(gains, {"2": 1.0, "3": 0.5})
    assert abs(value - ((2.0 / 3.0 + 0.5) / 2.0)) < 1e-9


def test_suite_inference_uses_task_identity_not_outcome():
    module = load_script()
    event = {
        "task_name": "travel_user_task_3",
        "causal_model_input": {"source": "agentdojo"},
    }
    assert module.infer_agentdojo_suite(event) == "travel"
