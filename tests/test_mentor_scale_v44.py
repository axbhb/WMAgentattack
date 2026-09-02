import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "v44_gate", ROOT / "scripts" / "295_gate_mentor_scale_best_wm_v44.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_protocol_budget_and_no_content_hashes():
    protocol = json.loads((ROOT / "configs" / "0902_mentor_scale_best_wm_v44_protocol.json").read_text())
    budget = protocol["fixed_budget"]
    assert budget["total_model_fits"] == 105
    assert budget["baseline_teacher_fits"] + budget["joint_teacher_fits"] + budget["residual_fits"] + budget["effect_baseline_fits"] + budget["effect_candidate_fits"] == 105
    assert "sha256" not in json.dumps(protocol).lower()
    assert protocol["scope"]["content_checksums"] is False


def test_positive_fraction():
    gate = load_gate()
    assert gate.positive_fraction([1.0, -1.0, 2.0, 3.0]) == 0.75
    assert gate.positive_fraction([]) == 0.0


def test_paired_gain_uses_common_keys_only():
    gate = load_gate()
    assert gate.paired_gain({"a": 1.0, "b": 5.0}, {"a": 3.0, "c": 9.0}) == {"a": 2.0}
