import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "29_export_frozen_replay_selections.py"
    )
    spec = importlib.util.spec_from_file_location("export_frozen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_export_reconstructs_exact_top_k():
    module = _module()
    candidates = {
        "candidates": [
            {
                "suite": "workspace",
                "user_task_id": f"user_task_{index}",
                "injection_task_id": "injection_task_0",
                "observed_security": index < 2,
                "observed_utility": index != 2,
                "risk_score": float(4 - index),
                "utility_score": float(index) / 4,
            }
            for index in range(4)
        ]
    }
    strict = {
        "pareto": {
            "selected_validation_config": {
                "top_k": 2,
                "utility_key": "utility_score",
                "threshold_mode": "quantile",
                "threshold_value": 0.5,
            },
            "test_frozen_validation_threshold": {
                "per_seed": [{"seed": 7, "threshold": 0.5}]
            },
        }
    }
    selected, metadata = module._select_frozen(candidates, strict)
    assert len(selected) == 2
    assert metadata["frozen_numeric_threshold"] == 0.5
    assert metadata["cached_asr_plus_bup"] >= 0.0
