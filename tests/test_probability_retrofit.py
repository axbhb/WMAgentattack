import importlib.util
from pathlib import Path


def _module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index, security=False, utility=True):
    return {
        "suite": "workspace",
        "user_task_id": f"user_task_{index}",
        "injection_task_id": "injection_task_1",
        "security": security,
        "utility": utility,
    }


def test_export_finds_only_target_pair_absent_from_reference_union():
    module = _module("34_export_probability_retrofit_selection.py")
    rows = [_row(index) for index in range(4)]
    selections = {
        "target": rows[:3],
        "left": [rows[0]],
        "right": [rows[1]],
    }
    assert module._missing_target_rows(
        selections, "target", ["left", "right"]
    ) == [rows[2]]


def test_reconstruct_uses_shared_cache_and_retrofit_pair():
    module = _module("35_reconstruct_six_seed_probability_eval.py")
    shared = _row(0, security=True, utility=True)
    missing = _row(1, security=False, utility=True)
    historic = {
        "seed": 7,
        "do_sample": True,
        "results": {"old": {"rows": [shared]}},
    }
    retrofit = {
        "seed": 7,
        "do_sample": True,
        "results": {"retro": {"rows": [missing]}},
    }
    output = module._reconstruct_seed(
        historic,
        retrofit,
        {"new": [shared, missing], "old": [shared]},
    )
    assert len(output["results"]["new"]["rows"]) == 2
    assert output["results"]["new"]["aggregate"]["ASR"] == 0.5
    assert output["unique_pair_count"] == 2
