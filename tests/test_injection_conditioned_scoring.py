import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "11_select_world_model_agentdojo_pairs.py"
    )
    spec = importlib.util.spec_from_file_location("pair_selector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_injection_text_is_deterministic_and_contains_all_locations():
    module = _module()
    raw = {
        "injections": {
            "z_location": "second instruction",
            "a_location": "first instruction",
        },
        "security": True,
        "utility": False,
    }
    text = module._injection_text_from_raw(raw)
    assert text == (
        "a_location: first instruction\n"
        "z_location: second instruction"
    )
    assert "security" not in text
    assert "utility" not in text
