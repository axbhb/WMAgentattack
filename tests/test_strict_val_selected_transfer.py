import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "25_compare_val_selected_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("strict_val_transfer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(key: str, top_k: int = 16):
    return {
        "top_k": top_k,
        "utility_key": key,
        "objective_asr_plus_bup": 1.4,
        "observed_bup": 0.7,
        "observed_asr": 0.7,
        "conditional_coverage": 0.6,
    }


def test_utility_priority_breaks_only_exact_validation_ties():
    module = _module()
    rows = [_row("candidate_preservation_score"), _row("final_utility_score")]
    selected = module._select_validation_config(
        rows,
        0.5,
        ["final_utility_score", "candidate_preservation_score"],
    )
    assert selected["utility_key"] == "final_utility_score"

    rows[0]["objective_asr_plus_bup"] = 1.41
    selected = module._select_validation_config(
        rows,
        0.5,
        ["final_utility_score", "candidate_preservation_score"],
    )
    assert selected["utility_key"] == "candidate_preservation_score"


def test_priority_parser_rejects_duplicates():
    module = _module()
    try:
        module._parse_key_priority("final_utility_score,final_utility_score")
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Expected duplicate priorities to fail")
