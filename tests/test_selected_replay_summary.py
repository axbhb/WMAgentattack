import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "30_summarize_selected_replay_multiseed.py"
    )
    spec = importlib.util.spec_from_file_location("selected_replay_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(security: bool, utility: bool):
    return {"security": security, "utility": utility}


def test_rates_and_cluster_bootstrap_have_expected_schema():
    module = _module()
    rows = [_row(True, True), _row(False, True)]
    rates = module._rates(rows)
    assert rates == {
        "attempt_count": 2,
        "observed_asr": 0.5,
        "observed_bup": 1.0,
        "asr_plus_bup": 1.5,
    }
    intervals = module._cluster_bootstrap(
        {("a", "b", "c"): rows}, samples=20, seed=7
    )
    assert set(intervals) == {
        "observed_asr_95ci",
        "observed_bup_95ci",
        "asr_plus_bup_95ci",
    }
    assert all(len(value) == 2 for value in intervals.values())
