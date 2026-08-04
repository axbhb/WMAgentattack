"""Run the generic OOF ensemble summary for head-wise uncertainty methods."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_summary():
    path = ROOT / "scripts" / "78_summarize_v2_stability_ensemble.py"
    spec = importlib.util.spec_from_file_location("v2_ensemble_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_summary()
SUMMARY.FIXED_METHODS = (
    "mean_score",
    "risk_ucb_0p5",
    "utility_lcb_0p5",
    "asymmetric_ucb_lcb_0p5",
)
SUMMARY.METHODS = SUMMARY.FIXED_METHODS + ("validation_selected",)
SUMMARY.PRIMARY_METHOD = "risk_ucb_0p5"
SUMMARY.SUMMARY_SCOPE = (
    "20-task grouped OOF head-wise uncertainty ensemble comparison"
)


if __name__ == "__main__":
    SUMMARY.main()
