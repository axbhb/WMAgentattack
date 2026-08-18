import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPEC=importlib.util.spec_from_file_location("v16s",ROOT/"scripts"/"235_summarize_retrieval_support_gate_v16.py");MODULE=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MODULE)


def test_module_loads():assert callable(MODULE.main)
