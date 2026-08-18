import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPEC=importlib.util.spec_from_file_location("v15",ROOT/"scripts"/"233_summarize_late_fusion_distillation_v15.py");MODULE=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MODULE)


def test_effect_direction():
    left=[];right=[]
    for seed in (7,17,29):
        for task in ("a","b"):
            common={"training_seed":seed,"task_name":task};left.append({**common,"x":2.0});right.append({**common,"x":1.5})
    assert MODULE._effect(left,right,"x")["mean"]==0.5
