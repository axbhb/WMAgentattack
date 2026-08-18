import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SPEC=importlib.util.spec_from_file_location("v16",ROOT/"scripts"/"234_run_retrieval_support_gate_v16.py");MODULE=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MODULE)


def test_topk_descending():
    query=np.asarray([[1.,0.]],np.float32);index=np.asarray([[0.,1.],[1.,0.],[.5,.5]],np.float32);values,indices=MODULE._topk(query,index,2);assert indices.tolist()==[[1,2]];assert values[0,0]>=values[0,1]


def test_normalize_finite_for_zero_rows():
    output=MODULE._normalize(np.zeros((2,3),np.float32));assert np.isfinite(output).all()
