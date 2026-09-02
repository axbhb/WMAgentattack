import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "large_cache", ROOT / "scripts" / "297_build_large_semantic_cache_v45.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mean_pool_ignores_padding():
    module = load_script()
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(module.mean_pool(hidden, mask), torch.tensor([[2.0, 4.0]]))


def test_v100_uses_fp16_and_ampere_uses_bf16():
    module = load_script()
    assert module.cuda_dtype_for_capability(7) is torch.float16
    assert module.cuda_dtype_for_capability(8) is torch.bfloat16
