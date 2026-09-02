from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from wmagentattack.qwen_agentdojo import TransformersQwenLLM


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "agentdojo_full", ROOT / "scripts" / "10_run_agentdojo_hf_full.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_summary_reports_joint_and_conditional_attack_success() -> None:
    results = {
        "utility_results": {("u0", "i0"): True, ("u1", "i0"): False},
        "security_results": {("u0", "i0"): True, ("u1", "i0"): True},
        "injection_tasks_utility_results": {"i0": True},
    }
    summary = module._summarize_results(results)
    assert summary["utility_rate"] == 0.5
    assert summary["targeted_asr"] == 1.0
    assert summary["joint_task_and_attack_rate"] == 0.5
    assert summary["conditional_targeted_asr_given_utility"] == 1.0


def test_four_bit_compute_dtype_uses_fp16_on_volta(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 0))
    dtype = TransformersQwenLLM._four_bit_compute_dtype("cuda:0")
    assert dtype is torch.float16


def test_four_bit_compute_dtype_keeps_bf16_on_ampere(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    dtype = TransformersQwenLLM._four_bit_compute_dtype("cuda:0")
    assert dtype is torch.bfloat16
