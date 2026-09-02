#!/usr/bin/env python3
"""Run one synthetic forward/backward pass for the full v45 neural architecture."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from wmagentattack.large_hybrid_world_model import (
    LargeHybridWorldModel,
    LargeWorldModelConfig,
    parameter_breakdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v45 GPU smoke test")
    torch.manual_seed(45)
    torch.cuda.manual_seed_all(45)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model = LargeHybridWorldModel(LargeWorldModelConfig()).to(device).train()
    fields = torch.randn(args.batch_size, 5, 768, device=device)
    field_mask = torch.ones(args.batch_size, 5, dtype=torch.bool, device=device)
    candidates = torch.randn(args.candidates, 768, device=device)
    labels = torch.arange(args.batch_size, device=device) % args.candidates
    joint_labels = torch.arange(args.batch_size, device=device) % 4

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.teacher(fields, field_mask, candidates)
        hidden = model.residual_dynamics.advance(
            output["state"], output["candidate_hidden"][labels], step=1
        )
        rollout_logits = model.residual_dynamics.rollout_logits(
            hidden, output["candidate_hidden"]
        )
        loss = (
            F.cross_entropy(output["action_logits"], labels)
            + F.cross_entropy(output["joint_logits"], joint_labels)
            + F.cross_entropy(rollout_logits, labels)
            + output["outcome_logits"].square().mean()
        )
    loss.backward()
    torch.cuda.synchronize(device)

    payload = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "autocast_dtype": "float16",
        "batch_size": args.batch_size,
        "candidate_count": args.candidates,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach().cpu()),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "parameters": parameter_breakdown(model),
    }
    if not payload["loss_finite"]:
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
