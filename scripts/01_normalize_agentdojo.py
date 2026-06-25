"""Normalize raw AgentDojo traces into step and trajectory JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import write_jsonl
from wmagentattack.normalize_agentdojo import normalize_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "runs"
        / "qwen2.5-7b-instruct-transformers-4bit-compact12000-robust",
    )
    parser.add_argument(
        "--step-out",
        type=Path,
        default=ROOT / "data" / "processed" / "agentdojo_steps.jsonl",
    )
    parser.add_argument(
        "--trajectory-out",
        type=Path,
        default=ROOT / "data" / "processed" / "agentdojo_trajectories.jsonl",
    )
    args = parser.parse_args()
    trajectories = normalize_directory(args.input)
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    write_jsonl(args.step_out, steps)
    write_jsonl(args.trajectory_out, trajectories)
    print(
        json.dumps(
            {
                "trajectories": len(trajectories),
                "steps": len(steps),
                "step_out": str(args.step_out.resolve()),
                "trajectory_out": str(args.trajectory_out.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
