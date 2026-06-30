"""Train the SheepRL DreamerV3-style offline world model on AgentDojo traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.dreamer_world_model import DreamerWorldModelConfig, SheepRLDreamerWorldModel
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "agentdojo_full_llama31_8b" / "splits" / "train_steps.jsonl")
    parser.add_argument("--model-out", type=Path, default=ROOT / "artifacts" / "agentdojo_full_llama31_8b_dreamer_world_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=768)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--recurrent-state-size", type=int, default=256)
    parser.add_argument("--stochastic-size", type=int, default=16)
    parser.add_argument("--discrete-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.train)]
    config = DreamerWorldModelConfig(
        obs_dim=args.obs_dim,
        dense_units=args.dense_units,
        recurrent_state_size=args.recurrent_state_size,
        stochastic_size=args.stochastic_size,
        discrete_size=args.discrete_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = SheepRLDreamerWorldModel(config).fit(steps, epochs=args.epochs, batch_size=args.batch_size)
    model.save(args.model_out)
    result = {
        "train_steps": len(steps),
        "model_out": str(args.model_out.resolve()),
        "skill_classes": model.skill_classes,
        "training_history": model.training_history,
        "backend": "sheeprl_dreamer_v3_offline",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
