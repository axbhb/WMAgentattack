"""Evaluate the SheepRL DreamerV3-style offline world model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.dreamer_world_model import SheepRLDreamerWorldModel, evaluate_dreamer_predictions
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "agentdojo_full_llama31_8b" / "splits" / "test_steps.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "agentdojo_full_llama31_8b_dreamer_world_model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "agentdojo_full_llama31_8b_dreamer_world_model_metrics.json")
    args = parser.parse_args()

    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.test)]
    model = SheepRLDreamerWorldModel.load(args.model)
    metrics = evaluate_dreamer_predictions(steps, model.predict(steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
