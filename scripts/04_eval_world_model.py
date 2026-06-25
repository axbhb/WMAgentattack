"""Evaluate the world model on held-out synthetic trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl
from wmagentattack.metrics import evaluate_predictions
from wmagentattack.schema import StepRecord
from wmagentattack.world_model import SklearnWorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data" / "splits" / "test_steps.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "world_model.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "world_model_metrics.json",
    )
    args = parser.parse_args()
    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.test)]
    model = SklearnWorldModel.load(args.model)
    metrics = evaluate_predictions(steps, model.predict(steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

