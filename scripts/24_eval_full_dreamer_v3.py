"""Evaluate continuous utility and policy predictions from full DreamerV3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.full_dreamer_v3 import (
    FullSheepRLDreamerV3,
    evaluate_full_dreamer_predictions,
)
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.test)]
    model = FullSheepRLDreamerV3.load(args.model)
    metrics = evaluate_full_dreamer_predictions(steps, model.predict(steps))
    payload = {
        "backend": "sheeprl_full_dreamer_v3_offline",
        "test_steps": len(steps),
        "model": str(args.model.resolve()),
        "model_info": model.model_info(),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
