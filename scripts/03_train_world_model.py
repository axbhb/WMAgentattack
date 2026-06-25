"""Train the sklearn multi-head world model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import StepRecord
from wmagentattack.world_model import SklearnWorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "splits" / "train_steps.jsonl",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=ROOT / "artifacts" / "world_model.joblib",
    )
    args = parser.parse_args()
    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.train)]
    model = SklearnWorldModel().fit(steps)
    model.save(args.model_out)
    print(
        json.dumps(
            {
                "train_steps": len(steps),
                "model_out": str(args.model_out.resolve()),
                "skill_classes": model.skill_head.classes_.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

