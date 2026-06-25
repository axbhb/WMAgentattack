"""Rank benchmark-safe symbolic perturbations with the world model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl
from wmagentattack.planner import rank_candidates
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
        default=ROOT / "artifacts" / "planner_ranking.json",
    )
    args = parser.parse_args()
    model = SklearnWorldModel.load(args.model)
    rows = read_jsonl(args.test)
    candidate_steps = [
        StepRecord.model_validate(row)
        for row in rows
        if row.get("target_skill") is not None
    ]
    known_targets = set(model.skill_head.classes_)
    step = next(
        (
            candidate
            for candidate in candidate_steps
            if candidate.target_skill in known_targets
        ),
        candidate_steps[0],
    )
    ranking = rank_candidates(model, step, step.target_skill or "send_message")
    output = {
        "trajectory_id": step.trajectory_id,
        "target_skill": step.target_skill,
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
