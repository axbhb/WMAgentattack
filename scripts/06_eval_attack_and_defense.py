"""Evaluate planner baselines and the risk detector on a held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.defense import evaluate_risk_detector
from wmagentattack.io_utils import read_jsonl
from wmagentattack.planner import evaluate_attack_strategies
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
        default=ROOT / "artifacts" / "attack_defense_metrics.json",
    )
    parser.add_argument("--scope", default="offline_sandbox_eval")
    args = parser.parse_args()
    all_steps = [
        StepRecord.model_validate(row) for row in read_jsonl(args.test)
    ]
    trajectory_states = {}
    for step in all_steps:
        if step.target_skill is not None:
            trajectory_states.setdefault(step.trajectory_id, step)
    states = list(trajectory_states.values())
    model = SklearnWorldModel.load(args.model)
    attack_metrics = evaluate_attack_strategies(model, states)
    predictions = model.predict(all_steps)
    defense_metrics = evaluate_risk_detector(
        all_steps, predictions["risk_score"], threshold=0.5
    )
    output = {
        "scope": args.scope,
        "attack_metrics": attack_metrics,
        "defense_metrics": defense_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
