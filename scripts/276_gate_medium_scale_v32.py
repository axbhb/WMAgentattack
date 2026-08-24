"""Apply the frozen v32 capacity scaling gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.medium_scale_diagnostic import evaluate_medium_scale_gate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--action-predictions", type=Path, required=True)
    parser.add_argument("--small-action-predictions", type=Path, required=True)
    parser.add_argument("--small-effect-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    controls = protocol["external_controls"]
    for path, expected in (
        (args.action_predictions, metrics["action_predictions_sha256"]),
        (args.small_action_predictions, controls["small_action_predictions_sha256"]),
        (args.small_effect_metrics, controls["small_effect_metrics_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"v32 control/output hash mismatch: {path}")
    baseline_effect = json.loads(args.small_effect_metrics.read_text(encoding="utf-8"))["runs"]
    payload = evaluate_medium_scale_gate(
        action_baseline_rows=read_jsonl(args.small_action_predictions),
        action_candidate_rows=read_jsonl(args.action_predictions),
        effect_baseline_rows=baseline_effect,
        effect_candidate_rows=metrics["effect_runs"],
        training_metrics=metrics,
        thresholds=protocol["acceptance_gate"],
    )
    payload["hashes"] = {
        "protocol": sha256(args.protocol),
        "training_metrics": sha256(args.metrics),
        "action_predictions": sha256(args.action_predictions),
        "small_action_predictions": sha256(args.small_action_predictions),
        "small_effect_metrics": sha256(args.small_effect_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed"], "total": payload["total"]}))


if __name__ == "__main__":
    main()
