"""Run a no-GPU selection grid over top-k and seed values.

This script repeatedly calls ``11_select_world_model_agentdojo_pairs.py`` and
collects compact method-level summaries. It is intentionally selection-only:
use it to validate the experimental design before launching expensive replay
jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-backend",
        choices=["sklearn", "dreamer"],
        default="sklearn",
    )
    parser.add_argument("--allowed-trajectories", type=Path, required=True)
    parser.add_argument("--standardized-steps", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", default="16,24,32,48")
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--scoring-mode", default="clean_prefix")
    parser.add_argument("--rollout-horizon", type=int, default=3)
    parser.add_argument("--max-per-user-task", type=int, default=2)
    parser.add_argument("--attack", default="important_instructions_no_model_name")
    parser.add_argument("--exclude-world-model-from-baselines", action="store_true")
    parser.add_argument("--score-risk-weight", type=float, default=1.0)
    parser.add_argument("--score-mean-risk-weight", type=float, default=0.3)
    parser.add_argument("--score-utility-weight", type=float, default=0.5)
    parser.add_argument("--score-target-weight", type=float, default=0.3)
    parser.add_argument("--score-target-reached-weight", type=float, default=0.2)
    args = parser.parse_args()

    top_ks = _parse_int_list(args.top_k)
    seeds = _parse_int_list(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    selector = ROOT / "scripts" / "11_select_world_model_agentdojo_pairs.py"
    for top_k in top_ks:
        for seed in seeds:
            output = args.out_dir / f"selection_{args.scoring_mode}_k{top_k}_seed{seed}.json"
            command = [
                sys.executable,
                str(selector),
                "--run-root",
                str(args.run_root),
                "--model",
                str(args.model),
                "--model-backend",
                args.model_backend,
                "--output",
                str(output),
                "--allowed-trajectories",
                str(args.allowed_trajectories),
                "--scoring-mode",
                args.scoring_mode,
                "--rollout-horizon",
                str(args.rollout_horizon),
                "--top-k",
                str(top_k),
                "--seed",
                str(seed),
                "--max-per-user-task",
                str(args.max_per_user_task),
                "--attack",
                args.attack,
                "--score-risk-weight",
                str(args.score_risk_weight),
                "--score-mean-risk-weight",
                str(args.score_mean_risk_weight),
                "--score-utility-weight",
                str(args.score_utility_weight),
                "--score-target-weight",
                str(args.score_target_weight),
                "--score-target-reached-weight",
                str(args.score_target_reached_weight),
            ]
            if args.standardized_steps:
                command.extend(["--standardized-steps", str(args.standardized_steps)])
            if args.exclude_world_model_from_baselines:
                command.append("--exclude-world-model-from-baselines")
            subprocess.run(command, check=True, cwd=ROOT)
            payload = json.loads(output.read_text(encoding="utf-8"))
            for method, metrics in payload["summary"].items():
                summary_rows.append(
                    {
                        "scoring_mode": args.scoring_mode,
                        "top_k": top_k,
                        "seed": seed,
                        "method": method,
                        **metrics,
                        "candidate_count": payload["candidate_count"],
                        "max_per_user_task": payload["max_per_user_task"],
                        "exclude_world_model_from_baselines": payload[
                            "exclude_world_model_from_baselines"
                        ],
                    }
                )

    summary_json = {
        "scope": "selection_grid",
        "top_k": top_ks,
        "seeds": seeds,
        "scoring_mode": args.scoring_mode,
        "model_backend": args.model_backend,
        "standardized_steps": str(args.standardized_steps.resolve()) if args.standardized_steps else None,
        "selection_score_weights": {
            "risk": args.score_risk_weight,
            "mean_risk": args.score_mean_risk_weight,
            "utility": args.score_utility_weight,
            "target": args.score_target_weight,
            "target_reached": args.score_target_reached_weight,
        },
        "rollout_horizon": args.rollout_horizon,
        "exclude_world_model_from_baselines": (
            args.exclude_world_model_from_baselines
        ),
        "rows": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
