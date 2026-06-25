"""Summarize clean AgentDojo traces produced by the local Qwen pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmagentattack.trajectory_stats import summarize_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "clean_baseline_summary.json",
    )
    parser.add_argument(
        "--pipeline",
        help="Only include traces whose pipeline_name exactly matches this value.",
    )
    args = parser.parse_args()

    paths = sorted(args.run_root.rglob("none/none.json"))
    if args.pipeline:
        paths = [
            path
            for path in paths
            if json.loads(path.read_text(encoding="utf-8")).get("pipeline_name")
            == args.pipeline
        ]
    summary = summarize_traces(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
