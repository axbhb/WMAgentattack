"""Average leakage-safe candidate predictions across repeated CV partitions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_seed_ensemble_module():
    path = ROOT / "scripts" / "27_seed_ensemble_candidates.py"
    spec = importlib.util.spec_from_file_location("seed_ensemble_candidates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import seed ensemble helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEED_ENSEMBLE = _load_seed_ensemble_module()


def _ensemble_payloads(
    payloads: list[dict[str, Any]],
    input_roots: list[Path],
) -> dict[str, Any]:
    candidates = SEED_ENSEMBLE._average_candidates(
        [payload["candidates"] for payload in payloads]
    )
    return {
        **payloads[0],
        "candidate_repeated_cv_ensemble": {
            "input_roots": [str(path.resolve()) for path in input_roots],
            "score_aggregation": "arithmetic_mean",
            "validation_prediction_rule": "mean of grouped OOF predictions",
            "test_prediction_rule": "mean of cross-fit ensemble predictions",
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", type=Path, required=True)
    parser.add_argument("--input-seed", type=int, default=7)
    parser.add_argument("--output-seed", type=int, default=7)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if len(args.input_root) < 2:
        raise ValueError("Repeated-CV ensemble requires at least two input roots")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "candidate_repeated_cv_ensemble",
        "input_roots": [str(path.resolve()) for path in args.input_root],
        "input_seed": args.input_seed,
        "output_seed": args.output_seed,
        "repeat_count": len(args.input_root),
        "splits": {},
    }
    for split in ("val", "test"):
        payloads = [
            json.loads(
                (root / f"seed{args.input_seed}_{split}_candidates.json").read_text(
                    encoding="utf-8"
                )
            )
            for root in args.input_root
        ]
        output = _ensemble_payloads(payloads, args.input_root)
        path = args.output_root / f"seed{args.output_seed}_{split}_candidates.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        summary["splits"][split] = {
            "candidate_count": len(output["candidates"]),
            "output": str(path.resolve()),
        }

    args.output_root.joinpath("repeated_cv_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
