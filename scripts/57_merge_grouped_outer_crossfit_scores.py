"""Merge held-fold Dreamer scores into one task-out-of-model OOF archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SEEDS = (7, 13, 21)
MODES = ("clean_prefix_rollout", "injection_conditioned_rollout")


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidates missing in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--expected-pairs", type=int, default=664)
    parser.add_argument("--expected-tasks", type=int, default=68)
    parser.add_argument("--output-archive", type=Path, required=True)
    args = parser.parse_args()

    summaries = {}
    key_sets = []
    for seed in SEEDS:
        for mode in MODES:
            rows = []
            sources = []
            for fold in range(args.folds):
                path = (
                    args.fold_root
                    / f"fold{fold}"
                    / f"seed{seed}"
                    / f"held_{mode}_candidates.json"
                )
                fold_rows = _load_candidates(path)
                rows.extend(fold_rows)
                sources.append(
                    {
                        "fold": fold,
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "candidate_count": len(fold_rows),
                    }
                )
            keys = [_pair_key(row) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate OOF pair for seed={seed} mode={mode}")
            if len(rows) != args.expected_pairs:
                raise ValueError(
                    f"Expected {args.expected_pairs} rows, found {len(rows)}"
                )
            tasks = {(row["suite"], row["user_task_id"]) for row in rows}
            if len(tasks) != args.expected_tasks:
                raise ValueError(
                    f"Expected {args.expected_tasks} tasks, found {len(tasks)}"
                )
            key_sets.append(set(keys))
            output = {
                "scope": "outer_crossfit_task_out_of_model_world_scores",
                "seed": seed,
                "scoring_mode": mode,
                "fold_count": args.folds,
                "candidate_count": len(rows),
                "task_count": len(tasks),
                "every_task_scored_by_checkpoint_that_excluded_it": True,
                "sources": sources,
                "candidates": sorted(rows, key=_pair_key),
            }
            output_path = (
                args.output_archive
                / f"seed{seed}"
                / f"train_{mode}_candidates.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            summaries[f"seed{seed}_{mode}"] = {
                "path": str(output_path.resolve()),
                "sha256": _sha256(output_path),
                "candidate_count": len(rows),
            }
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Merged seeds or views do not align")
    manifest = {
        "scope": "merged_grouped_outer_crossfit_oof_archive",
        "pair_count": len(key_sets[0]),
        "task_count": args.expected_tasks,
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "outputs": summaries,
    }
    args.output_archive.mkdir(parents=True, exist_ok=True)
    args.output_archive.joinpath("merge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
