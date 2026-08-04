"""Label-blind preflight for the frozen Stage 3 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from wmagentattack.markov_sufficiency import (
    FROZEN_SUFFICIENCY_VARIANTS,
    representation_feature_size,
    representation_feature_vector,
    validate_dataset_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--semantic-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = json.loads(args.source_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    if _sha256(args.source_dataset) != protocol["source"]["raw_dataset_sha256"]:
        raise ValueError("raw dataset hash mismatch")
    if _sha256(args.semantic_dataset) != protocol["source"]["semantic_dataset_sha256"]:
        raise ValueError("semantic dataset hash mismatch")
    if tuple(protocol["frozen_variants"]) != FROZEN_SUFFICIENCY_VARIANTS:
        raise ValueError("representation protocol mismatch")
    validate_dataset_alignment(source, semantic)
    hash_dimension = int(protocol["training"]["hash_dimension"])
    expected_size = representation_feature_size(hash_dimension)
    counts = {variant: 0 for variant in FROZEN_SUFFICIENCY_VARIANTS}
    nonfinite: list[str] = []
    nondeterministic: list[str] = []
    split_tasks: dict[str, set[str]] = {}
    prefixes = 0

    for source_episode, semantic_episode in zip(
        source["episodes"], semantic["episodes"]
    ):
        split_tasks.setdefault(str(source_episode["split"]), set()).add(
            str(source_episode["task_id"])
        )
        for index in range(len(source_episode["prefixes"])):
            prefixes += 1
            for variant in FROZEN_SUFFICIENCY_VARIANTS:
                first = representation_feature_vector(
                    variant=variant,
                    source_prefixes=source_episode["prefixes"],
                    semantic_prefixes=semantic_episode["prefixes"],
                    prefix_index=index,
                    hash_dimension=hash_dimension,
                )
                second = representation_feature_vector(
                    variant=variant,
                    source_prefixes=source_episode["prefixes"],
                    semantic_prefixes=semantic_episode["prefixes"],
                    prefix_index=index,
                    hash_dimension=hash_dimension,
                )
                identity = f"{source_episode['episode_id']}::p{index}::{variant}"
                if first.shape != (expected_size,) or not np.isfinite(first).all():
                    nonfinite.append(identity)
                if not np.array_equal(first, second):
                    nondeterministic.append(identity)
                counts[variant] += 1

    split_names = sorted(split_tasks)
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    checks = {
        "episode_budget": len(source["episodes"])
        == int(protocol["fixed_budget"]["episodes"]),
        "prefix_budget": prefixes == int(protocol["fixed_budget"]["prefixes"]),
        "all_variants_complete": all(
            value == prefixes for value in counts.values()
        ),
        "equal_finite_dimension": not nonfinite,
        "deterministic": not nondeterministic,
        "task_disjoint": not any(overlaps.values()),
    }
    audit = {
        "protocol_sha256": _sha256(args.protocol),
        "raw_dataset_sha256": _sha256(args.source_dataset),
        "semantic_dataset_sha256": _sha256(args.semantic_dataset),
        "episodes": len(source["episodes"]),
        "prefixes": prefixes,
        "feature_size": expected_size,
        "variant_prefix_counts": counts,
        "split_task_counts": {
            split: len(tasks) for split, tasks in sorted(split_tasks.items())
        },
        "split_task_overlaps": overlaps,
        "nonfinite_or_wrong_size": nonfinite,
        "nondeterministic": nondeterministic,
        "checks": checks,
        "decision": (
            "GO__MARKOV_SUFFICIENCY_TRAINING_AUTHORIZED"
            if all(checks.values())
            else "NO_GO__MARKOV_SUFFICIENCY_PREFLIGHT_FAILED"
        ),
    }
    _write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit("Stage 3 preflight gate failed")


if __name__ == "__main__":
    main()
