"""Build the five-seed AgentDojo-v2 final dataset and probability labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.io_utils import read_jsonl, write_jsonl
from wmagentattack.multiseed_labels import build_multiseed_labels
from wmagentattack.schema import TrajectoryRecord


def _write_checksums(out_dir: Path) -> None:
    lines = []
    for path in sorted(out_dir.glob("*.json*")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.name}")
    (out_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _set_final_dataset(
    trajectory: TrajectoryRecord,
    *,
    dataset_name: str,
    annotation: dict[str, Any],
) -> TrajectoryRecord:
    step_update = {
        key: value
        for key, value in annotation.items()
        if key
        in {
            "multiseed_group_id",
            "multiseed_trials",
            "base_task_success_rate",
            "preservation_trainable",
            "preservation_weight",
            "utility_probability_target",
            "preservation_probability_target",
            "attack_probability_target",
            "joint_success_probability_target",
            "probability_label_alpha",
            "probability_label_beta",
            "probability_label_variance",
            "probability_label_confidence",
            "attack_probability_confidence",
            "joint_success_probability_confidence",
            "joint_outcome_counts",
            "joint_outcome_dirichlet_alpha",
            "joint_outcome_probability_target",
            "joint_outcome_trials",
            "clean_utility_logit_prior",
            "attack_utility_logit_residual_target",
            "probability_label_source",
        }
    }
    steps = [
        step.model_copy(update={"dataset": dataset_name, **step_update})
        for step in trajectory.steps
    ]
    return trajectory.model_copy(update={"dataset": dataset_name, "steps": steps})


def _seed_audit(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        if row["source_kind"] == "attack":
            grouped[int(row["run_seed"])].append(row)
    return {
        str(seed): {
            "trajectories": len(rows),
            "utility_success": sum(bool(row["utility"]) for row in rows),
            "attack_success": sum(bool(row["security"]) for row in rows),
            "BUP": sum(bool(row["utility"]) for row in rows) / len(rows),
            "ASR": sum(bool(row["security"]) for row in rows) / len(rows),
        }
        for seed, rows in sorted(grouped.items())
    }


def _split_group_audit(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = [
            row
            for row in group_rows
            if row["source_kind"] == "attack" and row["task_split"] == split
        ]
        total_trials = sum(row["utility_probability_trials"] for row in rows)
        output[split] = {
            "attack_groups": len(rows),
            "attack_trajectories": total_trials,
            "empirical_BUP": sum(
                row["utility_probability_successes"] for row in rows
            )
            / total_trials,
            "empirical_ASR": sum(
                row["attack_probability_successes"] for row in rows
            )
            / total_trials,
            "mixed_utility_groups": sum(
                0
                < row["utility_probability_successes"]
                < row["utility_probability_trials"]
                for row in rows
            ),
            "mixed_attack_groups": sum(
                0
                < row["attack_probability_successes"]
                < row["attack_probability_trials"]
                for row in rows
            ),
            "family_groups": dict(Counter(row["attack_family"] for row in rows)),
            "domain_groups": dict(Counter(row["suite"] for row in rows)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="v2.1-final-5seed")
    parser.add_argument(
        "--expected-attack-seed", type=int, action="append", required=True
    )
    parser.add_argument("--expected-attack-groups", type=int, default=400)
    parser.add_argument("--min-clean-seeds", type=int, default=3)
    parser.add_argument("--min-base-success-rate", type=float, default=0.5)
    parser.add_argument("--preservation-weight-floor", type=float, default=0.05)
    args = parser.parse_args()

    metadata = read_jsonl(args.split_dir / "metadata.jsonl")
    raw_trajectories = read_jsonl(args.split_dir / "trajectories.jsonl")
    trajectories = {
        str(row["trajectory_id"]): TrajectoryRecord.model_validate(row)
        for row in raw_trajectories
    }
    if len(trajectories) != len(raw_trajectories):
        raise ValueError("Duplicate trajectory ids in source dataset")
    metadata_by_id = {str(row["trajectory_id"]): row for row in metadata}
    if set(trajectories) != set(metadata_by_id):
        raise ValueError("Trajectory/metadata id sets differ")

    annotations, group_rows, label_audit = build_multiseed_labels(
        metadata,
        expected_attack_seeds=args.expected_attack_seed,
        min_clean_seeds=args.min_clean_seeds,
        min_base_success_rate=args.min_base_success_rate,
        preservation_weight_floor=args.preservation_weight_floor,
    )
    if label_audit["attack_groups"] != args.expected_attack_groups:
        raise ValueError(
            f"Expected {args.expected_attack_groups} attack groups, "
            f"found {label_audit['attack_groups']}"
        )

    dataset_name = f"agentdojo-{args.dataset_version}"
    ordered_ids = sorted(trajectories)
    final_trajectories = [
        _set_final_dataset(
            trajectories[trajectory_id],
            dataset_name=dataset_name,
            annotation=annotations[trajectory_id],
        )
        for trajectory_id in ordered_ids
    ]
    final_metadata = [
        {
            **metadata_by_id[trajectory_id],
            **annotations[trajectory_id],
            "dataset_version": args.dataset_version,
        }
        for trajectory_id in ordered_ids
    ]
    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in final_trajectories
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "trajectories.jsonl", final_trajectories)
    write_jsonl(
        args.out_dir / "steps.jsonl",
        [step for trajectory in final_trajectories for step in trajectory.steps],
    )
    write_jsonl(args.out_dir / "metadata.jsonl", final_metadata)
    write_jsonl(args.out_dir / "label_groups.jsonl", group_rows)

    for split in ("train", "val", "test"):
        split_metadata = [row for row in final_metadata if row["task_split"] == split]
        split_trajectories = [
            trajectory_by_id[row["trajectory_id"]] for row in split_metadata
        ]
        split_groups = [row for row in group_rows if row["task_split"] == split]
        write_jsonl(args.out_dir / f"{split}_metadata.jsonl", split_metadata)
        write_jsonl(args.out_dir / f"{split}_trajectories.jsonl", split_trajectories)
        write_jsonl(
            args.out_dir / f"{split}_steps.jsonl",
            [step for trajectory in split_trajectories for step in trajectory.steps],
        )
        write_jsonl(args.out_dir / f"{split}_label_groups.jsonl", split_groups)

    source_audit_path = args.split_dir / "audit.json"
    source_audit = (
        json.loads(source_audit_path.read_text(encoding="utf-8"))
        if source_audit_path.exists()
        else None
    )
    split_group_audit = _split_group_audit(group_rows)
    final_audit = {
        "scope": "AgentDojo sandbox only; inert prompt-injection text",
        "dataset_version": args.dataset_version,
        "dataset_name": dataset_name,
        "source_split_dir": str(args.split_dir.resolve()),
        "expected_attack_seeds": sorted(set(args.expected_attack_seed)),
        "total_trajectories": len(final_trajectories),
        "total_steps": sum(len(item.steps) for item in final_trajectories),
        "attack_trajectories": sum(
            row["source_kind"] == "attack" for row in final_metadata
        ),
        "clean_trajectories": sum(
            row["source_kind"] == "clean" for row in final_metadata
        ),
        "label_groups": len(group_rows),
        "label_audit": label_audit,
        "seed_audit": _seed_audit(final_metadata),
        "split_group_audit": split_group_audit,
        "source_audit": source_audit,
        "checks": {
            "expected_attack_groups": (
                label_audit["attack_groups"] == args.expected_attack_groups
            ),
            "five_seed_rectangular_attack_design": all(
                row["utility_probability_trials"]
                == len(set(args.expected_attack_seed))
                for row in group_rows
                if row["source_kind"] == "attack"
            ),
            "trajectory_metadata_alignment": (
                set(trajectory_by_id)
                == {str(row["trajectory_id"]) for row in final_metadata}
            ),
            "task_overlap_zero": bool(
                source_audit
                and source_audit.get("checks", {}).get("task_overlap_zero")
            ),
            "test_family_unseen_in_train": bool(
                source_audit
                and source_audit.get("checks", {}).get(
                    "test_family_unseen_in_train"
                )
            ),
            "autodojo_transfer_not_in_val_or_test": bool(
                source_audit
                and source_audit.get("checks", {}).get(
                    "autodojo_transfer_not_in_val_or_test"
                )
            ),
        },
    }
    if not all(final_audit["checks"].values()):
        raise RuntimeError(f"v2-final audit failed: {final_audit['checks']}")
    (args.out_dir / "audit.json").write_text(
        json.dumps(final_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_checksums(args.out_dir)
    print(json.dumps(final_audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
