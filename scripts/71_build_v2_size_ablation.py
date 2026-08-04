"""Build nested, group-complete AgentDojo-v2 training-size subsets.

Sampling is deterministic and label-blind.  Every subset keeps all clean
trajectories, all victim-seed repetitions of a selected attack configuration,
and a balanced number of configurations from every training user task.  The
per-task ordering round-robins attack families before repeating a family, so
smaller subsets retain attack diversity while remaining nested in larger ones.
"""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: str) -> str:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_info(metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        group_id = row.get("multiseed_group_id")
        if not group_id:
            raise ValueError("Training metadata row lacks multiseed_group_id")
        grouped[str(group_id)].append(row)
    output = {}
    for group_id, rows in grouped.items():
        source_kinds = {str(row["source_kind"]) for row in rows}
        task_keys = {
            f"{row['suite']}|{row['user_task_id']}" for row in rows
        }
        if len(source_kinds) != 1 or len(task_keys) != 1:
            raise ValueError(f"Inconsistent metadata group: {group_id}")
        source_kind = next(iter(source_kinds))
        expected_trials = {
            int(row["multiseed_trials"])
            for row in rows
            if row.get("multiseed_trials") is not None
        }
        if len(expected_trials) != 1 or len(rows) != next(
            iter(expected_trials)
        ):
            raise ValueError(f"Incomplete metadata group: {group_id}")
        families = {
            str(row.get("attack_family", "clean")) for row in rows
        }
        if len(families) != 1:
            raise ValueError(f"Inconsistent attack family: {group_id}")
        output[group_id] = {
            "group_id": group_id,
            "source_kind": source_kind,
            "task_key": next(iter(task_keys)),
            "attack_family": next(iter(families)),
            "trajectory_ids": sorted(str(row["trajectory_id"]) for row in rows),
            "trials": len(rows),
        }
    return output


def _balanced_task_order(
    groups: list[dict[str, Any]], *, seed: int
) -> list[str]:
    if not groups:
        return []
    task_keys = {str(row["task_key"]) for row in groups}
    if len(task_keys) != 1:
        raise ValueError("Balanced ordering expects one user task")
    task_key = next(iter(task_keys))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        by_family[str(row["attack_family"])].append(row)
    for family, values in by_family.items():
        values.sort(
            key=lambda row: _stable_key(
                seed, task_key, family, str(row["group_id"])
            )
        )
    family_order = sorted(
        by_family,
        key=lambda family: _stable_key(seed, task_key, family),
    )
    ordered = []
    offset = 0
    while len(ordered) < len(groups):
        added = False
        for family in family_order:
            values = by_family[family]
            if offset < len(values):
                ordered.append(str(values[offset]["group_id"]))
                added = True
        if not added:
            break
        offset += 1
    if len(ordered) != len(groups) or len(set(ordered)) != len(ordered):
        raise RuntimeError("Failed to construct a complete unique task order")
    return ordered


def _filter_by_ids(
    rows: list[dict[str, Any]],
    *,
    selected_groups: set[str] | None = None,
    selected_trajectories: set[str] | None = None,
) -> list[dict[str, Any]]:
    if (selected_groups is None) == (selected_trajectories is None):
        raise ValueError("Specify exactly one filtering id set")
    if selected_groups is not None:
        return [
            row
            for row in rows
            if str(row.get("multiseed_group_id")) in selected_groups
        ]
    return [
        row
        for row in rows
        if str(row.get("trajectory_id")) in selected_trajectories
    ]


def _write_subset(
    *,
    output_dir: Path,
    selected_attack_groups: set[str],
    clean_groups: set[str],
    group_info: dict[str, dict[str, Any]],
    source_rows: dict[str, list[dict[str, Any]]],
    fraction: float,
    selection_seed: int,
) -> dict[str, Any]:
    selected_groups = selected_attack_groups | clean_groups
    selected_trajectories = {
        trajectory_id
        for group_id in selected_groups
        for trajectory_id in group_info[group_id]["trajectory_ids"]
    }
    filtered = {
        "train_steps.jsonl": _filter_by_ids(
            source_rows["train_steps.jsonl"],
            selected_groups=selected_groups,
        ),
        "train_metadata.jsonl": _filter_by_ids(
            source_rows["train_metadata.jsonl"],
            selected_groups=selected_groups,
        ),
        "train_label_groups.jsonl": _filter_by_ids(
            source_rows["train_label_groups.jsonl"],
            selected_groups=selected_groups,
        ),
        "train_trajectories.jsonl": _filter_by_ids(
            source_rows["train_trajectories.jsonl"],
            selected_trajectories=selected_trajectories,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in filtered.items():
        write_jsonl(output_dir / name, rows)

    attack_info = [group_info[group_id] for group_id in selected_attack_groups]
    all_info = [group_info[group_id] for group_id in selected_groups]
    trajectories_by_group = Counter(
        str(row["multiseed_group_id"])
        for row in filtered["train_metadata.jsonl"]
    )
    incomplete = {
        group_id: {
            "found": trajectories_by_group[group_id],
            "expected": group_info[group_id]["trials"],
        }
        for group_id in selected_groups
        if trajectories_by_group[group_id] != group_info[group_id]["trials"]
    }
    if incomplete:
        raise ValueError(f"Subset contains incomplete groups: {incomplete}")
    trajectory_ids_from_trajectories = {
        str(row["trajectory_id"])
        for row in filtered["train_trajectories.jsonl"]
    }
    trajectory_ids_from_metadata = {
        str(row["trajectory_id"])
        for row in filtered["train_metadata.jsonl"]
    }
    if trajectory_ids_from_trajectories != trajectory_ids_from_metadata:
        raise ValueError("Filtered trajectory and metadata ids differ")

    manifest = {
        "fraction_of_attack_configurations": fraction,
        "selection_seed": selection_seed,
        "selection_is_label_blind": True,
        "selected_attack_group_ids": sorted(selected_attack_groups),
        "selected_clean_group_ids": sorted(clean_groups),
        "selected_groups": sorted(all_info, key=lambda row: row["group_id"]),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    audit = {
        "scope": "nested AgentDojo-v2 training-size ablation",
        "fraction_of_attack_configurations": fraction,
        "selection_seed": selection_seed,
        "selection_is_label_blind": True,
        "attack_group_count": len(selected_attack_groups),
        "clean_group_count": len(clean_groups),
        "total_group_count": len(selected_groups),
        "attack_trajectory_count": sum(row["trials"] for row in attack_info),
        "clean_trajectory_count": sum(
            group_info[group_id]["trials"] for group_id in clean_groups
        ),
        "total_trajectory_count": len(selected_trajectories),
        "step_count": len(filtered["train_steps.jsonl"]),
        "user_task_count": len({row["task_key"] for row in attack_info}),
        "attack_groups_by_task": dict(
            sorted(Counter(row["task_key"] for row in attack_info).items())
        ),
        "attack_groups_by_family": dict(
            sorted(
                Counter(row["attack_family"] for row in attack_info).items()
            )
        ),
        "all_groups_complete": not incomplete,
        "trajectory_metadata_ids_match": True,
        "files": {
            name: {"rows": len(rows), "sha256": _sha256(output_dir / name)}
            for name, rows in filtered.items()
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    checksum_paths = sorted(output_dir.glob("*.json*"))
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25, 0.5]
    )
    parser.add_argument("--selection-seed", type=int, default=20260715)
    args = parser.parse_args()
    fractions = sorted(set(args.fractions))
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        parser.error("--fractions must be unique values strictly between 0 and 1")

    names = (
        "train_steps.jsonl",
        "train_metadata.jsonl",
        "train_label_groups.jsonl",
        "train_trajectories.jsonl",
    )
    source_rows = {name: read_jsonl(args.source_root / name) for name in names}
    info = _group_info(source_rows["train_metadata.jsonl"])
    clean_groups = {
        group_id
        for group_id, row in info.items()
        if row["source_kind"] == "clean"
    }
    attack_groups = [
        row for row in info.values() if row["source_kind"] == "attack"
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attack_groups:
        by_task[row["task_key"]].append(row)
    task_orders = {
        task_key: _balanced_task_order(values, seed=args.selection_seed)
        for task_key, values in sorted(by_task.items())
    }

    subsets = {}
    selected_by_fraction: dict[float, set[str]] = {}
    for fraction in fractions:
        selected = set()
        for task_key, ordered in task_orders.items():
            target = int(round(len(ordered) * fraction))
            if target < 1:
                raise ValueError(
                    f"Fraction {fraction} selects no groups for {task_key}"
            )
            selected.update(ordered[:target])
        label = f"pct{int(round(fraction * 100))}"
        audit = _write_subset(
            output_dir=args.output_root / label,
            selected_attack_groups=selected,
            clean_groups=clean_groups,
            group_info=info,
            source_rows=source_rows,
            fraction=fraction,
            selection_seed=args.selection_seed,
        )
        selected_by_fraction[fraction] = selected
        subsets[label] = audit

    nested_checks = {}
    for left, right in zip(fractions, fractions[1:], strict=False):
        key = f"{left}_subset_of_{right}"
        nested_checks[key] = selected_by_fraction[left] <= selected_by_fraction[right]
        if not nested_checks[key]:
            raise RuntimeError(f"Nested-subset invariant failed: {key}")
    summary = {
        "scope": "nested label-blind AgentDojo-v2 size ablation",
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selection_seed": args.selection_seed,
        "full_attack_group_count": len(attack_groups),
        "full_clean_group_count": len(clean_groups),
        "full_user_task_count": len(by_task),
        "task_attack_group_counts": {
            key: len(values) for key, values in sorted(by_task.items())
        },
        "nested_checks": nested_checks,
        "subsets": subsets,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
