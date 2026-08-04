"""Build five grouped OOF folds and nested 25% AgentDojo-v2 training sets.

The 20 user tasks are partitioned into five four-task cohorts (one task per
domain).  Every cohort is test exactly once and validation exactly once.  Each
fold therefore preserves the existing 12/4/4 train/validation/test task counts.
Attack-configuration subsampling is deterministic, family-interleaved,
group-complete, and label-blind.
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


COHORT_ORDER = (
    "original_test",
    "original_val",
    "train0",
    "train1",
    "train2",
)
FOLD_COHORTS = (
    ("original_test", "original_val"),
    ("original_val", "train0"),
    ("train0", "train1"),
    ("train1", "train2"),
    ("train2", "original_test"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(
        "|".join([str(seed), *parts]).encode("utf-8")
    ).hexdigest()


def _task_key(row: dict[str, Any]) -> str:
    domain = row.get("suite", row.get("domain"))
    task = row.get("user_task_id", row.get("task_id"))
    if domain is None or task is None:
        raise ValueError("Row lacks domain/task identity")
    return f"{domain}|{task}"


def _group_info(metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        group_id = row.get("multiseed_group_id")
        if not group_id:
            raise ValueError("Metadata row lacks multiseed_group_id")
        grouped[str(group_id)].append(row)
    output = {}
    for group_id, rows in grouped.items():
        task_keys = {_task_key(row) for row in rows}
        source_kinds = {str(row["source_kind"]) for row in rows}
        if len(task_keys) != 1 or len(source_kinds) != 1:
            raise ValueError(f"Inconsistent group metadata: {group_id}")
        expected = {
            int(row["multiseed_trials"])
            for row in rows
            if row.get("multiseed_trials") is not None
        }
        if len(expected) != 1 or len(rows) != next(iter(expected)):
            raise ValueError(f"Incomplete trajectory group: {group_id}")
        families = {str(row.get("attack_family", "clean")) for row in rows}
        original_splits = {str(row["task_split"]) for row in rows}
        if len(families) != 1 or len(original_splits) != 1:
            raise ValueError(f"Inconsistent group annotations: {group_id}")
        output[group_id] = {
            "group_id": group_id,
            "task_key": next(iter(task_keys)),
            "source_kind": next(iter(source_kinds)),
            "attack_family": next(iter(families)),
            "original_split": next(iter(original_splits)),
            "trials": len(rows),
        }
    return output


def _cohorts(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    task_splits: dict[str, str] = {}
    by_domain_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in metadata:
        task_key = _task_key(row)
        split = str(row["task_split"])
        previous = task_splits.setdefault(task_key, split)
        if previous != split:
            raise ValueError(f"Task spans original splits: {task_key}")
        domain, _ = task_key.split("|", 1)
        by_domain_split[domain][split].add(task_key)

    cohorts = {name: [] for name in COHORT_ORDER}
    for domain in sorted(by_domain_split):
        split_tasks = by_domain_split[domain]
        if {key: len(value) for key, value in split_tasks.items()} != {
            "train": 3,
            "val": 1,
            "test": 1,
        }:
            raise ValueError(
                f"Unexpected original task counts for {domain}: "
                f"{dict((key, len(value)) for key, value in split_tasks.items())}"
            )
        cohorts["original_test"].append(next(iter(split_tasks["test"])))
        cohorts["original_val"].append(next(iter(split_tasks["val"])))
        for index, task_key in enumerate(sorted(split_tasks["train"])):
            cohorts[f"train{index}"].append(task_key)
    for name, tasks in cohorts.items():
        if len(tasks) != 4 or len({task.split("|", 1)[0] for task in tasks}) != 4:
            raise ValueError(f"Cohort is not one task per domain: {name}")
        cohorts[name] = sorted(tasks)
    if len(set().union(*(set(tasks) for tasks in cohorts.values()))) != 20:
        raise ValueError("Cohorts do not partition all 20 user tasks")
    return cohorts


def _balanced_task_order(
    groups: list[dict[str, Any]], *, seed: int
) -> list[str]:
    task_keys = {str(row["task_key"]) for row in groups}
    if len(task_keys) != 1:
        raise ValueError("Balanced ordering expects one task")
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
        by_family, key=lambda family: _stable_key(seed, task_key, family)
    )
    ordered = []
    offset = 0
    while len(ordered) < len(groups):
        added = False
        for family in family_order:
            if offset < len(by_family[family]):
                ordered.append(str(by_family[family][offset]["group_id"]))
                added = True
        if not added:
            break
        offset += 1
    if len(ordered) != len(groups) or len(set(ordered)) != len(ordered):
        raise RuntimeError("Failed to build complete group order")
    return ordered


def _filter_tasks(
    rows: list[dict[str, Any]], tasks: set[str]
) -> list[dict[str, Any]]:
    return [row for row in rows if _task_key(row) in tasks]


def _filter_groups(
    rows: list[dict[str, Any]], groups: set[str]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("multiseed_group_id")) in groups
    ]


def _split_audit(
    metadata: list[dict[str, Any]], group_info: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    group_ids = {str(row["multiseed_group_id"]) for row in metadata}
    groups = [group_info[group_id] for group_id in group_ids]
    attack = [row for row in groups if row["source_kind"] == "attack"]
    clean = [row for row in groups if row["source_kind"] == "clean"]
    return {
        "user_tasks": len({row["task_key"] for row in groups}),
        "attack_groups": len(attack),
        "clean_groups": len(clean),
        "attack_trajectories": sum(row["trials"] for row in attack),
        "clean_trajectories": sum(row["trials"] for row in clean),
        "attack_groups_by_family": dict(
            sorted(Counter(row["attack_family"] for row in attack).items())
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260715)
    parser.add_argument("--reference-pct25-root", type=Path)
    args = parser.parse_args()

    steps = read_jsonl(args.source_root / "steps.jsonl")
    metadata = read_jsonl(args.source_root / "metadata.jsonl")
    groups = _group_info(metadata)
    cohorts = _cohorts(metadata)
    all_tasks = set().union(*(set(values) for values in cohorts.values()))
    if {_task_key(row) for row in steps} != all_tasks:
        raise ValueError("Step tasks and metadata cohorts differ")

    task_attack_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_clean_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, row in groups.items():
        if row["source_kind"] == "attack":
            task_attack_groups[row["task_key"]].append(row)
        else:
            task_clean_groups[row["task_key"]].add(group_id)
    for task_key in all_tasks:
        if len(task_attack_groups[task_key]) != 20:
            raise ValueError(f"Expected 20 attack groups for {task_key}")
        if len(task_clean_groups[task_key]) != 1:
            raise ValueError(f"Expected one clean group for {task_key}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    fold_summaries = {}
    test_task_counts: Counter[str] = Counter()
    val_task_counts: Counter[str] = Counter()
    for fold, (test_cohort, val_cohort) in enumerate(FOLD_COHORTS):
        test_tasks = set(cohorts[test_cohort])
        val_tasks = set(cohorts[val_cohort])
        train_tasks = all_tasks - test_tasks - val_tasks
        if not (
            len(train_tasks) == 12
            and len(val_tasks) == 4
            and len(test_tasks) == 4
            and not (train_tasks & val_tasks)
            and not (train_tasks & test_tasks)
            and not (val_tasks & test_tasks)
        ):
            raise RuntimeError(f"Invalid grouped split for fold {fold}")
        test_task_counts.update(test_tasks)
        val_task_counts.update(val_tasks)

        fold_root = args.output_root / f"fold{fold}"
        full_root = fold_root / "full"
        pct25_root = fold_root / "pct25"
        full_metadata = {}
        for split, split_tasks in (
            ("train", train_tasks),
            ("val", val_tasks),
            ("test", test_tasks),
        ):
            split_steps = _filter_tasks(steps, split_tasks)
            split_metadata = _filter_tasks(metadata, split_tasks)
            write_jsonl(full_root / f"{split}_steps.jsonl", split_steps)
            write_jsonl(
                full_root / f"{split}_metadata.jsonl", split_metadata
            )
            full_metadata[split] = split_metadata

        selected_attack_groups = set()
        task_orders = {}
        for task_key in sorted(train_tasks):
            ordered = _balanced_task_order(
                task_attack_groups[task_key], seed=args.selection_seed
            )
            selected_attack_groups.update(ordered[:5])
            task_orders[task_key] = ordered
        selected_clean_groups = set().union(
            *(task_clean_groups[task_key] for task_key in train_tasks)
        )
        selected_groups = selected_attack_groups | selected_clean_groups
        pct25_steps = _filter_groups(
            read_jsonl(full_root / "train_steps.jsonl"), selected_groups
        )
        pct25_metadata = _filter_groups(
            full_metadata["train"], selected_groups
        )
        write_jsonl(pct25_root / "train_steps.jsonl", pct25_steps)
        write_jsonl(pct25_root / "train_metadata.jsonl", pct25_metadata)

        audit = {
            "fold": fold,
            "test_cohort": test_cohort,
            "validation_cohort": val_cohort,
            "train_tasks": sorted(train_tasks),
            "validation_tasks": sorted(val_tasks),
            "test_tasks": sorted(test_tasks),
            "selection_seed": args.selection_seed,
            "selection_is_label_blind": True,
            "full": {
                split: {
                    **_split_audit(full_metadata[split], groups),
                    "steps": sum(
                        1
                        for _ in (full_root / f"{split}_steps.jsonl").open(
                            encoding="utf-8"
                        )
                    ),
                    "steps_sha256": _sha256(
                        full_root / f"{split}_steps.jsonl"
                    ),
                }
                for split in ("train", "val", "test")
            },
            "pct25": {
                **_split_audit(pct25_metadata, groups),
                "steps": len(pct25_steps),
                "steps_sha256": _sha256(pct25_root / "train_steps.jsonl"),
                "selected_attack_group_ids": sorted(selected_attack_groups),
                "selected_clean_group_ids": sorted(selected_clean_groups),
            },
        }
        if audit["full"]["train"]["attack_groups"] != 240:
            raise RuntimeError(f"Fold {fold} full train is not 240 groups")
        if audit["pct25"]["attack_groups"] != 60:
            raise RuntimeError(f"Fold {fold} pct25 train is not 60 groups")
        if audit["full"]["val"]["attack_groups"] != 80:
            raise RuntimeError(f"Fold {fold} val is not 80 groups")
        if audit["full"]["test"]["attack_groups"] != 80:
            raise RuntimeError(f"Fold {fold} test is not 80 groups")
        _write_json(fold_root / "audit.json", audit)
        (fold_root / "checksums.sha256").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(fold_root).as_posix()}\n"
                for path in sorted(fold_root.rglob("*.json*"))
                if path.name != "checksums.sha256"
            ),
            encoding="utf-8",
        )
        fold_summaries[f"fold{fold}"] = audit

    if set(test_task_counts.values()) != {1} or set(test_task_counts) != all_tasks:
        raise RuntimeError("Every task must be OOF test exactly once")
    if set(val_task_counts.values()) != {1} or set(val_task_counts) != all_tasks:
        raise RuntimeError("Every task must be validation exactly once")

    fold0_equivalence = {
        "task_split_matches_original": (
            fold_summaries["fold0"]["test_cohort"] == "original_test"
            and fold_summaries["fold0"]["validation_cohort"] == "original_val"
        ),
        "full_train_sha256_matches": (
            fold_summaries["fold0"]["full"]["train"]["steps_sha256"]
            == _sha256(args.source_root / "train_steps.jsonl")
        ),
        "full_val_sha256_matches": (
            fold_summaries["fold0"]["full"]["val"]["steps_sha256"]
            == _sha256(args.source_root / "val_steps.jsonl")
        ),
        "full_test_sha256_matches": (
            fold_summaries["fold0"]["full"]["test"]["steps_sha256"]
            == _sha256(args.source_root / "test_steps.jsonl")
        ),
    }
    if args.reference_pct25_root is not None:
        fold0_equivalence["pct25_train_sha256_matches"] = (
            fold_summaries["fold0"]["pct25"]["steps_sha256"]
            == _sha256(args.reference_pct25_root / "train_steps.jsonl")
        )
    if not all(fold0_equivalence.values()):
        raise RuntimeError(
            f"Fold0 cannot safely reuse existing checkpoints: {fold0_equivalence}"
        )

    summary = {
        "scope": "20-task grouped five-fold OOF AgentDojo-v2",
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selection_seed": args.selection_seed,
        "cohort_order": list(COHORT_ORDER),
        "cohorts": cohorts,
        "fold_cohorts": [
            {
                "fold": fold,
                "test": test,
                "validation": validation,
            }
            for fold, (test, validation) in enumerate(FOLD_COHORTS)
        ],
        "all_20_tasks_test_exactly_once": True,
        "all_20_tasks_validation_exactly_once": True,
        "fold0_equivalence": fold0_equivalence,
        "folds": fold_summaries,
    }
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
