"""Normalize AgentDojo-v2 runs, attach sidecar metadata, and audit splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.agentdojo_v2 import V2_SCOPE
from wmagentattack.io_utils import write_jsonl
from wmagentattack.normalize_agentdojo import normalize_trace


def _load_result_files(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("scope") == V2_SCOPE and isinstance(payload.get("results"), list):
            payloads.append((path, payload))
    return payloads


def _set_dataset_name(trajectory, dataset_name: str) -> None:
    trajectory.dataset = dataset_name
    for step in trajectory.steps:
        step.dataset = dataset_name


def _seed_from_text(text: str) -> int | None:
    matches = re.findall(r"seed[-_]?([0-9]+)", text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def _clean_paths(roots: list[Path], selected_tasks: set[tuple[str, str]]) -> list[Path]:
    output = []
    for root in roots:
        for path in sorted(root.rglob("none/none.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = (str(raw.get("suite_name")), str(raw.get("user_task_id")))
            if key in selected_tasks and raw.get("attack_type") in (None, "none"):
                output.append(path)
    return output


def _metadata_without_payload(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {"payload", "payload_by_vector", "payload_segments"}
    return {key: value for key, value in row.items() if key not in excluded}


def _split_audit(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = [row for row in metadata if row["task_split"] == split]
        attacked = [row for row in rows if row["source_kind"] == "attack"]
        completed = [row for row in attacked if row.get("status") == "completed"]
        payload[split] = {
            "trajectories": len(rows),
            "clean_trajectories": sum(row["source_kind"] == "clean" for row in rows),
            "attack_trajectories": len(attacked),
            "attack_success": sum(bool(row.get("security")) for row in completed),
            "task_success_under_attack": sum(bool(row.get("utility")) for row in completed),
            "ASR": (
                sum(bool(row.get("security")) for row in completed) / len(completed)
                if completed
                else 0.0
            ),
            "BUP": (
                sum(bool(row.get("utility")) for row in completed) / len(completed)
                if completed
                else 0.0
            ),
            "families": dict(Counter(row["attack_family"] for row in attacked)),
            "domains": dict(Counter(row["suite"] for row in rows)),
            "user_tasks": len({(row["suite"], row["user_task_id"]) for row in rows}),
        }
    task_sets = {
        split: {
            (row["suite"], row["user_task_id"])
            for row in metadata
            if row["task_split"] == split
        }
        for split in ("train", "val", "test")
    }
    payload["task_overlap"] = {
        "train_val": len(task_sets["train"] & task_sets["val"]),
        "train_test": len(task_sets["train"] & task_sets["test"]),
        "val_test": len(task_sets["val"] & task_sets["test"]),
    }
    family_sets = {
        split: {
            row["attack_family"]
            for row in metadata
            if row["task_split"] == split and row["source_kind"] == "attack"
        }
        for split in ("train", "val", "test")
    }
    payload["family_sets"] = {key: sorted(value) for key, value in family_sets.items()}
    payload["exclusive_families"] = {
        "train": sorted(family_sets["train"] - family_sets["val"] - family_sets["test"]),
        "val": sorted(family_sets["val"] - family_sets["train"] - family_sets["test"]),
        "test": sorted(family_sets["test"] - family_sets["train"] - family_sets["val"]),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--clean-run-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "agentdojo_v2" / "screen_dataset",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("scope") != V2_SCOPE:
        raise ValueError("Not an AgentDojo-v2 sandbox manifest")
    manifest_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    task_splits: dict[tuple[str, str], str] = {}
    for row in manifest["rows"]:
        key = (str(row["suite"]), str(row["user_task_id"]))
        previous = task_splits.setdefault(key, str(row["task_split"]))
        if previous != row["task_split"]:
            raise ValueError(f"Task split conflict in manifest: {key}")

    result_files = _load_result_files(args.result_root)
    if not result_files:
        raise RuntimeError(f"No AgentDojo-v2 result files under {args.result_root}")
    run_hashes = {payload.get("manifest_sha256") for _, payload in result_files}
    if len(run_hashes) != 1:
        raise ValueError(f"Mixed manifest hashes in result root: {run_hashes}")

    trajectories = []
    metadata: list[dict[str, Any]] = []
    seen_episode_keys: set[tuple[int, str]] = set()
    failed_rows = []
    completed_manifest_ids_by_seed: dict[int, set[str]] = defaultdict(set)
    for result_path, payload in result_files:
        run_seed = int(payload["run_seed"])
        for result in payload["results"]:
            row_id = str(result["row_id"])
            if row_id not in manifest_rows:
                raise KeyError(f"Result references unknown manifest row: {row_id}")
            key = (run_seed, row_id)
            if key in seen_episode_keys:
                raise ValueError(f"Duplicate result for seed/row: {key}")
            seen_episode_keys.add(key)
            if result.get("status") != "completed":
                failed_rows.append({**result, "result_file": str(result_path)})
                continue
            raw_path = Path(result["raw_trace"])
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            manifest_row = manifest_rows[row_id]
            expected = {
                "suite_name": manifest_row["suite"],
                "user_task_id": manifest_row["user_task_id"],
                "injection_task_id": manifest_row["injection_task_id"],
                "attack_type": result["attack_name"],
            }
            mismatches = {
                name: (raw.get(name), value)
                for name, value in expected.items()
                if raw.get(name) != value
            }
            if mismatches:
                raise ValueError(f"Raw trace/result mismatch for {row_id}: {mismatches}")
            trajectory = normalize_trace(raw_path)
            _set_dataset_name(trajectory, f"agentdojo-{manifest['dataset_version']}")
            trajectories.append(trajectory)
            metadata.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "source_kind": "attack",
                    "run_seed": run_seed,
                    "episode_seed": result.get("episode_seed"),
                    "status": "completed",
                    "utility": bool(result["utility"]),
                    "security": bool(result["security"]),
                    "raw_trace": str(raw_path.resolve()),
                    "result_file": str(result_path.resolve()),
                    **_metadata_without_payload(manifest_row),
                }
            )
            completed_manifest_ids_by_seed[run_seed].add(row_id)

    selected_tasks = set(task_splits)
    for raw_path in _clean_paths(args.clean_run_root, selected_tasks):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        key = (str(raw["suite_name"]), str(raw["user_task_id"]))
        trajectory = normalize_trace(raw_path)
        _set_dataset_name(trajectory, f"agentdojo-{manifest['dataset_version']}")
        if any(item.trajectory_id == trajectory.trajectory_id for item in trajectories):
            continue
        trajectories.append(trajectory)
        metadata.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "source_kind": "clean",
                "run_seed": _seed_from_text(str(raw_path))
                or _seed_from_text(str(raw.get("pipeline_name", ""))),
                "episode_seed": None,
                "status": "completed",
                "utility": bool(raw.get("utility")),
                "security": False,
                "raw_trace": str(raw_path.resolve()),
                "result_file": None,
                "row_id": None,
                "scope": V2_SCOPE,
                "suite": key[0],
                "task_split": task_splits[key],
                "user_task_id": key[1],
                "injection_task_id": None,
                "attack_name": "none",
                "attack_family": "clean",
                "attack_variant": "clean_multiseed_reuse",
                "attack_role": "clean",
                "payload_sha256": None,
                "base_success_rate": next(
                    (
                        row.get("base_success_rate")
                        for row in manifest["rows"]
                        if row["suite"] == key[0] and row["user_task_id"] == key[1]
                    ),
                    None,
                ),
            }
        )

    if len({row["trajectory_id"] for row in metadata}) != len(metadata):
        raise ValueError("Normalized trajectory IDs are not unique")
    if args.require_complete:
        if failed_rows:
            raise RuntimeError(f"There are {len(failed_rows)} failed v2 episodes")
        expected = set(manifest_rows)
        incomplete = {
            seed: sorted(expected - completed)
            for seed, completed in completed_manifest_ids_by_seed.items()
            if completed != expected
        }
        if incomplete:
            raise RuntimeError(
                "Incomplete v2 result seeds: "
                + ", ".join(f"{seed}={len(rows)} missing" for seed, rows in incomplete.items())
            )

    by_trajectory = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    metadata_by_id = {row["trajectory_id"]: row for row in metadata}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(trajectories, key=lambda item: item.trajectory_id)
    ordered_metadata = [metadata_by_id[item.trajectory_id] for item in ordered]
    write_jsonl(args.out_dir / "trajectories.jsonl", ordered)
    write_jsonl(
        args.out_dir / "steps.jsonl",
        [step for trajectory in ordered for step in trajectory.steps],
    )
    write_jsonl(args.out_dir / "metadata.jsonl", ordered_metadata)

    for split in ("train", "val", "test"):
        split_metadata = [row for row in ordered_metadata if row["task_split"] == split]
        split_trajectories = [by_trajectory[row["trajectory_id"]] for row in split_metadata]
        write_jsonl(args.out_dir / f"{split}_metadata.jsonl", split_metadata)
        write_jsonl(args.out_dir / f"{split}_trajectories.jsonl", split_trajectories)
        write_jsonl(
            args.out_dir / f"{split}_steps.jsonl",
            [step for trajectory in split_trajectories for step in trajectory.steps],
        )

    audit = {
        "scope": V2_SCOPE,
        "dataset_version": manifest["dataset_version"],
        "manifest": str(args.manifest.resolve()),
        "result_root": str(args.result_root.resolve()),
        "clean_run_roots": [str(path.resolve()) for path in args.clean_run_root],
        "result_files": len(result_files),
        "run_seeds": sorted(completed_manifest_ids_by_seed),
        "manifest_rows": len(manifest_rows),
        "attack_trajectories": sum(row["source_kind"] == "attack" for row in metadata),
        "clean_trajectories": sum(row["source_kind"] == "clean" for row in metadata),
        "total_trajectories": len(trajectories),
        "total_steps": sum(len(trajectory.steps) for trajectory in trajectories),
        "failed_episodes": len(failed_rows),
        "failed_rows": failed_rows,
        "split_audit": _split_audit(metadata),
        "safety_contract": manifest["safety_contract"],
        "checks": {
            "task_overlap_zero": all(
                value == 0
                for value in _split_audit(metadata)["task_overlap"].values()
            ),
            "test_family_unseen_in_train": "structured_holdout"
            not in {
                row["attack_family"]
                for row in metadata
                if row["task_split"] == "train"
            },
            "autodojo_transfer_not_in_val_or_test": not any(
                row["attack_family"] == "blackbox_adaptive_transfer"
                and row["task_split"] != "train"
                for row in metadata
            ),
            "all_raw_traces_present": all(Path(row["raw_trace"]).exists() for row in metadata),
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError(f"AgentDojo-v2 audit failed: {audit['checks']}")
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
