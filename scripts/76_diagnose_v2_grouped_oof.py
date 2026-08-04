"""Diagnose selection stability and attack-family effects in grouped OOF runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FOLDS = tuple(range(5))
SIZES = ("pct25", "pct100")
BUDGETS = (1, 2, 4)
OUTCOME_METRICS = ("ASR", "BUP", "ASR_plus_BUP")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _configuration_outcomes(root: Path) -> dict[str, dict[str, Any]]:
    first_steps: dict[tuple[str, str], dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        path = root / "data" / f"fold{fold}" / "full" / "test_steps.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                step = json.loads(line)
                group_id = step.get("multiseed_group_id")
                if group_id is None or step.get("attack_probability_target") is None:
                    continue
                group_id = str(group_id)
                key = (group_id, str(step["trajectory_id"]))
                previous = first_steps.get(key)
                if previous is None or int(step["step_id"]) < int(previous["step_id"]):
                    first_steps[key] = step
                metadata.setdefault(
                    group_id,
                    {
                        "task_key": f"{step['domain']}|{step['task_id']}",
                        "domain": str(step["domain"]),
                        "attack_action": step.get("attack_action"),
                        "attack_location": step.get("attack_location"),
                        "variant": (
                            group_id.rsplit("__", 2)[-2]
                            if group_id.startswith("attack::")
                            else "clean"
                        ),
                    },
                )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (group_id, _), step in first_steps.items():
        grouped[group_id].append(step)
    output = {}
    for group_id, rows in grouped.items():
        row = dict(metadata[group_id])
        row.update(
            {
                "ASR": statistics.fmean(bool(item["attack_success"]) for item in rows),
                "BUP": statistics.fmean(bool(item["task_success"]) for item in rows),
                "trials": len(rows),
            }
        )
        row["ASR_plus_BUP"] = row["ASR"] + row["BUP"]
        output[group_id] = row
    if len(output) != 400:
        raise ValueError(f"Expected 400 OOF attack configurations, found {len(output)}")
    return output


def _ids_by_task(
    group_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for group_id in group_ids:
        output[outcomes[group_id]["task_key"]].add(group_id)
    return dict(output)


def _profile(
    group_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    def summarize(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group_id in group_ids:
            grouped[str(outcomes[group_id][field])].append(outcomes[group_id])
        return {
            key: {
                "configuration_count": len(rows),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in rows)
                    for metric in OUTCOME_METRICS
                },
            }
            for key, rows in sorted(grouped.items())
        }

    return {
        "configuration_count": len(group_ids),
        "by_variant": summarize("variant"),
        "by_attack_action": summarize("attack_action"),
    }


def _selection_stability(
    root: Path,
    *,
    size: str,
    budget: int,
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairwise_jaccards: list[float] = []
    ensemble_seed_jaccards: list[float] = []
    unanimous = 0
    ensemble_matches_any = 0
    seed_majority = 0
    all_different = 0
    ensemble_matches_majority = 0
    task_count = 0
    for fold in FOLDS:
        result = _load(
            root / "prospective" / f"fold{fold}" / size / "result.json"
        )["test"][str(budget)]["calibrated"]
        ensemble = _ids_by_task(result["ensemble"]["selected_group_ids"], outcomes)
        seeds = [
            _ids_by_task(row["selected_group_ids"], outcomes)
            for row in result["per_seed"].values()
        ]
        for task, ensemble_set in ensemble.items():
            seed_sets = [seed[task] for seed in seeds]
            task_count += 1
            pairwise_jaccards.extend(
                _jaccard(seed_sets[left], seed_sets[right])
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            ensemble_seed_jaccards.extend(
                _jaccard(ensemble_set, seed_set) for seed_set in seed_sets
            )
            unanimous += int(seed_sets[0] == seed_sets[1] == seed_sets[2])
            ensemble_matches_any += int(ensemble_set in seed_sets)
            if budget == 1:
                ids = [next(iter(seed_set)) for seed_set in seed_sets]
                counts = Counter(ids)
                majority_id, majority_count = counts.most_common(1)[0]
                if majority_count >= 2:
                    seed_majority += 1
                    ensemble_matches_majority += int(
                        next(iter(ensemble_set)) == majority_id
                    )
                else:
                    all_different += 1
    output = {
        "task_count": task_count,
        "pairwise_seed_comparison_count": len(pairwise_jaccards),
        "mean_pairwise_seed_jaccard": statistics.fmean(pairwise_jaccards),
        "mean_ensemble_to_seed_jaccard": statistics.fmean(ensemble_seed_jaccards),
        "unanimous_seed_set_task_count": unanimous,
        "ensemble_matches_any_seed_set_task_count": ensemble_matches_any,
    }
    if budget == 1:
        output.update(
            {
                "seed_majority_task_count": seed_majority,
                "all_three_seed_ids_different_task_count": all_different,
                "ensemble_matches_seed_majority_task_count": ensemble_matches_majority,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.archive_root / "final_summary.json"
    summary = _load(summary_path)
    outcomes = _configuration_outcomes(args.archive_root)
    selected = summary["prospective_oof"]

    cross_scale = {}
    stability = {size: {} for size in SIZES}
    selected_profiles = {size: {} for size in SIZES}
    for budget in BUDGETS:
        by_size = {
            size: _ids_by_task(
                selected[size][str(budget)]["selected_group_ids"], outcomes
            )
            for size in SIZES
        }
        tasks = sorted(by_size["pct25"])
        task_jaccards = {
            task: _jaccard(by_size["pct25"][task], by_size["pct100"][task])
            for task in tasks
        }
        cross_scale[str(budget)] = {
            "task_count": len(tasks),
            "mean_task_jaccard": statistics.fmean(task_jaccards.values()),
            "exact_task_set_match_count": sum(value == 1.0 for value in task_jaccards.values()),
            "zero_overlap_task_count": sum(value == 0.0 for value in task_jaccards.values()),
            "per_task_jaccard": task_jaccards,
        }
        for size in SIZES:
            stability[size][str(budget)] = _selection_stability(
                args.archive_root,
                size=size,
                budget=budget,
                outcomes=outcomes,
            )
            selected_profiles[size][str(budget)] = _profile(
                selected[size][str(budget)]["selected_group_ids"], outcomes
            )

    primary_domain = {}
    left = selected["pct25"]["1"]["by_task"]
    right = selected["pct100"]["1"]["by_task"]
    for domain in sorted({task.split("|", 1)[0] for task in left}):
        tasks = [task for task in left if task.startswith(f"{domain}|")]
        primary_domain[domain] = {
            "task_count": len(tasks),
            **{
                metric: {
                    "pct25": statistics.fmean(float(left[task][metric]) for task in tasks),
                    "pct100": statistics.fmean(float(right[task][metric]) for task in tasks),
                    "pct100_minus_pct25": statistics.fmean(
                        float(right[task][metric]) - float(left[task][metric])
                        for task in tasks
                    ),
                }
                for metric in OUTCOME_METRICS
            },
        }

    payload = {
        "scope": "post-hoc diagnostics for 20-task grouped OOF scale test",
        "source_summary": str(summary_path),
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "candidate_pool_profile": _profile(list(outcomes), outcomes),
        "selected_profiles": selected_profiles,
        "cross_scale_selection_overlap": cross_scale,
        "within_size_seed_selection_stability": stability,
        "primary_top1_domain_effects": primary_domain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
