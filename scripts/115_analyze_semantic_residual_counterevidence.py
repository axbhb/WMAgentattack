"""Post-hoc counterevidence audit for the semantic-residual round.

This script does not retrain or tune the model.  It pairs each free-running
prediction with the frozen candidate-aware Markov baseline and reports
task-cluster bootstrap intervals plus subgroup failures.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_map(path: Path) -> dict[str, dict]:
    rows = _read_jsonl(path)
    mapping = {row["trajectory_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate trajectory prediction in {path}")
    return mapping


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit(truth: list[str], prediction: list[str]) -> float:
    return _levenshtein(truth, prediction) / max(len(truth), len(prediction), 1)


def _length_bin(length: int) -> str:
    if length == 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 8:
        return "5-8"
    return "9-16"


class CandidateMarkov:
    def __init__(self, skill_names: list[str]) -> None:
        self.skill_names = skill_names
        self.skill_vocab = {name: index for index, name in enumerate(skill_names)}
        self.num_skills = len(skill_names)
        self.global_counts = [0.0] * self.num_skills
        self.domain_counts: dict[str, list[float]] = defaultdict(
            lambda: [0.0] * self.num_skills
        )
        self.context_counts: dict[tuple[str, int], list[float]] = defaultdict(
            lambda: [0.0] * self.num_skills
        )

    def fit(self, trajectories: list[dict]) -> None:
        for trajectory in trajectories:
            previous = self.skill_vocab[BOS]
            for step in trajectory["steps"]:
                target = self.skill_vocab.get(step["selected_skill"], self.skill_vocab[UNK])
                self.global_counts[target] += 1.0
                self.domain_counts[trajectory["domain"]][target] += 1.0
                self.context_counts[(trajectory["domain"], previous)][target] += 1.0
                previous = target

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        total = sum(values)
        return [value / total for value in values]

    def probabilities(
        self, domain: str, previous: int, allowed: set[int]
    ) -> list[float]:
        global_probability = self._normalize(
            [value + 0.5 for value in self.global_counts]
        )
        if domain in self.domain_counts:
            domain_probability = self._normalize(
                [
                    count + 5.0 * prior
                    for count, prior in zip(
                        self.domain_counts[domain], global_probability
                    )
                ]
            )
        else:
            domain_probability = global_probability
        context = (domain, previous)
        if context in self.context_counts:
            probability = self._normalize(
                [
                    count + 3.0 * prior
                    for count, prior in zip(
                        self.context_counts[context], domain_probability
                    )
                ]
            )
        else:
            probability = domain_probability
        restricted = [
            value if index in allowed else 0.0
            for index, value in enumerate(probability)
        ]
        if sum(restricted) <= 0:
            restricted = [float(index in allowed) for index in range(self.num_skills)]
        return self._normalize(restricted)

    def generate(self, trajectory: dict, max_steps: int = 16) -> list[str]:
        allowed = {
            self.skill_vocab.get(name, self.skill_vocab[UNK])
            for name in trajectory["steps"][0]["candidate_skills"]
        }
        previous = self.skill_vocab[BOS]
        finish = self.skill_vocab.get("finish", -1)
        generated = []
        for _ in range(max_steps):
            probability = self.probabilities(trajectory["domain"], previous, allowed)
            selected = max(range(self.num_skills), key=probability.__getitem__)
            generated.append(self.skill_names[selected])
            previous = selected
            if selected == finish:
                break
        return generated


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int,
    draws: int = 10_000,
) -> dict:
    """Bootstrap task-level mean paired differences."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["task_group"]].append(float(row[metric]))
    task_means = [statistics.fmean(values) for values in grouped.values()]
    if not task_means:
        raise ValueError("cluster bootstrap needs at least one task")
    point = statistics.fmean(task_means)
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(
            statistics.fmean(generator.choice(task_means) for _ in task_means)
        )
    samples.sort()
    return {
        "task_count": len(task_means),
        "trajectory_count": len(rows),
        "task_equal_weight_point": point,
        "ci95": [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"trajectory_count": 0, "task_count": 0}
    return {
        "trajectory_count": len(rows),
        "task_count": len({row["task_group"] for row in rows}),
        "model_exact": statistics.fmean(row["model_exact"] for row in rows),
        "markov_exact": statistics.fmean(row["markov_exact"] for row in rows),
        "model_minus_markov_exact": statistics.fmean(
            row["delta_exact"] for row in rows
        ),
        "model_edit": statistics.fmean(row["model_edit"] for row in rows),
        "markov_edit": statistics.fmean(row["markov_edit"] for row in rows),
        "model_minus_markov_edit": statistics.fmean(
            row["delta_edit"] for row in rows
        ),
        "all_seed_support_coverage": statistics.fmean(
            row["all_seed_supported"] for row in rows
        ),
        "mean_seed_support_coverage": statistics.fmean(
            row["seed_support_fraction"] for row in rows
        ),
    }


def _group_summary(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: _summarize(group) for name, group in sorted(groups.items())}


def _split_rows(
    split: str,
    trajectories: list[dict],
    metadata: dict[str, dict],
    predictions: list[dict[str, dict]],
    markov: CandidateMarkov,
) -> list[dict]:
    trajectory_map = {row["trajectory_id"]: row for row in trajectories}
    expected = set(trajectory_map)
    if any(set(seed_rows) != expected for seed_rows in predictions):
        raise ValueError(f"{split} prediction IDs do not match trajectory data")
    rows = []
    for trajectory_id, trajectory in trajectory_map.items():
        truth = [step["selected_skill"] for step in trajectory["steps"]]
        markov_path = markov.generate(trajectory)
        seed_records = [seed_rows[trajectory_id] for seed_rows in predictions]
        if any(record["true_skill_path"] != truth for record in seed_records):
            raise ValueError(f"saved true path mismatch for {trajectory_id}")
        model_exact_values = [
            float(record["generated_skill_path"] == truth) for record in seed_records
        ]
        model_edit_values = [
            _normalized_edit(truth, record["generated_skill_path"])
            for record in seed_records
        ]
        model_exact = statistics.fmean(model_exact_values)
        model_edit = statistics.fmean(model_edit_values)
        markov_exact = float(markov_path == truth)
        markov_edit = _normalized_edit(truth, markov_path)
        support = [
            not record["conservative_rollout_would_truncate"]
            for record in seed_records
        ]
        context_unknown = [record["context_has_unknown"] for record in seed_records]
        if len(set(context_unknown)) != 1:
            raise ValueError("context unknown flag changed across seeds")
        meta = metadata[trajectory_id]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "task_group": f"{trajectory['domain']}|{trajectory['task_id']}",
                "domain": trajectory["domain"],
                "attack_family": str(meta.get("attack_family", "unknown")),
                "context_support": "unknown" if context_unknown[0] else "known",
                "length_bin": _length_bin(len(truth)),
                "true_length": len(truth),
                "model_exact": model_exact,
                "markov_exact": markov_exact,
                "delta_exact": model_exact - markov_exact,
                "model_edit": model_edit,
                "markov_edit": markov_edit,
                "delta_edit": model_edit - markov_edit,
                "all_seed_supported": float(all(support)),
                "seed_support_fraction": statistics.fmean(support),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    previous = json.loads(args.previous_summary.read_text(encoding="utf-8"))
    seed_dirs = [args.root / "model" / f"seed{seed}" for seed in (7, 17, 29)]
    seed_reports = [
        json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for directory in seed_dirs
    ]
    skill_names = seed_reports[0]["data"]["skill_vocabulary"]
    if any(report["data"]["skill_vocabulary"] != skill_names for report in seed_reports[1:]):
        raise ValueError("skill vocabulary changed across seeds")

    train = _read_jsonl(args.data_dir / "train_trajectories.jsonl")
    markov = CandidateMarkov(skill_names)
    markov.fit(train)
    split_analysis = {}
    paired_rows = {}
    for split, stem in (("validation", "val"), ("test", "test")):
        trajectories = _read_jsonl(args.data_dir / f"{stem}_trajectories.jsonl")
        meta_rows = _read_jsonl(args.data_dir / f"{stem}_metadata.jsonl")
        metadata = {row["trajectory_id"]: row for row in meta_rows}
        predictions = [
            _prediction_map(directory / f"{split}_free_predictions.jsonl")
            for directory in seed_dirs
        ]
        rows = _split_rows(split, trajectories, metadata, predictions, markov)
        paired_rows[split] = rows
        split_analysis[split] = {
            "overall": _summarize(rows),
            "task_cluster_bootstrap": {
                "exact_delta_model_minus_markov": cluster_bootstrap(
                    rows, "delta_exact", seed=20260722
                ),
                "edit_delta_model_minus_markov": cluster_bootstrap(
                    rows, "delta_edit", seed=20260723
                ),
            },
            "by_context_support": _group_summary(rows, "context_support"),
            "by_domain": _group_summary(rows, "domain"),
            "by_attack_family": _group_summary(rows, "attack_family"),
            "by_length": _group_summary(rows, "length_bin"),
            "supported_all_seeds": _summarize(
                [row for row in rows if row["all_seed_supported"]]
            ),
            "unsupported_in_any_seed": _summarize(
                [row for row in rows if not row["all_seed_supported"]]
            ),
        }

    previous_val = previous["event_summary"]["validation_next_tool_accuracy"]["mean"]
    previous_test = previous["event_summary"]["test_next_tool_accuracy"]["mean"]
    current_val = current["aggregate"]["validation_teacher_skill_accuracy"]["mean"]
    current_test = current["aggregate"]["test_teacher_skill_accuracy"]["mean"]
    subgroup_failures = {}
    for split, analysis in split_analysis.items():
        failures = []
        for grouping in ("by_context_support", "by_domain", "by_attack_family", "by_length"):
            for name, row in analysis[grouping].items():
                if row["trajectory_count"] < 5:
                    continue
                if row["model_minus_markov_edit"] >= 0 or row["model_minus_markov_exact"] <= 0:
                    failures.append(
                        {
                            "grouping": grouping,
                            "name": name,
                            "trajectory_count": row["trajectory_count"],
                            "task_count": row["task_count"],
                            "exact_delta": row["model_minus_markov_exact"],
                            "edit_delta": row["model_minus_markov_edit"],
                        }
                    )
        subgroup_failures[split] = failures

    audit = {
        "scope": "post-hoc counterevidence audit; no training or threshold changes",
        "fixed_gate_decision_unchanged": current["decision"],
        "paired_free_running": split_analysis,
        "historical_teacher_accuracy_context_only": {
            "previous_validation": previous_val,
            "current_validation": current_val,
            "delta_validation": current_val - previous_val,
            "previous_test": previous_test,
            "current_test": current_test,
            "delta_test": current_test - previous_test,
            "comparability_warning": (
                "The previous model scored selected-only vocabulary with UNK targets; "
                "the current model uses a training-candidate vocabulary and candidate masks."
            ),
        },
        "counterevidence": {
            "high_conservative_truncation": {
                split: current["aggregate"][
                    f"{split}_conservative_truncation_fraction"
                ]["mean"]
                for split in ("validation", "test")
            },
            "limited_full_sequence_seed_agreement": {
                split: current["ensemble_diagnostics"][split][
                    "all_seed_full_sequence_agreement"
                ]
                for split in ("validation", "test")
            },
            "historical_test_teacher_accuracy_regressed": current_test < previous_test,
            "subgroups_not_better_on_both_exact_and_edit": subgroup_failures,
        },
        "interpretation_boundary": (
            "Passing the frozen architecture gates establishes a feasible diagnostic "
            "victim-event model on the existing data.  High abstention, seed disagreement, "
            "subgroup failures, and the independent clean NO-GO prevent a formal attack-"
            "planning or Dreamer claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
