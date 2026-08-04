"""Select a score-stratified validation pilot for multi-seed replay.

Selection is intentionally label-blind. It covers the current final-utility
Pareto tail, the clean-solvability-aware expected-utility tail, and candidates
where calibrated and raw scores are uncertain or disagree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_pareto_module():
    path = ROOT / "scripts" / "18_pareto_utility_selection.py"
    spec = importlib.util.spec_from_file_location("pareto_utility_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Pareto selector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARETO = _load_pareto_module()


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _uncertainty_score(row: dict[str, Any]) -> float:
    risk = float(row.get("candidate_risk_score", row.get("risk_score", 0.0)))
    utility = float(
        row.get("candidate_expected_utility_score", row.get("utility_score", 0.0))
    )
    raw_risk = float(row.get("risk_score", risk))
    raw_utility = float(row.get("final_utility_score", utility))
    bernoulli_uncertainty = risk * (1.0 - risk) + utility * (1.0 - utility)
    disagreement = abs(risk - raw_risk) + abs(utility - raw_utility)
    return bernoulli_uncertainty + 0.5 * disagreement


def _rank_pareto(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    return PARETO._select_pareto(
        candidates,
        top_k=limit,
        utility_key=str(config["utility_key"]),
        threshold=threshold,
        max_per_user_task=0,
    )


def _add_stratum(
    output: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    name: str,
    quota: int,
    max_per_user_task: int,
) -> None:
    seen = {_pair_key(row) for row in output}
    task_counts = Counter(_task_key(row) for row in output)
    added = 0
    for row in candidates:
        if _pair_key(row) in seen:
            continue
        task = _task_key(row)
        if task_counts[task] >= max_per_user_task:
            continue
        output.append({**row, "pilot_stratum": name})
        seen.add(_pair_key(row))
        task_counts[task] += 1
        added += 1
        if added == quota:
            return
    raise RuntimeError(f"Unable to fill stratum {name}: {added}/{quota}")


def _select_pilot(
    candidates: list[dict[str, Any]],
    strict_payload: dict[str, Any],
    *,
    quota: int,
    max_per_user_task: int,
) -> list[dict[str, Any]]:
    final_config = strict_payload["pareto"]["selected_validation_config"]
    final_threshold = strict_payload["pareto"]["test_frozen_validation_threshold"][
        "per_seed"
    ][0]["threshold"]
    expected = strict_payload["pareto"]["validation_best_by_utility_key"][
        "candidate_expected_utility_score"
    ]
    expected_config = expected["selected_validation_config"]
    expected_threshold = expected["test_frozen_validation_threshold"]["per_seed"][0][
        "threshold"
    ]

    final_ranked = _rank_pareto(
        candidates, final_config, float(final_threshold), len(candidates)
    )
    expected_ranked = _rank_pareto(
        candidates, expected_config, float(expected_threshold), len(candidates)
    )
    uncertain_ranked = sorted(
        candidates, key=_uncertainty_score, reverse=True
    )

    output: list[dict[str, Any]] = []
    _add_stratum(
        output,
        final_ranked,
        name="final_utility_pareto",
        quota=quota,
        max_per_user_task=max_per_user_task,
    )
    _add_stratum(
        output,
        expected_ranked,
        name="expected_utility_pareto",
        quota=quota,
        max_per_user_task=max_per_user_task,
    )
    _add_stratum(
        output,
        uncertain_ranked,
        name="uncertainty_disagreement",
        quota=quota,
        max_per_user_task=max_per_user_task,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--strict-json", type=Path, required=True)
    parser.add_argument("--quota-per-stratum", type=int, default=16)
    parser.add_argument("--max-per-user-task", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    strict_payload = json.loads(args.strict_json.read_text(encoding="utf-8"))
    selected = _select_pilot(
        candidate_payload["candidates"],
        strict_payload,
        quota=args.quota_per_stratum,
        max_per_user_task=args.max_per_user_task,
    )
    output = {
        "scope": "validation_probability_label_pilot",
        "candidate_json": str(args.candidate_json.resolve()),
        "strict_json": str(args.strict_json.resolve()),
        "selection_uses_observed_labels": False,
        "quota_per_stratum": args.quota_per_stratum,
        "max_per_user_task": args.max_per_user_task,
        "stratum_counts": dict(Counter(row["pilot_stratum"] for row in selected)),
        "selections": {"validation_probability_pilot": selected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "selections"}, indent=2))


if __name__ == "__main__":
    main()
