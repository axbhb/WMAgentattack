"""Apply the predeclared confirmation gate to a contrast-model summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def apply_gate(
    summary: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    selected = str(summary["selected_model"])
    comparator = str(protocol["model_selection"]["comparator"])
    comparison_key = f"{selected}__minus__{comparator}"
    try:
        comparison = summary["comparisons"][comparison_key]
    except KeyError as exc:
        raise ValueError(
            f"Summary lacks required comparison {comparison_key!r}"
        ) from exc
    criteria = protocol["go_criteria_all_required"]
    pairwise_difference = float(comparison["pairwise_accuracy_difference"])
    pairwise_ci = [
        float(value)
        for value in comparison["pairwise_accuracy_difference_95ci"]
    ]
    brier_difference = float(comparison["brier_difference"])
    checks = {
        "pairwise_point_improvement": {
            "observed": pairwise_difference,
            "required": float(
                criteria["minimum_pairwise_accuracy_difference"]
            ),
            "passed": pairwise_difference
            >= float(criteria["minimum_pairwise_accuracy_difference"]),
        },
        "pairwise_uncertainty_bound": {
            "observed": pairwise_ci[0],
            "required": float(
                criteria[
                    "minimum_pairwise_accuracy_difference_95ci_lower"
                ]
            ),
            "passed": pairwise_ci[0]
            >= float(
                criteria[
                    "minimum_pairwise_accuracy_difference_95ci_lower"
                ]
            ),
        },
        "brier_non_degradation": {
            "observed": brier_difference,
            "required_maximum": float(criteria["maximum_brier_difference"]),
            "passed": brier_difference
            <= float(criteria["maximum_brier_difference"]),
        },
    }
    go = all(item["passed"] for item in checks.values())
    return {
        "scope": "within_task_confirmation_gate_decision",
        "selected_model": selected,
        "comparator": comparator,
        "comparison_key": comparison_key,
        "go": go,
        "decision": "GO" if go else "NO_GO",
        "checks": checks,
        "observed_pairwise_accuracy_difference_95ci": pairwise_ci,
        "next_action": (
            protocol["decision"]["go"]
            if go
            else protocol["decision"]["no_go"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = _load(args.summary)
    protocol = _load(args.protocol)
    decision = apply_gate(summary, protocol)
    decision["summary"] = str(args.summary.resolve())
    decision["protocol"] = str(args.protocol.resolve())
    decision["protocol_sha256"] = hashlib.sha256(
        args.protocol.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
