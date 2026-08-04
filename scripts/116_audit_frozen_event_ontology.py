"""Audit AgentDojo-v2 trajectories against the frozen event ontology."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.event_ontology import (
    EXCLUDED_OUTCOME_FIELDS,
    normalize_victim_event,
    ontology_fingerprint,
    ontology_specification,
)
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import TrajectoryRecord


def audit_split(path: Path) -> dict:
    trajectories = [TrajectoryRecord.model_validate(row) for row in read_jsonl(path)]
    status = Counter()
    candidate_source = Counter()
    termination = Counter()
    unavailable = Counter()
    task_events = Counter()
    candidate_fingerprints = set()
    forbidden_seen = set()
    for trajectory in trajectories:
        task_key = f"{trajectory.domain}|{trajectory.task_id}"
        for index, step in enumerate(trajectory.steps):
            event = normalize_victim_event(
                step, is_last_observed_event=(index == len(trajectory.steps) - 1)
            )
            payload = event.model_dump(mode="json")
            forbidden_seen.update(EXCLUDED_OUTCOME_FIELDS & set(payload))
            status[event.tool_execution_status] += 1
            candidate_source[event.candidate_source] += 1
            termination[event.termination_reason or "not_terminal"] += 1
            unavailable.update(event.unavailable_fields)
            task_events[task_key] += 1
            candidate_fingerprints.add(event.candidate_manifest_fingerprint)
    if forbidden_seen:
        raise ValueError(f"outcome fields leaked into ontology: {sorted(forbidden_seen)}")
    return {
        "trajectory_count": len(trajectories),
        "task_count": len(task_events),
        "event_count": sum(task_events.values()),
        "events_per_task": dict(sorted(task_events.items())),
        "tool_execution_status": dict(sorted(status.items())),
        "candidate_source": dict(sorted(candidate_source.items())),
        "termination_reason": dict(sorted(termination.items())),
        "unavailable_field_mentions": dict(sorted(unavailable.items())),
        "unique_candidate_manifests": len(candidate_fingerprints),
        "forbidden_outcome_fields_seen": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "label-blind ontology audit; existing AgentDojo-v2 data only",
        "ontology_fingerprint": ontology_fingerprint(),
        "specification": ontology_specification(),
        "splits": {
            "train": audit_split(args.train),
            "validation": audit_split(args.validation),
            "test": audit_split(args.test),
        },
        "training_readiness": {
            "skill_and_candidate_dynamics": True,
            "argument_slot_auxiliary_target": True,
            "canonical_state_delta_model": False,
            "task_progress_delta_model": False,
            "irreversible_effect_model": False,
            "reason": (
                "The current v2 archive has no exact canonical-state delta, goal-slot "
                "progress, entity-link, or irreversible-effect annotations."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
