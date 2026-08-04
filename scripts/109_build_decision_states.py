"""Build label-blind canonical decision states from an AgentDojo-v2 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.decision_state import build_manifest_decision_state
from wmagentattack.io_utils import write_jsonl


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_contexts(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
        return {
            f"{row['suite']}|{row['user_task_id']}": row
            for row in rows
        }
    if isinstance(payload, dict):
        return {str(key): dict(value) for key, value in payload.items()}
    raise ValueError("task context must be a mapping or an object with a tasks list")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--victim-model", required=True)
    parser.add_argument("--agent-scaffold", required=True)
    parser.add_argument("--defense", default="none")
    parser.add_argument("--exclude-clean-prior", action="store_true")
    parser.add_argument(
        "--smoke-allow-missing-task-context",
        action="store_true",
        help="Allow states without trusted goal/tool/environment context for schema smoke tests.",
    )
    args = parser.parse_args()

    payload = _read_json(args.manifest)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty rows list")
    contexts = _task_contexts(args.task_context)
    states = []
    missing_context = []
    for row in rows:
        key = f"{row['suite']}|{row['user_task_id']}"
        context = contexts.get(key)
        if context is None:
            missing_context.append(key)
        states.append(
            build_manifest_decision_state(
                row,
                victim_model=args.victim_model,
                agent_scaffold=args.agent_scaffold,
                defense=args.defense,
                task_context=context,
                include_clean_prior=not args.exclude_clean_prior,
            )
        )
    missing_context = sorted(set(missing_context))
    if missing_context and not args.smoke_allow_missing_task_context:
        raise ValueError(
            f"Missing trusted task context for {len(missing_context)} tasks; "
            "provide --task-context before a training-data build"
        )
    fingerprints = [state.state_fingerprint for state in states]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("semantic decision-state fingerprints are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, states)
    ignored_paths = Counter(
        path
        for state in states
        for path in state.source_audit.ignored_outcome_paths
    )
    audit = {
        "scope": "pre-rollout canonical states; AgentDojo sandbox only",
        "schema_version": states[0].schema_version,
        "manifest": str(args.manifest.resolve()),
        "state_count": len(states),
        "unique_fingerprints": len(set(fingerprints)),
        "missing_trusted_context_tasks": missing_context,
        "training_ready": not missing_context,
        "clean_prior_included": not args.exclude_clean_prior,
        "ignored_outcome_path_counts": dict(sorted(ignored_paths.items())),
        "attack_families": dict(Counter(state.attacker_action.family for state in states)),
        "task_splits": dict(Counter(state.task_split for state in states)),
        "output": str(args.output.resolve()),
    }
    audit_path = args.audit_output or args.output.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

