"""Label-blind coverage audit for the interface-aligned affordance graph."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import stack_interface_affordance_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-dimension", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-concepts", type=int, default=20)
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    events = data["events"]
    slots = stack_interface_affordance_states(
        events, hash_dimension=args.hash_dimension,
        max_nodes=args.max_nodes, max_concepts=args.max_concepts,
    )
    audits = slots["audit"]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for event, audit in zip(events, audits):
        by_task[event["task_name"]].append(audit)
    task_rows = {}
    for task, rows in sorted(by_task.items()):
        task_rows[task] = {
            "events": len(rows),
            "mean_matched_interface_concepts": float(np.mean([row["matched_interface_concepts"] for row in rows])),
            "fraction_with_any_concept": float(np.mean([row["matched_interface_concepts"] > 0 for row in rows])),
            "mean_tools_with_goal_overlap": float(np.mean([row["tools_with_goal_overlap"] for row in rows])),
            "mean_tools_with_observation_overlap": float(np.mean([row["tools_with_observation_overlap"] for row in rows])),
        }
    result = {
        "dataset": str(args.dataset), "dataset_sha256": file_sha256(args.dataset),
        "events": len(events), "tasks": len(by_task),
        "parameters": {
            "hash_dimension": args.hash_dimension, "max_nodes": args.max_nodes,
            "max_concepts": args.max_concepts,
        },
        "integrity": {
            "raw_values_encoded": any(row["raw_values_encoded"] for row in audits),
            "interface_only_lexical_encoding": all(row["interface_only_lexical_encoding"] for row in audits),
            "unmatched_text_tokens_encoded": sum(row["unmatched_text_tokens_encoded"] for row in audits),
            "truncated_rows": sum(row["truncated"] for row in audits),
            "concept_truncated_rows": sum(row["concepts_truncated"] for row in audits),
            "maximum_nodes": max(row["node_count"] for row in audits),
        },
        "coverage": {
            "mean_matched_interface_concepts": float(np.mean([row["matched_interface_concepts"] for row in audits])),
            "fraction_with_any_concept": float(np.mean([row["matched_interface_concepts"] > 0 for row in audits])),
            "fraction_with_goal_tool_overlap": float(np.mean([row["tools_with_goal_overlap"] > 0 for row in audits])),
            "fraction_with_observation_tool_overlap": float(np.mean([row["tools_with_observation_overlap"] > 0 for row in audits])),
        },
        "tasks": task_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["integrity"], sort_keys=True))
    print(json.dumps(result["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
