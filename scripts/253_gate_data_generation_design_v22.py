"""Gate the frozen data-generation design without pretending episodes were run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = protocol["data_generation_gate"]
    budget = cfg["smoke_budget"]
    gates = cfg["hard_gates"]
    clauses = {
        "four_linked_tables": cfg["episode_tables"] == ["episodes", "transitions", "outcomes", "pairs"],
        "four_joint_classes": len(set(cfg["trajectory_outcome_classes"])) == 4,
        "outcomes_are_episode_level": "episode-level" in cfg["outcome_rule"],
        "all_three_sources": set(cfg["sources"]) == {"agentdojo", "tool_sandbox", "injecagent"},
        "injecagent_is_observation_only": cfg["transition_tiers"]["injecagent"] == "observation_contrast_only",
        "single_70b_contract": cfg["shared_llm_contract"]["model"] == "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "fixed_two_seeds": budget["seeds"] == [431, 433],
        "episode_budget_sums": budget["agentdojo_episodes"] + budget["tool_sandbox_episodes"] + budget["injecagent_episodes"] == budget["episodes"] == 96,
        "pair_budget": budget["paired_instances"] * 2 == budget["episodes"],
        "no_post_result_additions": budget["post_result_additions"] == 0,
        "component_disjoint_split": "Connected components" in cfg["split_rule"],
        "complete_smoke_gate_frozen": gates["complete_episodes"] == 96 and gates["complete_pairs"] == 48,
        "zero_external_endpoint_gate": gates["real_external_endpoint_calls"] == 0,
        "joint_support_gate_frozen": gates["minimum_valid_joint_examples_per_cell"] == 3 and gates["maximum_joint_class_fraction"] == 0.7,
        "large_generation_not_authorized": protocol["scope"]["large_scale_generation"] == 0,
    }
    decision = (
        "GO_DATA_GENERATION_PROTOCOL_READY_V22__96_EPISODE_SMOKE_NOT_RUN"
        if all(clauses.values())
        else "NO_GO_DATA_GENERATION_PROTOCOL_DESIGN_V22"
    )
    payload = {
        "schema_version": "wmagentattack.data_generation_design_gate.v22",
        "decision": decision,
        "clauses": clauses,
        "passed": sum(clauses.values()),
        "total": len(clauses),
        "protocol_sha256": sha256(args.protocol),
        "authorization": {
            "build_manifest_twice": all(clauses.values()),
            "run_96_episode_smoke": all(clauses.values()),
            "medium_generation": false,
            "large_generation": false,
            "large_world_model_training": false,
        },
        "claim_boundary": "This gate validates the schema, frozen budget, source tiers, LLM contract, and leakage policy. It does not claim that 96 episodes have been generated."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": sum(clauses.values()), "total": len(clauses)}))


if __name__ == "__main__":
    main()
