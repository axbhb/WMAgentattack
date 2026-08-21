"""Combine the three independent v22 conclusions without relaxing any gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-gate", type=Path, required=True)
    parser.add_argument("--long-gate", type=Path, required=True)
    parser.add_argument("--data-design-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    open_gate = json.loads(args.open_gate.read_text(encoding="utf-8"))
    long_gate = json.loads(args.long_gate.read_text(encoding="utf-8"))
    data_gate = json.loads(args.data_design_gate.read_text(encoding="utf-8"))
    open_go = open_gate["decision"] == "GO_COMPOSITIONAL_OPEN_VOCABULARY_V22"
    long_go = long_gate["decision"] == "GO_LONG_HORIZON_H1_H5_V22"
    design_go = data_gate["decision"].startswith("GO_DATA_GENERATION_PROTOCOL_READY_V22")
    if open_go and long_go and design_go:
        decision = "GO_RUN_FROZEN_96_EPISODE_DATA_SMOKE_V22"
    else:
        decision = "NO_GO_FORMAL_SCALE_V22__RETAIN_INDEPENDENT_COUNTEREVIDENCE"
    payload = {
        "schema_version": "wmagentattack.parallel_world_model_gates.v22",
        "decision": decision,
        "stage_decisions": {
            "open_vocabulary": open_gate["decision"],
            "data_generation_design": data_gate["decision"],
            "long_horizon": long_gate["decision"],
        },
        "authorization": {
            "run_frozen_96_episode_data_smoke": decision == "GO_RUN_FROZEN_96_EPISODE_DATA_SMOKE_V22",
            "medium_scale_generation": False,
            "large_scale_generation": False,
            "large_world_model_training": False,
            "attack_selector_training": False,
        },
        "hashes": {
            "open_gate": sha256(args.open_gate),
            "long_gate": sha256(args.long_gate),
            "data_design_gate": sha256(args.data_design_gate),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Parallel world-model gates v22: results",
        "",
        f"Overall decision: `{decision}`",
        "",
        "## Independent stage decisions",
        "",
        f"- Open vocabulary: `{open_gate['decision']}` ({open_gate['passed']}/{open_gate['total']} clauses).",
        f"- Data generation design: `{data_gate['decision']}` ({data_gate['passed']}/{data_gate['total']} clauses).",
        f"- Long horizon: `{long_gate['decision']}` ({long_gate['passed']}/{long_gate['total']} clauses).",
        "",
        "The data-design result does not claim that the 96 episodes already exist. Formal scale remains prohibited. H10 remains diagnostic because its task support is incomplete.",
        "",
        "## Open-vocabulary panels",
        "",
        "```json",
        json.dumps(open_gate["panels"], indent=2, sort_keys=True),
        "```",
        "",
        "## Long-horizon metrics",
        "",
        "```json",
        json.dumps(long_gate["metrics"], indent=2, sort_keys=True),
        "```",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision}))


if __name__ == "__main__":
    main()
