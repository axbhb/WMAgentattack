"""Post-result audit only: no model/tool execution and no gate or label changes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wmagentattack.protocol_recovery_eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args()
    root = args.archive.resolve()
    protocol = json.loads((root / "preregistered_protocol.json").read_text())
    parent = json.loads(Path(protocol["parent_protocol"]).read_text())
    stage = protocol["v39"]
    assert Path(stage["archive"]).resolve() == root
    rows = json.loads((root / "results.json").read_text())
    original = json.loads((root / "gate.json").read_text())
    assert (root / "COMPLETE").is_file()
    assert len(rows) == 180
    assert len(list((root / "raw").rglob("none.json"))) == 180
    assert len(list((root / "diagnostics").glob("episode_*.json"))) == 180

    def artifacts(row):
        rawpath, diagpath = Path(row["raw_trace"]).resolve(), Path(row["diagnostics"]).resolve()
        assert rawpath.is_relative_to(root / "raw") and diagpath.is_relative_to(root / "diagnostics")
        raw, diag = json.loads(rawpath.read_text()), json.loads(diagpath.read_text())
        assert type(raw["utility"]) is bool
        assert len(diag["events"]) == row["generation_calls"]
        extra = len(diag["events"]) - sum(m["role"] == "assistant" for m in raw["messages"])
        assert extra == row["extra_retry_generations"]
        return raw, diag

    recomputed = evaluate(rows, parent["tasks"], stage["seeds"], stage["arms"], stage["gate"], artifacts)
    for key, value in recomputed.items():
        assert original[key] == value, f"Gate mismatch: {key}"
    assert recomputed["scientific_result"]
    evidence = {}
    by_key = {(r["arm"], r["task"], r["run_seed"]): r for r in rows}
    independent_pvalues = {}
    for arm in stage["arms"]:
        subset = [r for r in rows if r["arm"] == arm]
        diag = [artifacts(r)[1] for r in subset]
        evidence[arm] = {
            "failures_with_parsed_calls": sum(not r["utility"] and r["parsed_tool_calls"] > 0 for r in subset),
            "successes_without_parsed_calls": sum(r["utility"] and r["parsed_tool_calls"] == 0 for r in subset),
            "extra_retry_generations": sum(r["extra_retry_generations"] for r in subset),
            "episodes_with_recovered_calls": sum(d["recovered_calls"] > 0 for d in diag),
            "recovered_calls": sum(d["recovered_calls"] for d in diag),
            "input_tokens": sum(e["input_tokens"] for d in diag for e in d["events"]),
            "episode_elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in subset),
        }
        if arm == "strict":
            continue
        transitions = Counter()
        conversions = Counter()
        for row in subset:
            baseline = by_key[("strict", row["task"], row["run_seed"])]
            transitions[f"{int(baseline['utility'])}->{int(row['utility'])}"] += 1
            if not baseline["utility"] and baseline["parsed_tool_calls"] == 0:
                conversions[f"candidate_calls_{int(row['parsed_tool_calls'] > 0)}_success_{int(row['utility'])}"] += 1
        evidence[arm]["paired_utility_transitions"] = dict(transitions)
        evidence[arm]["strict_zero_call_failure_followup"] = dict(conversions)
        differences = [sum(int(by_key[(arm,t,s)]["utility"])-int(by_key[("strict",t,s)]["utility"]) for s in stage["seeds"]) for t in parent["tasks"]]
        nonzero = [abs(d) for d in differences if d]
        extreme = sum(sum(a*b for a,b in zip(signs, nonzero)) >= sum(differences) for signs in product([-1,1], repeat=len(nonzero)))
        pvalue = extreme / 2 ** len(nonzero)
        assert pvalue == original["comparisons"][arm]["task_signflip_p"]
        independent_pvalues[arm] = pvalue

    current = json.loads(Path("configs/0831_protocol_recovery_v38_v40_protocol.json").read_text())
    assert all(current[k] == v for k,v in protocol.items() if k != "status")
    source_paths = ["src/wmagentattack/qwen_agentdojo.py", "src/wmagentattack/clean_pairing.py", "src/wmagentattack/protocol_recovery.py", "src/wmagentattack/protocol_recovery_adapter.py", "src/wmagentattack/protocol_recovery_eval.py", "scripts/289_run_protocol_recovery_v39_v40.py", "scripts/server/run_0831_protocol_recovery_v39.sbatch", "scripts/server/run_0831_protocol_recovery_v40.sbatch"]
    assert not subprocess.check_output(["git", "diff", "ad3cecb", "--", *source_paths], text=True)
    sm = original["metrics"]["strict"]
    rm = original["metrics"]["syntax_retry"]
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "post-result read-only diagnostic; not candidate tuning or confirmation",
        "gate_recomputed_equal": True, "exact_pvalues_independently_enumerated": independent_pvalues,
        "raw_count": 180, "diagnostic_count": 180, "complete_sentinel": (root / "COMPLETE").read_text().strip(),
        "scientific_decision": original["decision"], "v40": "NOT_AUTHORIZED" if not original["selected_arm"] else "CONDITIONAL_GO",
        "frozen_contract_equal": True, "experiment_implementation_unchanged": True,
        "generation_call_increase_fraction": rm["generation_calls"] / sm["generation_calls"] - 1,
        "output_token_increase_fraction": rm["output_tokens"] / sm["output_tokens"] - 1,
        "diagnostics": evidence, "slurm_exit_code": None,
        "exit_code_note": "Scheduler record no longer retained; completion sentinel and all raw records verified, no invented exit code.",
        "new_generations": 0, "new_tool_executions": 0, "model_fits": 0,
    }
    out = root / "posthoc_closeout_audit.json"
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
