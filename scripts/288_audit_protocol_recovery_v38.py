"""Read-only parser replay; schemas inspected, tools never executed."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.functions_runtime import FunctionsRuntime
from wmagentattack.clean_trace_audit import content_text
from wmagentattack.clean_pairing import block_python_network
from wmagentattack.qwen_agentdojo import TransformersQwenLLM
from wmagentattack.protocol_recovery import explicit_call, valid_runtime_arguments


def normalized(calls):
    return [{"function": c.function, "args": c.args} for c in calls]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    stage = p["v38"]
    out = Path(stage["archive"]) / "gate.json"
    if out.exists():
        raise FileExistsError("v38 replay already recorded")
    results = json.loads((Path(stage["data"]) / "results.json").read_text())
    if len(results) != 60 or any(r["status"] != "completed" for r in results):
        raise ValueError("v37 must be complete")
    details, strict_changes, raw_mismatches = [], 0, 0
    with block_python_network() as network:
        runtimes = {s: FunctionsRuntime(get_suite("v1.2.2", s).tools) for s in ["banking", "slack", "travel", "workspace"]}
        for row in results:
            runtime = runtimes[row["task"].split("|")[0]]
            raw = json.loads(Path(row["raw_trace"]).read_text())
            assistants = [m for m in raw["messages"] if m["role"] == "assistant"]
            original_calls, recovered, legacy_new = 0, 0, 0
            for message in assistants:
                text = content_text(message.get("content"))
                strict = normalized(TransformersQwenLLM._parse_completion(text, set(runtime.functions))["tool_calls"])
                recorded = [{"function": c["function"], "args": c["args"]} for c in (message.get("tool_calls") or [])]
                raw_mismatches += int(strict != recorded)
                original_calls += len(strict)
                new = explicit_call(text, set(runtime.functions), lambda n, a: valid_runtime_arguments(runtime, n, a)) if not strict else None
                candidate = strict if strict else [new] if new else []
                if strict: strict_changes += int(candidate != strict)
                recovered += int(bool(new))
                legacy_new += int(not strict and bool(TransformersQwenLLM._parse_repaired_completion(text, set(runtime.functions))["tool_calls"]))
            details.append({"task": row["task"], "seed": row["run_seed"], "utility": row["utility"],
                            "original_parsed_calls": original_calls, "new_validated_calls": recovered,
                            "legacy_unvalidated_recoveries": legacy_new})
    extra_episodes = sum(d["new_validated_calls"] > 0 for d in details)
    changed_toolfree_success = sum(d["utility"] and d["original_parsed_calls"] == 0 and d["new_validated_calls"] > 0 for d in details)
    checks = {"at_least_one_new_valid_call_episode": extra_episodes >= stage["gate"]["minimum_new_valid_call_episodes"],
              "strict_calls_unchanged": strict_changes == 0,
              "successful_toolfree_episodes_unchanged": changed_toolfree_success == 0,
              "strict_replay_matches_raw": raw_mismatches == 0,
              "no_network_attempts": network["blocked_attempts"] == 0}
    result = {"decision": "GO_PARSER_VIABILITY_V38" if all(checks.values()) else "NO_GO_PARSER_VIABILITY_V38",
              "checks": checks, "new_valid_call_episodes": extra_episodes, "strict_changes": strict_changes,
              "raw_mismatches": raw_mismatches, "successful_toolfree_changes": changed_toolfree_success,
              "new_validated_calls": sum(d["new_validated_calls"] for d in details),
              "legacy_unvalidated_recoveries": sum(d["legacy_unvalidated_recoveries"] for d in details),
              "new_episodes": 0, "tool_executions": 0, "model_generations": 0, "details": details,
              "interpretation": "Parser viability only; a recovered call is not a task success."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, sort_keys=True))


if __name__ == "__main__":
    main()
