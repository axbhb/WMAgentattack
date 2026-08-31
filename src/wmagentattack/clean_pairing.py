"""Frozen task-level seed allocation and clean gate, without content digests."""
from collections import Counter
from contextlib import contextmanager
import socket


def paired_episode_seed(run_seed, task, frozen_tasks):
    if len(set(frozen_tasks)) != len(frozen_tasks) or len(frozen_tasks) >= 100:
        raise ValueError("invalid frozen task list")
    if task not in frozen_tasks or not 0 <= run_seed < 200000:
        raise ValueError("seed/task outside frozen allocation")
    return int(run_seed) * 10000 + frozen_tasks.index(task) * 100


def build_clean_plan(protocol, manifest_rows):
    expected = protocol["tasks"]
    observed = {f"{row['suite']}|{row['user_task_id']}" for row in manifest_rows}
    if len(expected) != 20 or set(expected) != observed or len(set(expected)) != 20:
        raise ValueError("frozen task alignment failed")
    seeds = protocol["run_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("three distinct run seeds required")
    plan = [{"task": task, "run_seed": seed,
             "episode_seed": paired_episode_seed(seed, task, expected)} for seed in seeds for task in expected]
    if len({row["episode_seed"] for row in plan}) != 60:
        raise ValueError("episode seed collision")
    return plan


def clean_gate(results, protocol, raw_trace_valid):
    expected = {(task, seed) for task in protocol["tasks"] for seed in protocol["run_seeds"]}
    keys = [(row.get("task"), row.get("run_seed")) for row in results]
    checks = {"complete_exact_episode_set": len(keys) == len(expected) and set(keys) == expected and len(set(keys)) == len(keys),
              "zero_runtime_failures": all(row.get("status") == "completed" for row in results),
              "valid_binary_utility": all(type(row.get("utility")) is bool for row in results),
              "zero_network_attempts": all(row.get("blocked_network_attempts") == 0 for row in results),
              "seed_alignment": all(row.get("task") in protocol["tasks"] and row.get("episode_seed") == paired_episode_seed(row["run_seed"], row["task"], protocol["tasks"]) for row in results),
              "raw_traces_complete": all(raw_trace_valid(row) for row in results)}
    if not all(checks.values()):
        return {"decision": "INVALID_CLEAN_PAIRING_V37", "integrity": checks, "recorded": len(results), "scientific_result": False}
    counts = Counter(row["task"] for row in results if row["utility"])
    stable = [task for task in protocol["tasks"] if counts[task] >= protocol["gate"]["successes_to_be_stable"]]
    by_suite = Counter(task.split("|", 1)[0] for task in stable)
    science = {"stable_task_coverage": len(stable) >= protocol["gate"]["minimum_stable_tasks"],
               "suite_coverage": sum(count >= 2 for count in by_suite.values()) >= protocol["gate"]["minimum_suites_with_two_stable_tasks"]}
    return {"decision": "GO_CLEAN_PAIRING_V37" if all(science.values()) else "NO_GO_CLEAN_PAIRING_V37",
            "integrity": checks, "checks": science, "scientific_result": True, "completed": len(results),
            "clean_successes": sum(counts.values()), "clean_utility": sum(counts.values()) / len(results),
            "stable_tasks": stable, "stable_task_count": len(stable), "stable_by_suite": dict(by_suite),
            "successes_by_task": {task: counts[task] for task in protocol["tasks"]}}


@contextmanager
def block_python_network():
    """Deny Python IPv4/IPv6 socket connect/sendto; leave local IPC available."""
    state = {"blocked_attempts": 0}
    original_connect, original_connect_ex, original_sendto = socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto
    def denied(sock):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            state["blocked_attempts"] += 1
            raise RuntimeError("Network denied in the offline AgentDojo clean sandbox")
    def connect(sock, address):
        denied(sock)
        return original_connect(sock, address)
    def connect_ex(sock, address):
        denied(sock)
        return original_connect_ex(sock, address)
    def sendto(sock, *args):
        denied(sock)
        return original_sendto(sock, *args)
    socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = connect, connect_ex, sendto
    try:
        yield state
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = original_connect, original_connect_ex, original_sendto
