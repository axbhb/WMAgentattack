import json
from pathlib import Path
import socket

import pytest
from wmagentattack.clean_pairing import paired_episode_seed, build_clean_plan, clean_gate, block_python_network


def protocol():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/0831_clean_pairing_v37_protocol.json").read_text())


def complete_results(p):
    return [{"task": task, "run_seed": seed, "episode_seed": paired_episode_seed(seed, task, p["tasks"]),
             "utility": True, "status": "completed", "blocked_network_attempts": 0}
            for seed in p["run_seeds"] for task in p["tasks"]]


def test_common_seed_has_no_variant_input_and_is_unique_across_episodes():
    p = protocol()
    allocation = [paired_episode_seed(s, t, p["tasks"]) for s in p["run_seeds"] for t in p["tasks"]]
    assert len(set(allocation)) == 60
    assert [paired_episode_seed(601, p["tasks"][0], p["tasks"]) for _variant in range(5)] == [6010000] * 5


def test_clean_plan_retains_all_twenty_tasks_without_label_selection():
    p = protocol()
    manifest = [{"suite": t.split("|")[0], "user_task_id": t.split("|")[1]} for t in p["tasks"]]
    assert len(build_clean_plan(p, manifest * 20)) == 60
    with pytest.raises(ValueError, match="alignment"):
        build_clean_plan(p, manifest[:-1])


def test_complete_gate_passes_and_infrastructure_missing_is_invalid():
    p = protocol()
    rows = complete_results(p)
    assert clean_gate(rows, p, lambda row: True)["decision"] == "GO_CLEAN_PAIRING_V37"
    assert clean_gate(rows[:-1], p, lambda row: True)["decision"] == "INVALID_CLEAN_PAIRING_V37"
    assert clean_gate(rows + rows[:1], p, lambda row: True)["decision"] == "INVALID_CLEAN_PAIRING_V37"


def test_all_failures_are_scientific_no_go_not_infrastructure():
    p = protocol()
    rows = complete_results(p)
    for row in rows: row["utility"] = False
    result = clean_gate(rows, p, lambda row: True)
    assert result["decision"] == "NO_GO_CLEAN_PAIRING_V37"
    assert result["scientific_result"] is True


def test_wrong_seed_or_missing_trace_or_network_attempt_invalidates():
    p = protocol()
    rows = complete_results(p)
    rows[0]["episode_seed"] += 1
    assert not clean_gate(rows, p, lambda row: True)["scientific_result"]
    rows = complete_results(p)
    assert not clean_gate(rows, p, lambda row: False)["scientific_result"]
    rows[0]["blocked_network_attempts"] = 1
    assert not clean_gate(rows, p, lambda row: True)["scientific_result"]


def test_network_guard_rejects_before_contact_and_restores():
    original = socket.socket.connect
    with socket.socket() as sock:
        with block_python_network() as state:
            with pytest.raises(RuntimeError, match="Network denied"):
                sock.connect(("127.0.0.1", 9))
            assert state["blocked_attempts"] == 1
    assert socket.socket.connect is original
