import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "0904_llama31_8b_dualsource_full_protocol.json"
RUNNER = ROOT / "scripts" / "server" / "run_0904_llama31_8b_dualsource_full_friend.sh"


def test_full_budget_is_frozen_and_explicitly_exploratory():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "user_authorized_exploratory_full_collection_frozen_before_run"
    assert protocol["scientific_context"]["pilot_result_is_not_overridden"] is True
    assert protocol["victim_model"]["decoding"]["seeds"] == [1103, 1109, 1117]
    assert protocol["agentdojo"]["budget"] == {
        "user_tasks": 97,
        "compatible_user_injection_pairs_per_attack": 629,
        "clean_trajectories": 291,
        "attack_trajectories": 15096,
        "selected_trajectories": 15387,
    }
    assert protocol["injecagent"]["budget"]["cases"] == 1054
    assert protocol["injecagent"]["budget"]["decisions"] == 6324
    assert protocol["fixed_total_model_interactions"] == 21711


def test_single_model_and_offline_boundary_are_frozen():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["victim_model"]["model"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert protocol["authorization_boundary"]["synthetic_or_offline_benchmarks_only"] is True
    assert protocol["authorization_boundary"]["real_external_endpoint_calls"] == 0
    assert len(protocol["agentdojo"]["published_attacks"]) == 8
    assert "never consecutive time steps" in protocol["injecagent"]["transition_semantics"]


def test_runner_is_resumable_and_refuses_duplicate_launch():
    text = RUNNER.read_text(encoding="utf-8")
    assert "--force-rerun" not in text
    assert "Full collection already launched; refusing duplicate." in text
    assert "run_agentdojo_seed 1103 0" in text
    assert "run_agentdojo_seed 1109 1" in text
    assert "run_agentdojo_seed 1117 0" in text
    assert "full_collection.complete" in text


def test_content_checksums_remain_disabled():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization_boundary"]["content_checksums_disabled_by_user"] is True
    for path in (
        ROOT / "scripts" / "306_build_llama31_8b_dualsource_full.py",
        ROOT / "scripts" / "307_run_llama31_8b_injecagent_full.py",
        ROOT / "scripts" / "308_finalize_llama31_8b_dualsource_full.py",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "sha256" not in text
