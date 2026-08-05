import copy

from wmagentattack.tau3_interactive import (
    build_interactive_dataset,
    reconstruct_agent_surface,
    role_seed,
    runtime_agent_model_input,
)


def _schema(name: str = "lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up one synthetic record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }


def _interface():
    return {"tool_schemas": [_schema()], "policy": "Use the synthetic tool."}


def _decision(index: int, decision, messages):
    return {
        "generation_index": index,
        "decision": decision,
        "natural_user_messages": messages,
        "agent_visible_dialogue_sha256": f"visible-{index}",
        "agent_input_provenance": [
            "official_agent_instruction",
            "domain_policy",
            "agent_tool_schemas",
            "natural_user_messages",
            "own_text_messages",
            "own_tool_calls_and_receipts",
        ],
        "private_user_scenario_directly_serialized": False,
    }


def _assistant_event():
    return {
        "combined_index": 1,
        "requestor": "assistant",
        "action": {"name": "lookup", "arguments": {"record_id": "alice"}},
        "status": "success",
        "error": None,
        "output": {"record_id": "alice", "status": "active"},
        "state_before_sha256": "before",
        "state_after_sha256": "after",
        "state_changed": True,
        "replica_identical": True,
    }


def _user_event():
    return {
        "combined_index": 0,
        "requestor": "user",
        "action": {"name": "user_lookup", "arguments": {}},
        "status": "success",
        "error": None,
        "output": "synthetic",
        "state_before_sha256": "initial",
        "state_after_sha256": "before",
        "state_changed": True,
        "replica_identical": True,
    }


def test_role_seeds_are_deterministic_and_role_separated():
    assert role_seed(401, "agent", 2) == role_seed(401, "agent", 2)
    assert role_seed(401, "agent", 2) != role_seed(401, "user", 2)
    assert role_seed(401, "agent", 2) != role_seed(401, "agent", 3)


def test_runtime_agent_input_is_whitelisted_and_causal():
    value = runtime_agent_model_input(
        _interface(), ["Find Alice.", "The identifier is A-1."]
    )
    assert set(value) == {"trusted_goal", "tool_schemas", "policy"}
    assert value["trusted_goal"] == (
        "User turn 1: Find Alice.\nUser turn 2: The identifier is A-1."
    )
    assert "private" not in repr(value).lower()


def test_reconstruction_keeps_user_tools_exogenous_and_maps_agent_prefixes():
    decisions = [
        _decision(
            0,
            {
                "kind": "tool_call",
                "name": "lookup",
                "arguments": {"record_id": "alice"},
                "repair": "strict",
            },
            ["Find Alice."],
        ),
        _decision(
            1,
            {"kind": "text", "text": "Alice is active.", "repair": None},
            ["Find Alice."],
        ),
    ]
    prefixes, transitions, audit = reconstruct_agent_surface(
        episode_id="episode",
        domain="retail",
        agent_interface=_interface(),
        agent_decisions=decisions,
        combined_tool_events=[_user_event(), _assistant_event()],
    )
    assert len(prefixes) == 2
    assert len(transitions) == 1
    assert transitions[0]["decision_prefix_index"] == 0
    assert prefixes[0]["decision_kind"] == "tool_call"
    assert prefixes[1]["decision_kind"] == "text"
    assert len(prefixes[1]["features"]["ledger_v2"]["records"]) == 1
    assert audit == {
        "agent_private_scenario_exposures": 0,
        "assistant_tool_events": 1,
        "user_tool_events": 1,
    }


def test_interactive_dataset_preserves_parent_pair_and_task_split():
    decisions = [
        _decision(
            0,
            {
                "kind": "tool_call",
                "name": "lookup",
                "arguments": {"record_id": "alice"},
                "repair": "strict",
            },
            ["Find Alice."],
        ),
        _decision(
            1,
            {"kind": "text", "text": "Done.", "repair": None},
            ["Find Alice."],
        ),
    ]
    manifest = {
        "protocol_id": "test",
        "rows": [
            {
                "episode_id": "interactive",
                "agent_interface": _interface(),
            }
        ],
    }
    episodes = [
        {
            "episode_id": "interactive",
            "parent_episode_id": "parent",
            "task_key": "task",
            "domain": "retail",
            "split": "training",
            "llm_seed": 401,
            "agent_decisions": decisions,
            "combined_tool_events": [_assistant_event()],
            "termination": "user_stop",
            "natural_user_message_count": 1,
        }
    ]
    dataset, audit = build_interactive_dataset(manifest, episodes)
    assert audit["episodes"] == 1
    assert audit["adjacent_transitions"] == 1
    assert audit["task_disjoint"]
    assert audit["causal_label_blind_states"]
    assert dataset["episodes"][0]["parent_episode_id"] == "parent"
    assert dataset["episodes"][0]["transitions"][0]["prefix_index"] == 0


def test_reconstruction_rejects_nonidentical_exact_replay():
    event = _assistant_event()
    event["replica_identical"] = False
    decisions = [
        _decision(
            0,
            {
                "kind": "tool_call",
                "name": "lookup",
                "arguments": {"record_id": "alice"},
                "repair": "strict",
            },
            ["Find Alice."],
        )
    ]
    try:
        reconstruct_agent_surface(
            episode_id="episode",
            domain="retail",
            agent_interface=_interface(),
            agent_decisions=decisions,
            combined_tool_events=[copy.deepcopy(event)],
        )
    except ValueError as error:
        assert "replicas differ" in str(error)
    else:
        raise AssertionError("non-identical replay was accepted")
