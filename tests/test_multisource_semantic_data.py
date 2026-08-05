from wmagentattack.multisource_semantic_data import (
    MANIFEST_SCHEMA_VERSION,
    audit_manifest,
    build_model_input,
    injecagent_tool_schema,
    parse_function_tag_completion,
    stable_hash,
    summarize_generation,
)


def _tool(name="lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Lookup a record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }


def test_injecagent_schema_conversion_and_shared_input_sorting():
    converted = injecagent_tool_schema(
        {
            "name": "GmailSendEmail",
            "summary": "Send mail.",
            "parameters": [
                {
                    "name": "to",
                    "type": "string",
                    "description": "Recipient",
                    "required": True,
                },
                {
                    "name": "attachments",
                    "type": "array",
                    "description": "Paths",
                    "required": False,
                },
            ],
        }
    )
    assert converted["function"]["parameters"]["required"] == ["to"]
    assert converted["function"]["parameters"]["properties"]["attachments"][
        "items"
    ] == {"type": "string"}
    model_input = build_model_input(
        trusted_goal="Find it", tool_schemas=[_tool("zeta"), _tool("alpha")]
    )
    assert [row["function"]["name"] for row in model_input["tool_schemas"]] == [
        "alpha",
        "zeta",
    ]


def test_parser_accepts_only_presented_unambiguous_tools():
    strict = parse_function_tag_completion(
        '<function=lookup>{"record_id":"a"}</function>', {"lookup"}
    )
    assert strict == {
        "kind": "tool_call",
        "name": "lookup",
        "arguments": {"record_id": "a"},
        "repair": "strict",
    }
    repaired = parse_function_tag_completion(
        '{"function":"lookup","parameters":{"record_id":"b"}}', {"lookup"}
    )
    assert repaired["kind"] == "tool_call"
    rejected = parse_function_tag_completion(
        '<function=delete_everything>{}</function>', {"lookup"}
    )
    assert rejected["kind"] == "text"


def test_manifest_gate_rejects_label_leakage_and_contract_drift():
    contract = stable_hash({"model": "same"})
    row = {
        "row_id": "tool_sandbox::a::clean::seed1",
        "source": "tool_sandbox",
        "llm_contract_sha256": contract,
        "model_input": build_model_input(
            trusted_goal="Find it", tool_schemas=[_tool()]
        ),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "real_external_endpoint_calls": 0,
        "rows": [row],
    }
    assert audit_manifest(
        manifest, expected_rows=1, expected_llm_contract_sha256=contract
    )["passed"]

    leaked = {**row, "model_input": {**row["model_input"], "reward": 1}}
    manifest["rows"] = [leaked]
    audit = audit_manifest(
        manifest, expected_rows=1, expected_llm_contract_sha256=contract
    )
    assert not audit["passed"]
    assert audit["leaked_model_input_keys"][row["row_id"]] == ["reward"]


def test_generation_gate_keeps_text_answers_and_requires_exact_replicas():
    base = {
        "source": "tool_sandbox",
        "group_id": "tool_sandbox::a",
        "variant": "clean",
        "model_input": build_model_input(
            trusted_goal="Find it", tool_schemas=[_tool()]
        ),
        "completion": "I need more information.",
        "decision": {"kind": "text"},
        "runtime_error": None,
        "execution": {"tier": "not_executed_text_response"},
    }
    assert summarize_generation(
        [base], expected_rows=1, require_exact_replica_determinism=True
    )["passed"]

    exact = {
        **base,
        "completion": '<function=lookup>{"record_id":"a"}</function>',
        "decision": {
            "kind": "tool_call",
            "name": "lookup",
            "arguments": {"record_id": "a"},
        },
        "execution": {"tier": "exact", "replica_identical": False},
    }
    assert not summarize_generation(
        [exact], expected_rows=1, require_exact_replica_determinism=True
    )["passed"]


def test_injecagent_pair_completeness_is_a_gate():
    clean = {
        "source": "injecagent",
        "group_id": "injecagent::case",
        "variant": "clean",
        "model_input": build_model_input(trusted_goal="Read", tool_schemas=[_tool()]),
        "completion": "Done",
        "decision": {"kind": "text"},
        "runtime_error": None,
        "execution": {"tier": "observation_only"},
    }
    assert not summarize_generation(
        [clean], expected_rows=1, require_exact_replica_determinism=True
    )["passed"]
    poisoned = {**clean, "variant": "poisoned", "completion": "Still done"}
    assert summarize_generation(
        [clean, poisoned], expected_rows=2, require_exact_replica_determinism=True
    )["passed"]
