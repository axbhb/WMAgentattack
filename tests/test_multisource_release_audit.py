from wmagentattack.multisource_release_audit import audit_sharded_output
from wmagentattack.multisource_semantic_data import build_model_input, stable_hash


def _row(variant: str) -> dict:
    contract_hash = stable_hash({"model": "same"})
    return {
        "row_id": f"injec::{variant}",
        "source": "injecagent",
        "group_id": "injec::pair",
        "variant": variant,
        "llm_contract_sha256": contract_hash,
        "model_input": build_model_input(
            trusted_goal="Read",
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        "completion": "Done",
        "decision": {"kind": "text"},
        "runtime_error": None,
        "execution": {
            "tier": "observation_only",
            "real_external_endpoint_calls": 0,
        },
    }


def test_cross_shard_pair_failure_is_deferred_without_rewriting_output():
    clean = _row("clean")
    poisoned = _row("poisoned")
    manifest = {"source": "injecagent", "rows": [clean, poisoned]}
    audit = audit_sharded_output(
        manifest=manifest,
        protocol={"shared_llm_contract": {"model": "same"}},
        output_payload={"complete": True, "records": [clean]},
        original_audit={
            "checks": {
                "expected_outputs": True,
                "zero_runtime_failures": True,
                "injecagent_pair_completeness": False,
            }
        },
        chunk_index=0,
        num_chunks=2,
    )
    assert audit["passed"]
    assert audit["records_regenerated"] == 0
    assert audit["outputs_overwritten"] is False
    assert audit["local_incomplete_pair_groups"] == 1


def test_reaudit_rejects_any_non_pair_failure():
    clean = _row("clean")
    manifest = {"source": "injecagent", "rows": [clean]}
    audit = audit_sharded_output(
        manifest=manifest,
        protocol={"shared_llm_contract": {"model": "same"}},
        output_payload={"complete": True, "records": [clean]},
        original_audit={
            "checks": {
                "zero_runtime_failures": False,
                "injecagent_pair_completeness": False,
            }
        },
        chunk_index=0,
        num_chunks=2,
    )
    assert not audit["passed"]
