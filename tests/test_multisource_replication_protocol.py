import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = ROOT / "scripts" / "177_build_multisource_replication_manifests.py"
    spec = importlib.util.spec_from_file_location("replication_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replication_is_auxiliary_fixed_budget_and_single_gpu():
    protocol = json.loads((ROOT / "configs" / "0811_multisource_replication_protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "completed_after_preregistered_measurement_recovery"
    assert protocol["result"]["passed"] is True
    assert protocol["result"]["rows"] == 4406
    assert protocol["scientific_scope"]["does_not_overturn_current_method_no_go"]
    assert protocol["scientific_scope"]["does_not_increase_independent_task_count"]
    assert protocol["fixed_budget"]["maximum_concurrent_gpus"] == 1
    assert protocol["sources"]["tool_sandbox"]["replication_expected_rows"] == 190
    assert protocol["sources"]["injecagent"]["replication_expected_rows"] == 4216
    assert protocol["fixed_budget"]["total_llm_decisions"] == 4406
    assert all(protocol["frozen_manifests"]["replication"].values())
    assert protocol["implementation_commit"]


def test_injecagent_pairs_stay_inside_modulo_chunks():
    parent_rows = []
    for group in range(4):
        for variant in ("clean", "poisoned"):
            parent_rows.append(
                {
                    "row_id": f"injecagent::family::{group}::{variant}::seed307",
                    "group_id": f"injecagent::family::{group}",
                    "variant": variant,
                    "run_seed": 307,
                }
            )
    parent = {
        "schema_version": "v1",
        "source_commit": "commit",
        "llm_contract_sha256": "contract",
        "execution_preflight": {"passed": True},
        "rows": parent_rows,
    }
    protocol = {
        "protocol_id": "test",
        "sources": {
            "injecagent": {
                "replication_seeds": [311, 313],
                "replication_expected_rows": 16,
                "generation_chunks": 4,
            }
        },
    }
    manifest, audit = _builder().replicate(parent, protocol, "injecagent")
    assert audit["passed"]
    positions = {}
    for index, row in enumerate(manifest["rows"]):
        positions.setdefault((row["group_id"], row["run_seed"]), set()).add(index % 4)
    assert all(len(value) == 1 for value in positions.values())
