import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_replication_is_auxiliary_fixed_budget_and_single_gpu():
    protocol = json.loads((ROOT / "configs" / "0811_multisource_replication_protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "manifests_frozen_before_replication_outcomes"
    assert protocol["scientific_scope"]["does_not_overturn_current_method_no_go"]
    assert protocol["scientific_scope"]["does_not_increase_independent_task_count"]
    assert protocol["fixed_budget"]["maximum_concurrent_gpus"] == 1
    assert protocol["sources"]["tool_sandbox"]["replication_expected_rows"] == 190
    assert protocol["sources"]["injecagent"]["replication_expected_rows"] == 4216
    assert protocol["fixed_budget"]["total_llm_decisions"] == 4406
    assert all(protocol["frozen_manifests"]["replication"].values())
    assert protocol["implementation_commit"]
