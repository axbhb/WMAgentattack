import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

from wmagentattack.agentdojo_v2 import ManifestPayloadAttack, stable_episode_seed


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentdojo_v2_manifest",
    ROOT / "scripts" / "61_build_agentdojo_v2_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v2_manifest_has_fixed_budget_and_disjoint_tasks():
    protocol = json.loads(
        (ROOT / "configs" / "0714_agentdojo_v2_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = copy.deepcopy(protocol)
    protocol["external_sources"]["autodojo"]["allowed_suites"] = []
    manifest = MODULE.build_manifest(
        protocol,
        solvability={},
        grouped_splits={},
        autodojo_root=None,
    )
    rows = manifest["rows"]
    assert len(rows) == 400
    assert Counter(row["task_split"] for row in rows) == {
        "train": 240,
        "val": 80,
        "test": 80,
    }
    pair_counts = Counter(
        (row["suite"], row["user_task_id"], row["injection_task_id"])
        for row in rows
    )
    assert set(pair_counts.values()) == {5}
    task_sets = {
        split: {
            (row["suite"], row["user_task_id"])
            for row in rows
            if row["task_split"] == split
        }
        for split in ("train", "val", "test")
    }
    assert not task_sets["train"] & task_sets["val"]
    assert not task_sets["train"] & task_sets["test"]
    assert not task_sets["val"] & task_sets["test"]
    assert all(
        row["task_split"] == "test"
        for row in rows
        if row["attack_family"] == "structured_holdout"
    )
    assert manifest["safety_contract"]["allow_real_network_endpoints"] is False
    assert manifest["safety_contract"]["allow_external_side_effects"] is False


def test_manifest_payload_split_is_only_across_synthetic_vectors():
    attack = object.__new__(ManifestPayloadAttack)
    attack.name = "test"
    attack.payload = "full"
    attack.payload_by_vector = {}
    attack.payload_segments = ["bridge", "execute"]
    attack.endpoint_policy = "split"
    attack.get_injection_candidates = lambda _: ["first", "middle", "last"]
    injections = attack.attack(object(), object())
    assert injections == {
        "first": "bridge",
        "middle": "bridge",
        "last": "execute",
    }


def test_episode_seed_is_stable_and_row_specific():
    assert stable_episode_seed(7, "row-a") == stable_episode_seed(7, "row-a")
    assert stable_episode_seed(7, "row-a") != stable_episode_seed(7, "row-b")
