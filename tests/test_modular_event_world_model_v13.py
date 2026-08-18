import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SPEC=importlib.util.spec_from_file_location("v13",ROOT/"scripts"/"226_build_modular_event_world_model_v13.py")
MODULE=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(MODULE)


def test_pair_key_excludes_arm_but_keeps_experiment_identity():
    row={"fold":1,"training_seed":7,"horizon":3,"event_id":"e","arm":"x"}
    assert MODULE._key(row)==(1,7,3,"e")


def test_sources_are_disjoint_field_groups():
    assert set(MODULE.ACTION_FIELDS).isdisjoint(MODULE.JOINT_FIELDS)
