import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "124_diagnose_clean_evidence_counterevidence.py"
SPEC = importlib.util.spec_from_file_location("clean_evidence_counterevidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_field_and_span_parsing_are_explicit():
    line = (
        "entity UNLINKED attribute price value 180 source tool arguments {} "
        "status success link AMBIGUOUS goal_overlap 0.5 novelty new "
        "conflict conflict provenance read_only"
    )
    assert MODULE._span(line, "entity ", " attribute ") == "UNLINKED"
    assert MODULE._span(line, " attribute ", " value ") == "price"
    assert MODULE._field(line, "conflict") == "conflict"


def test_donors_are_within_task_and_cyclic():
    episodes = [
        {"episode_id": "a1", "task_id": "a"},
        {"episode_id": "a2", "task_id": "a"},
        {"episode_id": "b1", "task_id": "b"},
        {"episode_id": "b2", "task_id": "b"},
    ]
    assert MODULE._donors(episodes) == {
        "a1": "a2",
        "a2": "a1",
        "b1": "b2",
        "b2": "b1",
    }
