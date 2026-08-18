import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "partition_v14", ROOT / "scripts" / "229_build_event_graph_partitions_v14.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_partition_is_exclusive():
    assert MODULE._partition("action.skill=x", ["action."], ["receipt."]) == "exact"
    assert MODULE._partition("receipt.format=x", ["action."], ["receipt."]) == "evidence"


def test_partition_rejects_uncovered_and_overlapping_features():
    with pytest.raises(ValueError):
        MODULE._partition("other=x", ["action."], ["receipt."])
    with pytest.raises(ValueError):
        MODULE._partition("action.x", ["action."], ["action.x"])
