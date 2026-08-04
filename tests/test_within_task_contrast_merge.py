import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "37_merge_within_task_contrast_replays.py"
    )
    spec = importlib.util.spec_from_file_location("contrast_merge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_merge_builds_five_outcomes_per_pair(tmp_path):
    module = _module()
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"injections": {"location": "injection text"}}),
        encoding="utf-8",
    )
    rows = [
        {
            "suite": "banking",
            "user_task_id": "user_task_1",
            "injection_task_id": f"injection_task_{chunk}",
            "source_trace": str(trace),
            "contrast_task_stratum": "high_score_span",
            "contrast_injection_slot": chunk,
        }
        for chunk in range(4)
    ]
    selection = {
        "selections": {
            "within_task_contrast": rows,
            **{
                f"within_task_contrast_chunk{chunk}": [rows[chunk]]
                for chunk in range(4)
            },
        }
    }
    replay_root = tmp_path / "replays"
    for base_seed in (51, 57, 63, 69, 75):
        for chunk in range(4):
            path = replay_root / f"base_seed{base_seed}" / f"chunk{chunk}"
            path.mkdir(parents=True)
            path.joinpath("replay.json").write_text(
                json.dumps(
                    {
                        "seed": base_seed + 1000 * chunk,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "results": {
                            f"within_task_contrast_chunk{chunk}": {
                                "rows": [
                                    {
                                        **rows[chunk],
                                        "security": base_seed % 2 == 1,
                                        "utility": chunk % 2 == 0,
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
    dataset, summary = module._merge(
        selection,
        replay_root,
        base_seeds=[51, 57, 63, 69, 75],
        chunks=4,
        output_dir=tmp_path / "output",
    )
    assert dataset["pair_count"] == 4
    assert dataset["attempt_count"] == 20
    assert {row["replay_attempt_count"] for row in dataset["pairs"]} == {5}
    assert all(row["injection_text"] == "location: injection text" for row in dataset["pairs"])
    assert summary["task_count"] == 1
    json.dumps(summary)
    assert summary["pairs_per_task"] == [
        {
            "suite": "banking",
            "user_task_id": "user_task_1",
            "pair_count": 4,
        }
    ]
