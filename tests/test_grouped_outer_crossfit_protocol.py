import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_outer_crossfit_protocol_freezes_fold_hash_and_budget():
    protocol = json.loads(
        (ROOT / "configs" / "0713_grouped_outer_crossfit_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["folds"]["manifest_sha256"] == (
        "da58322b18913aebcfef7558255c74ff81d4117a4b5fdd5172900e8aaa540cf4"
    )
    assert protocol["research_budget"]["checkpoint_count"] == 12
    assert protocol["research_budget"]["llm_replay_outcome_budget"] == 0
    assert protocol["folds"]["every_task_held_exactly_once"]


def test_outer_crossfit_finalizer_accepts_empty_done_sentinels():
    script = (
        ROOT
        / "scripts"
        / "server"
        / "finalize_grouped_outer_crossfit_if_complete.sh"
    ).read_text(encoding="utf-8")
    assert '[[ ! -f "$ARCHIVE/scores/fold${fold}/seed${seed}/DONE" ]]' in script
    assert '[[ ! -s "$ARCHIVE/scores/fold${fold}/seed${seed}/DONE" ]]' not in script
