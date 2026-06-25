from wmagentattack.normalize_agentdojo import normalize_directory


def test_normalize_directory_skips_summary_files(tmp_path):
    (tmp_path / "batch_summary.json").write_text(
        '{"scope": "summary"}', encoding="utf-8"
    )
    assert normalize_directory(tmp_path) == []

