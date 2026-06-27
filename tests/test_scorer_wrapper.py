"""Tests for the grace.eval.scorer wrapper.

The real organizer scoring programs are non-redistributable, so these tests
point the wrapper at the in-repo fake scorer (see ``tests/fixtures/fake_scorer.py``)
via the ``fake_scorer_paths`` fixture. This exercises ``_load``,
``scorer_fingerprint`` and the ``score_track{1,2}_from_file`` entry points
end to end without redistributing organizer code.
"""

from pathlib import Path

from grace.eval.scorer import (
    score_track1_from_file,
    score_track2_from_file,
    scorer_fingerprint,
)
from grace.io.loaders import save_predictions


def test_scorer_fingerprint_is_64_hex_and_deterministic(fake_scorer_paths: Path) -> None:
    fp1 = scorer_fingerprint(1)
    fp2 = scorer_fingerprint(2)
    for fp in (fp1, fp2):
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)
    assert scorer_fingerprint(1) == fp1  # deterministic


def test_score_track1_from_file_returns_scores(
    fake_scorer_paths: Path, track1_cases, tmp_path: Path
) -> None:
    out = tmp_path / "gold_preds.json"
    save_predictions(track1_cases, out, track=1)
    results = score_track1_from_file(out)
    assert results["subtask1"]["official_score"] == 1.0
    assert results["subtask2"]["official_score"] == 1.0


def test_score_track2_from_file_returns_scores(
    fake_scorer_paths: Path, track2_cases, tmp_path: Path
) -> None:
    out = tmp_path / "gold_preds.json"
    save_predictions(track2_cases, out, track=2)
    results = score_track2_from_file(out)
    assert results["subtask1"]["official_score"] == 1.0
    assert results["subtask2"]["official_score"] == 1.0
    assert results["subtask3"]["official_score"] == 1.0
