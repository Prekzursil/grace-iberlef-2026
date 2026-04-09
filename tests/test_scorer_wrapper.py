"""Tests for grace.eval.scorer wrapper."""

import tempfile
from pathlib import Path

from grace.eval.scorer import (
    score_track1_from_file,
    score_track2_from_file,
    scorer_fingerprint,
)
from grace.io.loaders import load_track1, load_track2, save_predictions


def test_scorer_fingerprints_are_stable_strings() -> None:
    fp1 = scorer_fingerprint(1)
    fp2 = scorer_fingerprint(2)
    assert isinstance(fp1, str) and len(fp1) == 64
    assert isinstance(fp2, str) and len(fp2) == 64
    assert fp1 != fp2


def test_scorer_fingerprint_is_deterministic() -> None:
    """Re-computing the fingerprint must return the same value."""
    assert scorer_fingerprint(1) == scorer_fingerprint(1)
    assert scorer_fingerprint(2) == scorer_fingerprint(2)


def test_score_track1_gold_predictions_return_perfect_scores() -> None:
    cases = load_track1(Path("downloaded_data/public_data/public_data/track_1_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gold_preds.json"
        save_predictions(cases, out, track=1)
        results = score_track1_from_file(out)
    assert results["subtask1"]["official_score"] == 1.0
    assert results["subtask2"]["official_score"] == 1.0


def test_score_track2_gold_predictions_return_perfect_scores() -> None:
    cases = load_track2(Path("downloaded_data/public_data/public_data/track_2_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "gold_preds.json"
        save_predictions(cases, out, track=2)
        results = score_track2_from_file(out)
    assert results["subtask1"]["official_score"] == 1.0
    assert results["subtask2"]["official_score"] == 1.0
    assert results["subtask3"]["official_score"] == 1.0
