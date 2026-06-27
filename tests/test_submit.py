"""Tests for grace.submit.formatter and validator (synthetic fixtures + fake scorer)."""

import json
from pathlib import Path

import pytest

from grace.submit.formatter import format_predictions, format_submission
from grace.submit.validator import SubmissionValidationError, validate_submission


def test_format_track1_submission_passes_validator(
    fake_scorer_paths: Path, track1_cases, tmp_path: Path
) -> None:
    out = tmp_path / "sub.json"
    format_submission(track1_cases, out, track=1)
    scores = validate_submission(out, track=1)
    assert scores["subtask1"] == 1.0
    assert scores["subtask2"] == 1.0
    # Non-conforming scorer entries are filtered out of the returned dict.
    assert "scorer_version" not in scores
    assert "details" not in scores


def test_format_track2_submission_passes_validator(
    fake_scorer_paths: Path, track2_cases, tmp_path: Path
) -> None:
    out = tmp_path / "sub.json"
    format_submission(track2_cases, out, track=2)
    scores = validate_submission(out, track=2)
    assert scores["subtask1"] == 1.0
    assert scores["subtask3"] == 1.0


def test_format_predictions_track1_writes_predictions_block(track1_cases, tmp_path: Path) -> None:
    out = tmp_path / "preds.json"
    format_predictions(track1_cases, out, track=1)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "predictions" in data[0]
    assert "entities" in data[0]["predictions"]
    assert "relations" in data[0]["predictions"]
    # Track 1 must not carry track-2-only blocks.
    assert "sentence_relevancy" not in data[0]["predictions"]
    assert "metadata" not in data[0]


def test_format_predictions_track2_includes_relevancy_and_metadata(
    track2_cases, tmp_path: Path
) -> None:
    out = tmp_path / "preds.json"
    format_predictions(track2_cases, out, track=2)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "sentence_relevancy" in data[0]["predictions"]
    assert "metadata" in data[0]
    assert "context_sentences" in data[0]["metadata"]
    assert "choices" in data[0]["metadata"]


def test_validator_rejects_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "empty.json"
    out.write_text("[]", encoding="utf-8")
    with pytest.raises(SubmissionValidationError, match="empty"):
        validate_submission(out, track=1)


def test_validator_rejects_non_array(tmp_path: Path) -> None:
    out = tmp_path / "obj.json"
    out.write_text('{"not": "an array"}', encoding="utf-8")
    with pytest.raises(SubmissionValidationError, match="JSON array"):
        validate_submission(out, track=1)


def test_validator_rejects_unknown_track(fake_scorer_paths: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub.json"
    out.write_text('[{"id": "x"}]', encoding="utf-8")
    with pytest.raises(SubmissionValidationError, match="unknown track"):
        validate_submission(out, track=99)


def test_validator_wraps_scorer_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-validation exception from the scorer is wrapped, not leaked."""
    import grace.submit.validator as validator_mod

    def boom(*_a, **_k):
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr(validator_mod, "score_track1_from_file", boom)
    out = tmp_path / "sub.json"
    out.write_text('[{"id": "x"}]', encoding="utf-8")
    with pytest.raises(SubmissionValidationError, match="scorer crashed"):
        validate_submission(out, track=1)
