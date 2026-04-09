"""Tests for grace.submit.formatter and validator."""

import tempfile
from pathlib import Path

import pytest

from grace.io.loaders import load_track1, load_track2
from grace.submit.formatter import format_submission
from grace.submit.validator import SubmissionValidationError, validate_submission


def test_format_track1_submission_passes_validator() -> None:
    cases = load_track1(Path("downloaded_data/public_data/public_data/track_1_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub.json"
        format_submission(cases, out, track=1)
        # Self-consistency mode (gold_path=None): scorer uses annotations as
        # predictions when the predictions block is empty, so identical saves
        # score 1.0.
        scores = validate_submission(out, track=1)
    assert scores["subtask1"] == 1.0
    assert scores["subtask2"] == 1.0


def test_format_track2_submission_passes_validator() -> None:
    cases = load_track2(Path("downloaded_data/public_data/public_data/track_2_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub.json"
        format_submission(cases, out, track=2)
        scores = validate_submission(out, track=2)
    assert scores["subtask1"] == 1.0
    assert scores["subtask2"] == 1.0
    assert scores["subtask3"] == 1.0


def test_validator_rejects_empty_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "empty.json"
        out.write_text("[]", encoding="utf-8")
        with pytest.raises(SubmissionValidationError):
            validate_submission(out, track=1)


def test_validator_rejects_non_array() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "obj.json"
        out.write_text('{"not": "an array"}', encoding="utf-8")
        with pytest.raises(SubmissionValidationError):
            validate_submission(out, track=1)


def test_validator_rejects_unknown_track() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub.json"
        out.write_text('[{"id": "x"}]', encoding="utf-8")
        with pytest.raises(SubmissionValidationError):
            validate_submission(out, track=99)
