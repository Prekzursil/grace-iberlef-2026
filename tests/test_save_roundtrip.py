"""Round-trip tests for load <-> save (synthetic fixtures)."""

import json
from pathlib import Path

from grace.io.loaders import load_track1, load_track2, save_predictions
from tests.conftest import TRACK1_FIXTURE, TRACK2_FIXTURE


def test_track1_roundtrip_preserves_data(tmp_path: Path) -> None:
    orig = load_track1(TRACK1_FIXTURE)
    out = tmp_path / "round.json"
    save_predictions(orig, out, track=1)
    again = load_track1(out)
    assert len(again) == len(orig)
    for o, a in zip(orig, again, strict=True):
        assert o == a


def test_track2_roundtrip_preserves_data(tmp_path: Path) -> None:
    orig = load_track2(TRACK2_FIXTURE)
    out = tmp_path / "round.json"
    save_predictions(orig, out, track=2)
    again = load_track2(out)
    assert len(again) == len(orig)
    for o, a in zip(orig, again, strict=True):
        assert o == a


def test_save_predictions_accepts_list_input(tmp_path: Path) -> None:
    """save_predictions accepts a plain list, not only a tuple."""
    orig = list(load_track1(TRACK1_FIXTURE))
    out = tmp_path / "list.json"
    save_predictions(orig, out, track=1)
    assert len(load_track1(out)) == len(orig)


def test_save_predictions_rejects_unknown_track(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown track"):
        save_predictions(load_track1(TRACK1_FIXTURE), tmp_path / "x.json", track=7)


def test_track1_saved_file_shape_matches_scorer_expectations(tmp_path: Path) -> None:
    """Scorer expects annotations as a dict with entities/relations keys."""
    cases = load_track1(TRACK1_FIXTURE)
    out = tmp_path / "preds.json"
    save_predictions(cases, out, track=1)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert isinstance(data[0]["annotations"], dict)
    assert "entities" in data[0]["annotations"]
    assert "relations" in data[0]["annotations"]


def test_track2_saved_file_shape_matches_scorer_expectations(tmp_path: Path) -> None:
    cases = load_track2(TRACK2_FIXTURE)
    out = tmp_path / "preds.json"
    save_predictions(cases, out, track=2)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert isinstance(data[0]["annotations"], dict)
    assert "sentence_relevancy" in data[0]["annotations"]
    assert "entities" in data[0]["annotations"]
    assert "relations" in data[0]["annotations"]
    assert "metadata" in data[0]
    assert "context_sentences" in data[0]["metadata"]
    assert "choices" in data[0]["metadata"]
    assert "correct_choice_id" in data[0]["metadata"]
