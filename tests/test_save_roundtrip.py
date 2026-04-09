"""Round-trip tests for load ↔ save."""

import json
import tempfile
from pathlib import Path

from grace.io.loaders import load_track1, load_track2, save_predictions


def test_track1_roundtrip_preserves_data() -> None:
    orig = load_track1(Path("downloaded_data/public_data/public_data/track_1_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "round.json"
        save_predictions(orig, out, track=1)
        again = load_track1(out)
    assert len(again) == len(orig)
    for o, a in zip(orig, again, strict=False):
        assert o == a


def test_track2_roundtrip_preserves_data() -> None:
    orig = load_track2(Path("downloaded_data/public_data/public_data/track_2_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "round.json"
        save_predictions(orig, out, track=2)
        again = load_track2(out)
    assert len(again) == len(orig)
    for o, a in zip(orig, again, strict=False):
        assert o == a


def test_track1_saved_file_shape_matches_scorer_expectations() -> None:
    """Scorer expects annotations as a dict with entities/relations keys."""
    cases = load_track1(Path("downloaded_data/public_data/public_data/track_1_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "preds.json"
        save_predictions(cases, out, track=1)
        data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert isinstance(data[0]["annotations"], dict)
    assert "entities" in data[0]["annotations"]
    assert "relations" in data[0]["annotations"]


def test_track2_saved_file_shape_matches_scorer_expectations() -> None:
    cases = load_track2(Path("downloaded_data/public_data/public_data/track_2_dev.json"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "preds.json"
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
