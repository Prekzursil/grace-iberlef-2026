"""Error-path coverage for grace.io.loaders validation."""

import json
from pathlib import Path

import pytest

from grace.io.loaders import GraceLoadError, load_track1, load_track2


def _write(tmp_path: Path, payload) -> Path:
    p = tmp_path / "case.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_track1_top_level_must_be_array(tmp_path: Path) -> None:
    with pytest.raises(GraceLoadError, match="JSON array"):
        load_track1(_write(tmp_path, {"not": "a list"}))


def test_track1_annotations_must_be_dict(tmp_path: Path) -> None:
    payload = [{"id": "c1", "raw_text": "hi", "annotations": [1, 2, 3]}]
    with pytest.raises(GraceLoadError, match="annotations must be a dict"):
        load_track1(_write(tmp_path, payload))


def test_track1_entity_offsets_out_of_range(tmp_path: Path) -> None:
    payload = [
        {
            "id": "c1",
            "raw_text": "short",
            "annotations": {
                "entities": [{"id": "T1", "text": "x", "start": 0, "end": 99, "type": "Premise"}],
                "relations": [],
            },
        }
    ]
    with pytest.raises(GraceLoadError, match="offsets out of range"):
        load_track1(_write(tmp_path, payload))


def test_track1_entity_substring_mismatch(tmp_path: Path) -> None:
    payload = [
        {
            "id": "c1",
            "raw_text": "hello",
            "annotations": {
                "entities": [{"id": "T1", "text": "zz", "start": 0, "end": 2, "type": "Premise"}],
                "relations": [],
            },
        }
    ]
    with pytest.raises(GraceLoadError, match="does not match raw_text slice"):
        load_track1(_write(tmp_path, payload))


def test_track2_top_level_must_be_array(tmp_path: Path) -> None:
    with pytest.raises(GraceLoadError, match="JSON array"):
        load_track2(_write(tmp_path, {"not": "a list"}))


def test_track2_annotations_must_be_dict(tmp_path: Path) -> None:
    payload = [{"id": "c1", "raw_text": "hi", "metadata": {}, "annotations": [1]}]
    with pytest.raises(GraceLoadError, match="annotations must be a dict"):
        load_track2(_write(tmp_path, payload))


def test_track2_relevancy_length_mismatch(tmp_path: Path) -> None:
    payload = [
        {
            "id": "c1",
            "raw_text": "hi",
            "metadata": {"context_sentences": [{"sentence": "hi", "start": 0, "end": 2}]},
            "annotations": {"sentence_relevancy": [], "entities": [], "relations": []},
        }
    ]
    with pytest.raises(GraceLoadError, match="sentence_relevancy length"):
        load_track2(_write(tmp_path, payload))


def test_track2_correct_choice_not_in_choices(tmp_path: Path) -> None:
    payload = [
        {
            "id": "c1",
            "raw_text": "hi",
            "metadata": {
                "context_sentences": [],
                "choices": [{"id": "A", "text": "x", "start": 0, "end": 0}],
                "correct_choice_id": "ZZ",
            },
            "annotations": {"sentence_relevancy": [], "entities": [], "relations": []},
        }
    ]
    with pytest.raises(GraceLoadError, match="not in choices"):
        load_track2(_write(tmp_path, payload))
