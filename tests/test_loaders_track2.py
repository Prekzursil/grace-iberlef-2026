"""Tests for grace.io.loaders.load_track2."""

from pathlib import Path

from grace.io.loaders import load_track2

_TRAIN = Path("downloaded_data/public_data/public_data/track_2_train.json")
_DEV = Path("downloaded_data/public_data/public_data/track_2_dev.json")


def test_load_track2_train_returns_128_cases() -> None:
    assert len(load_track2(_TRAIN)) == 128


def test_load_track2_dev_returns_24_cases() -> None:
    assert len(load_track2(_DEV)) == 24


def test_load_track2_has_correct_choice_id_for_every_case() -> None:
    for case in load_track2(_TRAIN):
        assert case.correct_choice_id is not None
        assert case.correct_choice_id in {c.id for c in case.choices}


def test_load_track2_sentence_relevancy_matches_sentences() -> None:
    for case in load_track2(_TRAIN):
        assert len(case.sentence_relevancy) == len(case.context_sentences)


def test_load_track2_sentence_relevancy_labels_are_valid() -> None:
    for case in load_track2(_TRAIN):
        for lbl in case.sentence_relevancy:
            assert lbl in {"relevant", "not-relevant"}


def test_load_track2_variable_choice_count() -> None:
    counts = {len(c.choices) for c in load_track2(_TRAIN)}
    assert counts <= {4, 5}
    assert counts  # at least one choice count present


def test_load_track2_entity_substrings_are_exact() -> None:
    for case in load_track2(_DEV):
        for e in case.entities:
            assert case.raw_text[e.start : e.end] == e.text


def test_load_track2_relations_reference_known_entities() -> None:
    for case in load_track2(_DEV):
        ent_ids = {e.id for e in case.entities}
        for r in case.relations:
            assert r.arg1_id in ent_ids and r.arg2_id in ent_ids


def test_load_track2_case_track_field_is_2() -> None:
    cases = load_track2(_DEV)
    assert all(c.track == 2 for c in cases)
