"""Tests for grace.io.loaders.load_track2 (synthetic fixtures)."""

from grace.io.loaders import load_track2
from tests.conftest import TRACK2_FIXTURE


def test_load_track2_returns_all_cases() -> None:
    assert len(load_track2(TRACK2_FIXTURE)) == 2


def test_load_track2_correct_choice_id_is_valid_when_present() -> None:
    for case in load_track2(TRACK2_FIXTURE):
        if case.correct_choice_id is not None:
            assert case.correct_choice_id in {c.id for c in case.choices}


def test_load_track2_sentence_relevancy_matches_sentences() -> None:
    for case in load_track2(TRACK2_FIXTURE):
        assert len(case.sentence_relevancy) == len(case.context_sentences)


def test_load_track2_sentence_relevancy_labels_are_valid() -> None:
    for case in load_track2(TRACK2_FIXTURE):
        for lbl in case.sentence_relevancy:
            assert lbl in {"relevant", "not-relevant"}


def test_load_track2_choices_are_loaded() -> None:
    cases = load_track2(TRACK2_FIXTURE)
    # First fixture case has choices, second has none — both must round-trip.
    assert len(cases[0].choices) == 2
    assert len(cases[1].choices) == 0


def test_load_track2_entity_substrings_are_exact() -> None:
    for case in load_track2(TRACK2_FIXTURE):
        for e in case.entities:
            assert case.raw_text[e.start : e.end] == e.text


def test_load_track2_relations_reference_known_entities() -> None:
    for case in load_track2(TRACK2_FIXTURE):
        ent_ids = {e.id for e in case.entities}
        for r in case.relations:
            assert r.arg1_id in ent_ids and r.arg2_id in ent_ids


def test_load_track2_case_track_field_is_2() -> None:
    assert all(c.track == 2 for c in load_track2(TRACK2_FIXTURE))
