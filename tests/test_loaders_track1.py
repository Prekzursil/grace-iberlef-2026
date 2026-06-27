"""Tests for grace.io.loaders.load_track1 (synthetic fixtures).

The organizer competition data is non-redistributable; these tests use the
committed synthetic fixture instead, exercising the same loader invariants.
"""

from grace.io.loaders import load_track1
from tests.conftest import TRACK1_FIXTURE


def test_load_track1_returns_all_cases() -> None:
    assert len(load_track1(TRACK1_FIXTURE)) == 2


def test_load_track1_entity_substrings_are_exact() -> None:
    """For every gold entity, raw_text[start:end] must equal entity.text."""
    for case in load_track1(TRACK1_FIXTURE):
        for e in case.entities:
            assert case.raw_text[e.start : e.end] == e.text, (
                f"substring mismatch in case {case.id} entity {e.id}: "
                f"gold={e.text!r}, slice={case.raw_text[e.start : e.end]!r}"
            )


def test_load_track1_relations_reference_known_entities() -> None:
    """Dangling relations (arg ids absent from entities) must be dropped."""
    for case in load_track1(TRACK1_FIXTURE):
        ent_ids = {e.id for e in case.entities}
        for r in case.relations:
            assert r.arg1_id in ent_ids and r.arg2_id in ent_ids


def test_load_track1_entity_types_are_in_vocabulary() -> None:
    seen = {e.type for c in load_track1(TRACK1_FIXTURE) for e in c.entities}
    assert seen <= {"Premise", "Claim", "MajorClaim"}


def test_load_track1_relation_types_are_in_vocabulary() -> None:
    seen = {r.relation_type for c in load_track1(TRACK1_FIXTURE) for r in c.relations}
    assert seen <= {"Support", "Attack", "Partial-Attack"}


def test_load_track1_case_track_field_is_1() -> None:
    assert all(c.track == 1 for c in load_track1(TRACK1_FIXTURE))


def test_load_track1_track2_fields_are_empty() -> None:
    """Track 1 cases should have no Track 2-specific fields set."""
    for c in load_track1(TRACK1_FIXTURE):
        assert c.context_sentences == ()
        assert c.choices == ()
        assert c.correct_choice_id is None
        assert c.sentence_relevancy == ()
