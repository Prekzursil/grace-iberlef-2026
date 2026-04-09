"""Tests for grace.io.loaders.load_track1."""

from pathlib import Path

from grace.io.loaders import load_track1

_TRAIN = Path("downloaded_data/public_data/public_data/track_1_train.json")
_DEV = Path("downloaded_data/public_data/public_data/track_1_dev.json")


def test_load_track1_train_returns_350_cases() -> None:
    assert len(load_track1(_TRAIN)) == 350


def test_load_track1_dev_returns_50_cases() -> None:
    assert len(load_track1(_DEV)) == 50


def test_load_track1_entity_substrings_are_exact() -> None:
    """For every gold entity, raw_text[start:end] must equal entity.text."""
    for case in load_track1(_DEV):
        for e in case.entities:
            assert case.raw_text[e.start : e.end] == e.text, (
                f"substring mismatch in case {case.id} entity {e.id}: "
                f"gold={e.text!r}, slice={case.raw_text[e.start : e.end]!r}"
            )


def test_load_track1_relations_reference_known_entities() -> None:
    for case in load_track1(_DEV):
        ent_ids = {e.id for e in case.entities}
        for r in case.relations:
            assert r.arg1_id in ent_ids and r.arg2_id in ent_ids


def test_load_track1_entity_types_are_in_vocabulary() -> None:
    cases = load_track1(_TRAIN)
    seen = {e.type for c in cases for e in c.entities}
    assert seen <= {"Premise", "Claim", "MajorClaim"}


def test_load_track1_relation_types_are_in_vocabulary() -> None:
    cases = load_track1(_TRAIN)
    seen = {r.relation_type for c in cases for r in c.relations}
    assert seen <= {"Support", "Attack", "Partial-Attack"}


def test_load_track1_case_track_field_is_1() -> None:
    cases = load_track1(_DEV)
    assert all(c.track == 1 for c in cases)


def test_load_track1_track2_fields_are_empty() -> None:
    """Track 1 cases should have no Track 2-specific fields set."""
    cases = load_track1(_DEV)
    for c in cases:
        assert c.context_sentences == ()
        assert c.choices == ()
        assert c.correct_choice_id is None
        assert c.sentence_relevancy == ()
