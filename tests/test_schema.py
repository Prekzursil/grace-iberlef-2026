"""Tests for grace.io.schema dataclasses."""

import pytest

from grace.io.schema import (
    GraceCase,
    GraceChoice,
    GraceEntity,
    GraceRelation,
    GraceSentence,
)


def test_entity_is_frozen() -> None:
    e = GraceEntity(id="T1", text="foo", start=0, end=3, type="Premise")
    with pytest.raises((AttributeError, TypeError)):
        e.text = "bar"  # type: ignore[misc]


def test_case_defaults_are_empty_tuples() -> None:
    case = GraceCase(id="x", raw_text="hello", track=1)
    assert case.entities == ()
    assert case.relations == ()
    assert case.context_sentences == ()
    assert case.choices == ()
    assert case.correct_choice_id is None
    assert case.metadata_context is None


def test_case_with_full_track2_payload() -> None:
    e = GraceEntity(id="e1", text="ab", start=0, end=2, type="Premise")
    r = GraceRelation(id="r1", arg1_id="e1", arg2_id="e1", relation_type="Support")
    s = GraceSentence(sentence="ab.", start=0, end=3)
    ch = GraceChoice(id="1", text="ab", start=0, end=2)
    case = GraceCase(
        id="y",
        raw_text="ab.",
        track=2,
        metadata_context="ab.",
        context_sentences=(s,),
        choices=(ch,),
        correct_choice_id="1",
        sentence_relevancy=("relevant",),
        entities=(e,),
        relations=(r,),
    )
    assert case.track == 2
    assert case.correct_choice_id == "1"
    assert case.entities[0].text == "ab"


def test_case_is_hashable() -> None:
    """Frozen + slots should allow the dataclass to be used in sets/dicts."""
    case = GraceCase(id="x", raw_text="hello", track=1)
    case_set = {case}
    assert case in case_set


def test_entity_equality_by_fields() -> None:
    a = GraceEntity(id="T1", text="foo", start=0, end=3, type="Premise")
    b = GraceEntity(id="T1", text="foo", start=0, end=3, type="Premise")
    c = GraceEntity(id="T1", text="foo", start=0, end=3, type="Claim")
    assert a == b
    assert a != c
