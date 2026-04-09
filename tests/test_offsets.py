"""Tests for grace.io.offsets.SpanAligner - snap + validate invariants."""

import pytest

from grace.io.offsets import AlignmentError, SpanAligner
from grace.io.schema import GraceEntity


def test_snap_to_token_boundary_exact_match_is_identity() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    # "cáncer" runs from offset 3 to 9 inclusive/exclusive
    assert a.snap_to_token_boundary(text, 3, 9) == (3, 9)


def test_snap_to_token_boundary_extends_ragged_end() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    # end lands mid-token ("cánc") — should extend to end of "cáncer" (offset 9)
    assert a.snap_to_token_boundary(text, 3, 7) == (3, 9)


def test_snap_is_idempotent() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    once = a.snap_to_token_boundary(text, 2, 7)
    twice = a.snap_to_token_boundary(text, once[0], once[1])
    assert once == twice


def test_snap_empty_span_is_noop() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    assert a.snap_to_token_boundary(text, 5, 5) == (5, 5)


def test_validate_round_trip_passes_exact_substrings() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    entities = (GraceEntity(id="T1", text="cáncer", start=3, end=9, type="Premise"),)
    a.validate_round_trip(text, entities)  # should not raise


def test_validate_round_trip_raises_on_mismatch() -> None:
    a = SpanAligner.without_hf()
    text = "El cáncer de mama."
    # wrong text (missing accent)
    entities = (GraceEntity(id="T1", text="cancer", start=3, end=9, type="Premise"),)
    with pytest.raises(AlignmentError):
        a.validate_round_trip(text, entities)
