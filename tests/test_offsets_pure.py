"""Pure (no-HF) coverage for grace.io.offsets.SpanAligner snap/decode paths."""

import pytest

from grace.io.offsets import _ID_TO_LABEL, SpanAligner


def _label_id(name: str) -> int:
    for k, v in _ID_TO_LABEL.items():
        if v == name:
            return k
    raise AssertionError(name)


def test_snap_empty_text_returns_inputs() -> None:
    a = SpanAligner.without_hf()
    # whitespace-only -> no tokens -> returns inputs unchanged
    assert a.snap_to_token_boundary("   ", 0, 2) == (0, 2)


def test_snap_start_inside_token_extends_left() -> None:
    a = SpanAligner.without_hf()
    text = "El cancer de mama"
    # start=5 falls inside the token 'cancer' (3..9) -> snaps left to 3
    assert a.snap_to_token_boundary(text, 5, 9)[0] == 3


def test_snap_beyond_all_tokens_keeps_inputs() -> None:
    a = SpanAligner.without_hf()
    text = "ab cd"
    # start/end land past every token boundary -> both loops fall through
    assert a.snap_to_token_boundary(text, 6, 7) == (6, 7)


def test_encode_with_labels_requires_hf() -> None:
    a = SpanAligner.without_hf()
    with pytest.raises(RuntimeError, match="requires hf_tokenizer"):
        a.encode_with_labels("hello", [])


def test_decode_orphan_i_label_starts_entity() -> None:
    """An I- label with no matching open entity opens a fresh span (lines 195-198)."""
    a = SpanAligner.without_hf()
    ents = a.decode_bio_to_entities("hello", [_label_id("I-Claim")], [(0, 5)])
    assert len(ents) == 1
    assert ents[0].type == "Claim"
    assert ents[0].text == "hello"


def test_decode_zero_width_span_is_dropped() -> None:
    """A B- token whose offsets are zero-width snaps to s>=t and is dropped (165->176)."""
    a = SpanAligner.without_hf()
    ents = a.decode_bio_to_entities("hello world", [_label_id("B-Premise")], [(3, 3)])
    assert ents == ()


def test_decode_skips_zero_offset_tokens() -> None:
    """A (0, 0) special-token offset in the stream is skipped (line 186)."""
    a = SpanAligner.without_hf()
    text = "El cancer"
    labels = [_label_id("O"), _label_id("B-Premise")]
    offsets = [(0, 0), (3, 9)]
    ents = a.decode_bio_to_entities(text, labels, offsets)
    assert len(ents) == 1
    assert ents[0].text == "cancer"


def test_decode_i_label_extends_current_entity() -> None:
    """An I- label matching the open type extends the span (line 193)."""
    a = SpanAligner.without_hf()
    text = "El cancer de mama"
    labels = [_label_id("B-Premise"), _label_id("I-Premise")]
    offsets = [(3, 9), (10, 12)]
    ents = a.decode_bio_to_entities(text, labels, offsets)
    assert len(ents) == 1
    assert ents[0].start == 3
    assert ents[0].end == 12


def test_decode_o_label_flushes_open_entity() -> None:
    a = SpanAligner.without_hf()
    text = "El cancer"
    labels = [_label_id("B-Premise"), _label_id("O")]
    offsets = [(0, 2), (3, 9)]
    ents = a.decode_bio_to_entities(text, labels, offsets)
    assert len(ents) == 1
    assert ents[0].type == "Premise"


def test_validate_round_trip_raises_on_mismatch() -> None:
    from grace.io.offsets import AlignmentError
    from grace.io.schema import GraceEntity

    a = SpanAligner.without_hf()
    bad = GraceEntity(id="T1", text="zzz", start=0, end=2, type="Premise")
    with pytest.raises(AlignmentError):
        a.validate_round_trip("hello", [bad])


def test_id_to_label_and_num_labels() -> None:
    a = SpanAligner.without_hf()
    assert a.num_labels == 7
    assert a.id_to_label([0, 1]) == ["O", "B-Premise"]
