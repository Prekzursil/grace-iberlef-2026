"""Tests for grace.io.tokenizer.OfficialTokenizer."""

from grace.io.tokenizer import OfficialTokenizer


def test_tokenize_splits_words_and_punctuation() -> None:
    tok = OfficialTokenizer()
    tokens = tok.tokenize("Hola, mundo!")
    assert [t.text for t in tokens] == ["Hola", ",", "mundo", "!"]
    assert tokens[0].start == 0 and tokens[0].end == 4
    assert tokens[1].start == 4 and tokens[1].end == 5
    assert tokens[2].start == 6 and tokens[2].end == 11
    assert tokens[3].start == 11 and tokens[3].end == 12


def test_tokenize_handles_spanish_accents() -> None:
    tok = OfficialTokenizer()
    tokens = tok.tokenize("niño cáncer")
    assert [t.text for t in tokens] == ["niño", "cáncer"]


def test_tokenize_handles_empty_string() -> None:
    tok = OfficialTokenizer()
    assert tok.tokenize("") == ()


def test_tokenize_handles_numbers_and_units() -> None:
    tok = OfficialTokenizer()
    tokens = tok.tokenize("Hb 11.5 g/dl")
    # Digits join with the preceding word char via \w+, but the period
    # is non-word and splits. So "11" ".", "5" are three tokens.
    texts = [t.text for t in tokens]
    assert "Hb" in texts
    assert "g" in texts
    assert "dl" in texts
    # Verify all offsets are within bounds
    for t in tokens:
        assert 0 <= t.start < t.end <= len("Hb 11.5 g/dl")


def test_token_iou_empty_sets_returns_one() -> None:
    tok = OfficialTokenizer()
    assert tok.token_iou(frozenset(), frozenset()) == 1.0


def test_token_iou_partial_overlap() -> None:
    tok = OfficialTokenizer()
    a = frozenset({1, 2, 3})
    b = frozenset({2, 3, 4})
    assert tok.token_iou(a, b) == 2 / 4  # 0.5


def test_token_iou_identical_sets() -> None:
    tok = OfficialTokenizer()
    a = frozenset({1, 2, 3})
    assert tok.token_iou(a, a) == 1.0


def test_token_iou_disjoint_sets() -> None:
    tok = OfficialTokenizer()
    a = frozenset({1, 2})
    b = frozenset({3, 4})
    assert tok.token_iou(a, b) == 0.0


def test_token_indices_in_span_exclusive_end() -> None:
    tok = OfficialTokenizer()
    tokens = tok.tokenize("abc def")
    # span covering "abc" only
    idx = tok.token_indices_in_span(tokens, 0, 3)
    assert idx == frozenset({0})
    # span covering both
    idx = tok.token_indices_in_span(tokens, 0, 7)
    assert idx == frozenset({0, 1})
    # span covering nothing
    idx = tok.token_indices_in_span(tokens, 3, 4)  # the space
    assert idx == frozenset()
