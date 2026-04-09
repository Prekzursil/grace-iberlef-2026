"""Cross-check OfficialTokenizer against the real scoring program's tokenizer.

This is the single most important correctness test at the tokenizer layer.
If our ``OfficialTokenizer`` drifts from the scorer's ``_tokenize``, relaxed-match
F1 numbers will silently disagree with Codabench.
"""

import importlib.util
from pathlib import Path

from grace.io.tokenizer import OfficialTokenizer

_SCORER_PATH = (
    Path("downloaded_data")
    / "track2_scoring_program"
    / "track2_scoring_program"
    / "track2_scoring_program.py"
)


def _load_scorer_module():
    spec = importlib.util.spec_from_file_location("grace_scorer_parity", _SCORER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tokenizer_matches_official_scorer_byte_for_byte() -> None:
    mod = _load_scorer_module()
    ours = OfficialTokenizer()
    samples = [
        "El papel de la quimioterapia (HDCT) en el cáncer de mama.",
        "Hb 11.5 g/dl, Hto. 35%, Fe 38 ug/dl.",
        "¿Cuál es la opción más apropiada?",
        "niño con dolor en el oído derecho",
        "   leading and trailing spaces   ",
        "",
        "Single",
        "a.b.c",
        "12345",
        "Ac antitransglutaminasa IgH 177 U/ml",
    ]
    for sample in samples:
        our_tokens = ours.tokenize(sample)
        their_tokens = mod._tokenize(sample)  # returns list[tuple[int, int]]
        assert len(our_tokens) == len(their_tokens), f"length mismatch on: {sample!r}"
        for our_t, their_t in zip(our_tokens, their_tokens, strict=False):
            assert (
                our_t.start,
                our_t.end,
            ) == their_t, f"offset drift on {sample!r}: ours={our_t}, theirs={their_t}"
