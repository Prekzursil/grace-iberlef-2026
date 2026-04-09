"""Verbatim port of the GRACE scoring programs' tokenizer regex.

Both track{1,2}_scoring_program.py use ``re.compile(r"\\w+|[^\\w\\s]", re.UNICODE)``.
We reproduce that exact behaviour so our IoU computations match the scorer
byte-for-byte. Any drift here silently breaks relaxed-match evaluation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

# Identical to the pattern used in track{1,2}_scoring_program.py
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Token(NamedTuple):
    """A single token produced by OfficialTokenizer.

    start is inclusive, end is exclusive, consistent with GraceEntity offsets.
    """

    text: str
    start: int
    end: int


class OfficialTokenizer:
    """Matches track{1,2}_scoring_program.py tokenizer byte-for-byte."""

    def tokenize(self, text: str) -> tuple[Token, ...]:
        """Return the tuple of tokens for ``text``."""
        return tuple(Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text))

    def token_indices_in_span(
        self,
        tokens: Sequence[Token],
        start: int,
        end: int,
    ) -> frozenset[int]:
        """Indices of tokens whose character range is fully within ``[start, end)``."""
        return frozenset(i for i, t in enumerate(tokens) if t.start >= start and t.end <= end)

    def token_iou(self, a: frozenset[int], b: frozenset[int]) -> float:
        """Token-level Jaccard index.

        Returns 1.0 when both sets are empty (matching the scorer convention).
        """
        if not a and not b:
            return 1.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0
