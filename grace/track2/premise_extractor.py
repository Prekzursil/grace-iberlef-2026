"""Track 2 Subtask 2: Premise span extractor with snap-to-offset.

Two-stage extraction pipeline:
1. An LLM (or encoder) proposes candidate premise phrases as verbatim
   substrings from the clinical case text
2. This module aligns each proposed phrase to exact char offsets in
   ``case.raw_text`` via substring match + fuzzy fallback, then snaps
   to the official tokenizer's boundaries

The alignment step is critical because strict-match F1 requires
character-exact offsets. An LLM that returns "fiebre de 39 grados"
when the text says "fiebre fue de 39 grados" would score 0% strict
without fuzzy recovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from grace.io.offsets import SpanAligner
from grace.io.schema import GraceCase, GraceEntity

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class PremiseExtractorConfig:
    """Configuration for the premise extractor."""

    max_fuzzy_distance: int = 2


class PremiseExtractor:
    """Aligns proposed phrase strings to exact char offsets in raw_text."""

    def __init__(self, max_fuzzy_distance: int = 2) -> None:
        self.max_fuzzy_distance = max_fuzzy_distance
        self._aligner = SpanAligner.without_hf()

    def align_proposals(
        self,
        case: GraceCase,
        sentence_idx: int,
        phrases: Sequence[str],
        entity_id_prefix: str = "P",
    ) -> tuple[GraceEntity, ...]:
        """Align a list of proposed phrase strings to char offsets.

        For each phrase:
        1. Try exact substring match within the sentence's char range
        2. If exact fails, try fuzzy match (Levenshtein distance <= max_fuzzy_distance)
        3. If both fail, drop the phrase (log a warning)
        4. Snap matched offsets to official tokenizer boundaries

        Returns a tuple of GraceEntity objects with auto-generated IDs.
        """
        sentence = case.context_sentences[sentence_idx]
        # Search within the sentence's char range in raw_text
        search_text = case.raw_text[sentence.start : sentence.end]

        out: list[GraceEntity] = []
        counter = 0

        for phrase in phrases:
            if not phrase.strip():
                continue

            # Stage 1: exact substring match
            pos = search_text.find(phrase)
            if pos >= 0:
                abs_start = sentence.start + pos
                abs_end = abs_start + len(phrase)
            else:
                # Stage 2: fuzzy match
                match = self._fuzzy_find(search_text, phrase)
                if match is not None:
                    rel_start, rel_end = match
                    abs_start = sentence.start + rel_start
                    abs_end = sentence.start + rel_end
                else:
                    _log.warning(
                        "case %s: dropping unmatched phrase %r in sentence %d",
                        case.id,
                        phrase[:50],
                        sentence_idx,
                    )
                    continue

            # Stage 3: snap to official tokenizer boundaries
            abs_start, abs_end = self._aligner.snap_to_token_boundary(
                case.raw_text, abs_start, abs_end
            )

            if abs_start >= abs_end:
                continue

            counter += 1
            out.append(
                GraceEntity(
                    id=f"{entity_id_prefix}{counter}",
                    text=case.raw_text[abs_start:abs_end],
                    start=abs_start,
                    end=abs_end,
                    type=cast(Any, "Premise"),
                )
            )

        return tuple(out)

    def _fuzzy_find(
        self,
        haystack: str,
        needle: str,
    ) -> tuple[int, int] | None:
        """Find the best fuzzy match of needle in haystack.

        Returns (start, end) char offsets within haystack, or None if
        no match within max_fuzzy_distance is found.
        """
        try:
            from Levenshtein import distance as lev_distance
        except ImportError:
            _log.warning("Levenshtein not installed; fuzzy matching disabled")
            return None

        best_dist = self.max_fuzzy_distance + 1
        best_start = 0
        best_end = 0
        needle_len = len(needle)

        # Slide a window of needle_len +/- 2 chars across haystack
        for window_delta in range(-2, 3):
            wlen = needle_len + window_delta
            if wlen <= 0 or wlen > len(haystack):
                continue
            for start in range(len(haystack) - wlen + 1):
                candidate = haystack[start : start + wlen]
                d = lev_distance(needle, candidate)
                if d < best_dist:
                    best_dist = d
                    best_start = start
                    best_end = start + wlen

        if best_dist <= self.max_fuzzy_distance:
            return (best_start, best_end)
        return None
