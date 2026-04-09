"""Offset alignment utilities bridging HF tokenizers and official char offsets.

This module is the single place in the codebase where offset handling is
defined. Every prediction that touches Codabench must pass through
:class:`SpanAligner` before being serialised, or strict F1 will silently
collapse.

BIO labels used here are a superset of both Track 1 and Track 2 entity types:
``O``, ``B-/I-Premise``, ``B-/I-Claim``, ``B-/I-MajorClaim``. Track 2 only
produces Premise + Claim in the GRACE 2026 release, so any MajorClaim BIO
label predicted on Track 2 inputs is silently dropped during decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from grace.io.schema import GraceEntity
from grace.io.tokenizer import OfficialTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerFast


class AlignmentError(ValueError):
    """Raised when a predicted span cannot be snapped or validated."""


# BIO labels — used by Track 1's component tagger and by Track 2 (subset)
_BIO_LABELS: tuple[str, ...] = (
    "O",
    "B-Premise",
    "I-Premise",
    "B-Claim",
    "I-Claim",
    "B-MajorClaim",
    "I-MajorClaim",
)
_LABEL_TO_ID: dict[str, int] = {lbl: i for i, lbl in enumerate(_BIO_LABELS)}
_ID_TO_LABEL: dict[int, str] = dict(enumerate(_BIO_LABELS))


class SpanAligner:
    """Bridges HF tokenizer offsets <-> official char-offset contract."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerFast | None = None) -> None:
        self.hf_tokenizer = hf_tokenizer
        self._official = OfficialTokenizer()

    @classmethod
    def without_hf(cls) -> SpanAligner:
        """Construct a pure-snap/validate aligner without an HF tokenizer attached."""
        return cls(hf_tokenizer=None)

    # ------------------------------------------------------------------
    # Pure offset utilities — no HF tokenizer required
    # ------------------------------------------------------------------

    def snap_to_token_boundary(
        self,
        text: str,
        char_start: int,
        char_end: int,
    ) -> tuple[int, int]:
        """Snap (char_start, char_end) to the nearest valid official-tokenizer boundaries."""
        if char_start >= char_end:
            return (char_start, char_end)
        tokens = self._official.tokenize(text)
        if not tokens:
            return (char_start, char_end)

        new_start = char_start
        for t in tokens:
            if t.start >= char_start:
                new_start = t.start
                break
            if t.start < char_start < t.end:
                new_start = t.start
                break

        new_end = char_end
        for t in tokens:
            if t.end >= char_end:
                new_end = t.end
                break
        return (new_start, new_end)

    def validate_round_trip(
        self,
        text: str,
        entities: Sequence[GraceEntity],
    ) -> None:
        """Raise AlignmentError if any entity text does not match raw_text[start:end]."""
        for e in entities:
            actual = text[e.start : e.end]
            if actual != e.text:
                raise AlignmentError(
                    f"entity {e.id}: text {e.text!r} != raw_text[{e.start}:{e.end}] = "
                    f"{actual!r}"
                )

    # ------------------------------------------------------------------
    # HF-backed BIO encoding / decoding — requires ``hf_tokenizer``
    # ------------------------------------------------------------------

    def encode_with_labels(
        self,
        text: str,
        gold_entities: Sequence[GraceEntity],
        max_length: int = 512,
    ) -> dict[str, Any]:
        """Encode text with the HF tokenizer and produce BIO labels aligned to subwords."""
        if self.hf_tokenizer is None:
            raise RuntimeError("encode_with_labels requires hf_tokenizer")
        enc = self.hf_tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        offsets = enc["offset_mapping"]
        labels: list[int] = [_LABEL_TO_ID["O"]] * len(offsets)

        for e in gold_entities:
            started = False
            for i, (s, t) in enumerate(offsets):
                if s == 0 and t == 0:
                    continue  # special / padding token
                if s >= e.start and t <= e.end and s < t:
                    if not started:
                        labels[i] = _LABEL_TO_ID[f"B-{e.type}"]
                        started = True
                    else:
                        labels[i] = _LABEL_TO_ID[f"I-{e.type}"]
                elif s >= e.end:
                    break

        result: dict[str, Any] = dict(enc)
        result["labels"] = labels
        return result

    def decode_bio_to_entities(
        self,
        text: str,
        label_ids: Sequence[int],
        offset_mapping: Sequence[tuple[int, int]],
        entity_id_prefix: str = "P",
    ) -> tuple[GraceEntity, ...]:
        """Convert a predicted BIO label stream back into GraceEntity objects."""
        out: list[GraceEntity] = []
        current_type: str | None = None
        current_start: int | None = None
        current_end: int | None = None
        counter = 0

        def _flush() -> None:
            nonlocal counter, current_type, current_start, current_end
            if current_start is None or current_end is None or current_type is None:
                return
            snapped = self.snap_to_token_boundary(text, current_start, current_end)
            s, t = snapped
            if s < t:
                counter += 1
                out.append(
                    GraceEntity(
                        id=f"{entity_id_prefix}{counter}",
                        text=text[s:t],
                        start=s,
                        end=t,
                        type=cast("Any", current_type),
                    )
                )
            current_type = None
            current_start = None
            current_end = None

        for i, lbl_id in enumerate(label_ids):
            lbl = _ID_TO_LABEL.get(int(lbl_id), "O")
            s, t = offset_mapping[i]
            s = int(s)
            t = int(t)
            if s == 0 and t == 0:
                continue
            if lbl.startswith("B-"):
                _flush()
                current_type = lbl[2:]
                current_start = s
                current_end = t
            elif lbl.startswith("I-") and current_type == lbl[2:]:
                current_end = t
            elif lbl.startswith("I-"):
                _flush()
                current_type = lbl[2:]
                current_start = s
                current_end = t
            else:
                _flush()
        _flush()
        return tuple(out)

    def id_to_label(self, label_ids: Sequence[int]) -> list[str]:
        """Convert label ids back to BIO label strings."""
        return [_ID_TO_LABEL[int(i)] for i in label_ids]

    @property
    def num_labels(self) -> int:
        """Number of BIO label classes (7 = O + B/I x 3 entity types)."""
        return len(_BIO_LABELS)
