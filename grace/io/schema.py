"""Immutable dataclasses representing GRACE 2026 cases and annotations.

Every dataclass here is `frozen=True, slots=True` — the invariants we enforce in
`grace.io.loaders` (entity.text == raw_text[start:end], etc.) would be meaningless
if a downstream caller could mutate a case in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EntityType = Literal["Premise", "Claim", "MajorClaim"]
RelationType = Literal["Support", "Attack", "Partial-Attack"]
SentRelevancy = Literal["relevant", "not-relevant"]


@dataclass(frozen=True, slots=True)
class GraceEntity:
    """A labeled span inside a case's raw_text.

    start is inclusive, end is exclusive. raw_text[start:end] MUST equal text.
    """

    id: str
    text: str
    start: int
    end: int
    type: EntityType


@dataclass(frozen=True, slots=True)
class GraceRelation:
    """A directed relation between two entities inside the same case."""

    id: str
    arg1_id: str
    arg2_id: str
    relation_type: RelationType


@dataclass(frozen=True, slots=True)
class GraceSentence:
    """One sentence from a Track 2 case's `metadata.context_sentences`."""

    sentence: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class GraceChoice:
    """One multiple-choice option from a Track 2 case's `metadata.choices`."""

    id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class GraceCase:
    """One case from either Track 1 (RCT abstract) or Track 2 (MIR exam case).

    Track 1 uses only: id, raw_text, entities, relations.
    Track 2 additionally uses: metadata_context, context_sentences, choices,
        correct_choice_id, sentence_relevancy.
    """

    id: str
    raw_text: str
    track: Literal[1, 2]
    metadata_context: str | None = None
    context_sentences: tuple[GraceSentence, ...] = ()
    choices: tuple[GraceChoice, ...] = ()
    correct_choice_id: str | None = None
    sentence_relevancy: tuple[SentRelevancy, ...] = ()
    entities: tuple[GraceEntity, ...] = ()
    relations: tuple[GraceRelation, ...] = ()
