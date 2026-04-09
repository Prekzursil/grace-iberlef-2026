"""Loaders for GRACE 2026 train/dev/test JSON files.

Targets the actual scoring-program format where ``annotations`` is a dict
with ``entities``/``relations`` keys (Track 1) or
``sentence_relevancy``/``entities``/``relations`` (Track 2). The
``instance_examples/`` directory uses a flat-list format that is OUT OF DATE
and must not be targeted — see the design doc Appendix B for the discrepancy
report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from grace.io.schema import (
    GraceCase,
    GraceChoice,
    GraceEntity,
    GraceRelation,
    GraceSentence,
)

_log = logging.getLogger(__name__)


class GraceLoadError(ValueError):
    """Raised when a case fails schema validation at load time."""


def _validate_entity_substring(case_id: str, raw_text: str, e: GraceEntity) -> None:
    if not (0 <= e.start < e.end <= len(raw_text)):
        raise GraceLoadError(
            f"entity {e.id} in case {case_id}: offsets out of range "
            f"[{e.start}, {e.end}) vs text length {len(raw_text)}"
        )
    if raw_text[e.start : e.end] != e.text:
        raise GraceLoadError(
            f"entity {e.id} in case {case_id}: text does not match raw_text slice; "
            f"gold={e.text!r}, slice={raw_text[e.start : e.end]!r}"
        )


def _keep_valid_relations(
    case_id: str,
    ent_ids: set[str],
    rels: tuple[GraceRelation, ...],
) -> tuple[GraceRelation, ...]:
    """Drop relations whose argument IDs are missing from entities.

    Matches the official scoring programs' behavior — both track1 and track2
    scorers silently skip dangling relations in ``_to_rel_tuples``. The GRACE
    2026 training data contains at least one case (10561201) with a dangling
    relation, so the loader MUST tolerate this rather than fail.
    """
    kept: list[GraceRelation] = []
    for r in rels:
        if r.arg1_id not in ent_ids or r.arg2_id not in ent_ids:
            _log.warning(
                "case %s: dropping dangling relation %s (arg1=%s, arg2=%s)",
                case_id,
                r.id,
                r.arg1_id,
                r.arg2_id,
            )
            continue
        kept.append(r)
    return tuple(kept)


def load_track1(path: Path) -> tuple[GraceCase, ...]:
    """Load a Track 1 JSON file (350 train / 50 dev).

    Expects ``annotations`` to be a dict with ``entities`` and ``relations``
    keys. Raises :class:`GraceLoadError` on any schema violation.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise GraceLoadError(f"{path}: top-level must be a JSON array")

    out: list[GraceCase] = []
    for raw in data:
        case_id = raw["id"]
        raw_text = raw["raw_text"]
        ann = raw.get("annotations") or {}
        if not isinstance(ann, dict):
            raise GraceLoadError(
                f"case {case_id}: annotations must be a dict (scoring format), got "
                f"{type(ann).__name__}"
            )
        ents_raw: list[dict[str, Any]] = ann.get("entities") or []
        rels_raw: list[dict[str, Any]] = ann.get("relations") or []

        entities = tuple(
            GraceEntity(
                id=str(e["id"]),
                text=e["text"],
                start=int(e["start"]),
                end=int(e["end"]),
                type=cast(Any, e["type"]),
            )
            for e in ents_raw
        )
        for e in entities:
            _validate_entity_substring(case_id, raw_text, e)

        relations = tuple(
            GraceRelation(
                id=str(r["id"]),
                arg1_id=str(r["arg1_id"]),
                arg2_id=str(r["arg2_id"]),
                relation_type=cast(Any, r["relation_type"]),
            )
            for r in rels_raw
        )
        ent_ids = {e.id for e in entities}
        relations = _keep_valid_relations(case_id, ent_ids, relations)

        out.append(
            GraceCase(
                id=case_id,
                raw_text=raw_text,
                track=1,
                entities=entities,
                relations=relations,
            )
        )
    return tuple(out)


def load_track2(path: Path) -> tuple[GraceCase, ...]:
    """Load a Track 2 JSON file (128 train / 24 dev — MIR exam cases).

    Expects ``annotations`` to be a dict with ``sentence_relevancy``, ``entities``,
    and ``relations`` keys. ``metadata`` contains ``context``, ``context_sentences``,
    ``choices``, and ``correct_choice_id``. Raises :class:`GraceLoadError` on any
    schema violation.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise GraceLoadError(f"{path}: top-level must be a JSON array")

    out: list[GraceCase] = []
    for raw in data:
        case_id = raw["id"]
        raw_text = raw["raw_text"]
        md = raw.get("metadata") or {}
        sentences = tuple(
            GraceSentence(
                sentence=s["sentence"],
                start=int(s["start"]),
                end=int(s["end"]),
            )
            for s in md.get("context_sentences", [])
        )
        choices = tuple(
            GraceChoice(
                id=str(c["id"]),
                text=c["text"],
                start=int(c["start"]),
                end=int(c["end"]),
            )
            for c in md.get("choices", [])
        )
        correct = md.get("correct_choice_id")
        correct_id = str(correct) if correct is not None else None
        metadata_context = md.get("context")

        ann = raw.get("annotations") or {}
        if not isinstance(ann, dict):
            raise GraceLoadError(
                f"case {case_id}: annotations must be a dict, got {type(ann).__name__}"
            )

        relevancy = tuple(cast(Any, lbl) for lbl in ann.get("sentence_relevancy", []))
        if len(relevancy) != len(sentences):
            raise GraceLoadError(
                f"case {case_id}: sentence_relevancy length {len(relevancy)} != "
                f"context_sentences length {len(sentences)}"
            )

        entities = tuple(
            GraceEntity(
                id=str(e["id"]),
                text=e["text"],
                start=int(e["start"]),
                end=int(e["end"]),
                type=cast(Any, e["type"]),
            )
            for e in ann.get("entities", [])
        )
        for e in entities:
            _validate_entity_substring(case_id, raw_text, e)

        relations = tuple(
            GraceRelation(
                id=str(r["id"]),
                arg1_id=str(r["arg1_id"]),
                arg2_id=str(r["arg2_id"]),
                relation_type=cast(Any, r["relation_type"]),
            )
            for r in ann.get("relations", [])
        )
        ent_ids = {e.id for e in entities}
        relations = _keep_valid_relations(case_id, ent_ids, relations)

        if correct_id is not None and correct_id not in {c.id for c in choices}:
            raise GraceLoadError(f"case {case_id}: correct_choice_id {correct_id!r} not in choices")

        out.append(
            GraceCase(
                id=case_id,
                raw_text=raw_text,
                track=2,
                metadata_context=metadata_context,
                context_sentences=sentences,
                choices=choices,
                correct_choice_id=correct_id,
                sentence_relevancy=relevancy,
                entities=entities,
                relations=relations,
            )
        )
    return tuple(out)


def _case_to_track1_dict(case: GraceCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "raw_text": case.raw_text,
        "metadata": {},
        "annotations": {
            "entities": [
                {
                    "id": e.id,
                    "text": e.text,
                    "start": e.start,
                    "end": e.end,
                    "type": e.type,
                }
                for e in case.entities
            ],
            "relations": [
                {
                    "id": r.id,
                    "arg1_id": r.arg1_id,
                    "arg2_id": r.arg2_id,
                    "relation_type": r.relation_type,
                }
                for r in case.relations
            ],
        },
    }


def _case_to_track2_dict(case: GraceCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "raw_text": case.raw_text,
        "metadata": {
            "context": case.metadata_context,
            "context_sentences": [
                {"sentence": s.sentence, "start": s.start, "end": s.end}
                for s in case.context_sentences
            ],
            "choices": [
                {"id": c.id, "text": c.text, "start": c.start, "end": c.end} for c in case.choices
            ],
            "correct_choice_id": case.correct_choice_id,
        },
        "annotations": {
            "sentence_relevancy": list(case.sentence_relevancy),
            "entities": [
                {
                    "id": e.id,
                    "text": e.text,
                    "start": e.start,
                    "end": e.end,
                    "type": e.type,
                }
                for e in case.entities
            ],
            "relations": [
                {
                    "id": r.id,
                    "arg1_id": r.arg1_id,
                    "arg2_id": r.arg2_id,
                    "relation_type": r.relation_type,
                }
                for r in case.relations
            ],
        },
    }


def save_predictions(
    cases: tuple[GraceCase, ...] | list[GraceCase],
    path: Path,
    track: int,
) -> None:
    """Serialize cases to the exact JSON format the official scorer expects.

    Track 1: flat ``{id, raw_text, metadata, annotations{entities, relations}}``.
    Track 2: ``annotations`` additionally includes ``sentence_relevancy``, and
    ``metadata`` carries ``context``, ``context_sentences``, ``choices``, and
    ``correct_choice_id``.
    """
    if track == 1:
        payload = [_case_to_track1_dict(c) for c in cases]
    elif track == 2:
        payload = [_case_to_track2_dict(c) for c in cases]
    else:
        raise ValueError(f"unknown track: {track}")
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
