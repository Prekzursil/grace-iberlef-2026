"""Convert GraceCase tuples into Codabench-ready JSON submissions.

Two functions:
- ``format_submission``: writes gold annotations (for round-trip tests)
- ``format_predictions``: writes model output into the ``predictions``
  block for proper two-file scoring against a gold file
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from grace.io.loaders import save_predictions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from grace.io.schema import GraceCase


def format_submission(
    cases: Sequence[GraceCase],
    output_path: Path,
    track: int,
) -> None:
    """Write cases with gold annotations. Used for round-trip self-consistency tests."""
    save_predictions(tuple(cases), output_path, track=track)


def format_predictions(
    pred_cases: Sequence[GraceCase],
    output_path: Path,
    track: int,
) -> None:
    """Write model predictions into the ``predictions`` block.

    The scorer's two-file mode reads ``predictions`` from this file and
    ``annotations`` from the gold file (passed as ``gold_path``). This
    is the function to use for real evaluation, NOT ``format_submission``.
    """
    payload: list[dict] = []
    for case in pred_cases:
        entry: dict = {
            "id": case.id,
            "raw_text": case.raw_text,
        }
        preds_block: dict = {
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
        }
        if track == 2:
            preds_block["sentence_relevancy"] = list(case.sentence_relevancy)
        entry["predictions"] = preds_block
        # Also need annotations block for two-file scoring to work
        # (scorer merges from gold file, but some fields like metadata
        # come from the predictions file)
        if track == 2:
            entry["metadata"] = {
                "context": case.metadata_context,
                "context_sentences": [
                    {"sentence": s.sentence, "start": s.start, "end": s.end}
                    for s in case.context_sentences
                ],
                "choices": [
                    {"id": c.id, "text": c.text, "start": c.start, "end": c.end}
                    for c in case.choices
                ],
                "correct_choice_id": case.correct_choice_id,
            }
        payload.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
