"""Error analysis utilities for GRACE predictions.

Consumes gold + predicted ``GraceCase`` tuples and produces a diagnostics
dict suitable for writing to ``diagnostics.json`` alongside ``metrics.json``
in every run directory. Used by Track 1 + Track 2 training scripts and by
the paper's error analysis section.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grace.io.schema import GraceCase, GraceEntity


def _entity_key(e: GraceEntity) -> tuple[int, int, str]:
    return (e.start, e.end, e.type)


def _match_entities(
    gold: Sequence[GraceEntity],
    pred: Sequence[GraceEntity],
) -> tuple[list[tuple[GraceEntity, GraceEntity]], list[GraceEntity], list[GraceEntity]]:
    """Greedy 1-to-1 strict match (char-exact + same type).

    Returns (matched_pairs, unmatched_gold, unmatched_pred).
    """
    used: set[int] = set()
    matched: list[tuple[GraceEntity, GraceEntity]] = []
    unmatched_gold: list[GraceEntity] = []
    for g in gold:
        match_idx: int | None = None
        for j, p in enumerate(pred):
            if j in used:
                continue
            if _entity_key(g) == _entity_key(p):
                match_idx = j
                break
        if match_idx is not None:
            matched.append((g, pred[match_idx]))
            used.add(match_idx)
        else:
            unmatched_gold.append(g)
    unmatched_pred = [pred[j] for j in range(len(pred)) if j not in used]
    return matched, unmatched_gold, unmatched_pred


def build_diagnostics(
    gold_cases: Sequence[GraceCase],
    pred_cases: Sequence[GraceCase],
    track: int,
    worst_n: int = 10,
) -> dict[str, Any]:
    """Compute per-class confusion + offset error histogram + worst-N cases.

    Returns a JSON-serializable dict.
    """
    gold_by_id = {c.id: c for c in gold_cases}
    pred_by_id = {c.id: c for c in pred_cases}
    ids = sorted(set(gold_by_id.keys()) & set(pred_by_id.keys()))

    per_case_f1: list[tuple[str, float]] = []
    type_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    offset_errors: list[int] = []

    for case_id in ids:
        g = gold_by_id[case_id]
        p = pred_by_id[case_id]
        matched, unmatched_g, unmatched_p = _match_entities(g.entities, p.entities)
        tp = len(matched)
        fp = len(unmatched_p)
        fn = len(unmatched_g)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_case_f1.append((case_id, f1))
        # Confusion: gold type -> predicted type (strict matches only reach here)
        for gold_e, _pred_e in matched:
            type_confusion[gold_e.type][gold_e.type] += 1
        # Any unmatched gold counts as gold_type -> "MISS"
        for e in unmatched_g:
            type_confusion[e.type]["MISS"] += 1

    # Worst-N cases (lowest F1 first)
    per_case_f1.sort(key=lambda kv: kv[1])
    worst_cases = [{"case_id": cid, "f1": f1} for cid, f1 in per_case_f1[:worst_n]]

    corpus_f1 = statistics.mean(f for _, f in per_case_f1) if per_case_f1 else 0.0
    length_vs_score = [
        {"case_id": cid, "f1": f1, "text_len": len(gold_by_id[cid].raw_text)}
        for cid, f1 in per_case_f1
    ]

    return {
        "track": track,
        "num_cases": len(ids),
        "corpus_f1_mean": round(corpus_f1, 4),
        "per_type_confusion": {k: dict(v) for k, v in type_confusion.items()},
        "offset_error_histogram": {
            "count": len(offset_errors),
            "max": max(offset_errors) if offset_errors else 0,
        },
        "worst_cases": worst_cases,
        "length_vs_score": length_vs_score,
    }
