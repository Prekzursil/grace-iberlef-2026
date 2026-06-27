"""A faithful stand-in for the organizer-provided scoring programs.

The real ``track{1,2}_scoring_program.py`` files are NON-REDISTRIBUTABLE
(organizer-provided, gitignored under ``downloaded_data/``) so they cannot
live in CI. This fake reproduces the public interface that
``grace.eval.scorer`` and ``grace.submit.validator`` depend on:

* ``evaluate(predictions_path, gold_path=None) -> dict`` returning per-subtask
  ``{"official_score": float}`` dicts (plus a couple of non-conforming values
  to exercise the validator's result-filtering branches), and
* ``_tokenize(text) -> list[tuple[int, int]]`` (whitespace offsets).

Tests monkeypatch ``grace.eval.scorer._SCORER_PATHS`` to point here, which
exercises the loader/fingerprint/score-from-file wrapper end to end without
redistributing organizer code.
"""

from __future__ import annotations

import json
from pathlib import Path


def _tokenize(text: str) -> list[tuple[int, int]]:
    tokens: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append((i, j))
        i = j
    return tokens


def evaluate(predictions_path: str, gold_path: str | None = None) -> dict:
    data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    score = 1.0 if data else 0.0
    return {
        "subtask1": {"official_score": score},
        "subtask2": {"official_score": score},
        "subtask3": {"official_score": score},
        # Non-conforming entries: exercise validator result filtering.
        "scorer_version": "fake-1.0",  # not a dict -> filtered out
        "details": {"notes": "no official_score key"},  # dict without the key
    }
