"""Convert GraceCase tuples into Codabench-ready JSON submissions.

This is a thin wrapper around :func:`grace.io.loaders.save_predictions` —
it lives in the submit package so that submission packaging can be
extended later with metadata, zip creation, etc., without touching the
io layer.
"""

from __future__ import annotations

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
    """Write a submission file in the exact format the scorer expects."""
    save_predictions(tuple(cases), output_path, track=track)
