"""Pre-upload validation: run the official scorer on a submission.

Catches format drift before Codabench upload. Every submission MUST pass
this validator before any Codabench upload per the design doc §5.5.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from grace.eval.scorer import score_track1_from_file, score_track2_from_file

if TYPE_CHECKING:
    from pathlib import Path


class SubmissionValidationError(Exception):
    """Raised when a submission file fails format or scorer validation."""


def validate_submission(
    submission_path: Path,
    track: int,
    gold_path: Path | None = None,
) -> dict[str, float]:
    """Run the official scorer on the submission file.

    Returns a dict of ``{subtask_name: official_score}`` on success, or
    raises :class:`SubmissionValidationError` on any failure.
    """
    data = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SubmissionValidationError("submission must be a JSON array")
    if not data:
        raise SubmissionValidationError("submission is empty")

    try:
        if track == 1:
            results = score_track1_from_file(submission_path, gold_path)
        elif track == 2:
            results = score_track2_from_file(submission_path, gold_path)
        else:
            raise SubmissionValidationError(f"unknown track: {track}")
    except SubmissionValidationError:
        raise
    except Exception as e:
        raise SubmissionValidationError(f"scorer crashed: {e}") from e

    return {
        k: v["official_score"]
        for k, v in results.items()
        if isinstance(v, dict) and "official_score" in v
    }
