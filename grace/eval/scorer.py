"""Wrapper around the organizer-provided scoring programs.

We import the organizer scripts verbatim via ``importlib`` rather than
reimplementing the metrics. This guarantees every dev-set number we report
matches what Codabench will report on the leaderboard byte-for-byte.

Each scorer's SHA256 is logged with every experiment run so that organizer
updates to the scoring program are detected immediately.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

_SCORER_ROOT = Path("downloaded_data")

_SCORER_PATHS: dict[int, Path] = {
    1: (
        _SCORER_ROOT
        / "track1_scoring_program"
        / "track1_scoring_program"
        / "track1_scoring_program.py"
    ),
    2: (
        _SCORER_ROOT
        / "track2_scoring_program"
        / "track2_scoring_program"
        / "track2_scoring_program.py"
    ),
}


def _load(track: int) -> Any:
    path = _SCORER_PATHS[track]
    spec = importlib.util.spec_from_file_location(f"grace_scorer_t{track}", path)
    assert spec and spec.loader, f"cannot load scorer for track {track}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scorer_fingerprint(track: int) -> str:
    """SHA256 of the scorer script.

    Logged alongside every run's metrics. If the organizers push an update
    to the scoring program mid-competition, this hash will change and we
    can re-evaluate prior runs on the new version.
    """
    return hashlib.sha256(_SCORER_PATHS[track].read_bytes()).hexdigest()


def score_track1_from_file(
    predictions_path: Path,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    """Run the Track 1 scorer on a predictions file.

    When ``gold_path`` is None, the scorer uses the ``annotations`` field of
    the predictions file as gold (self-consistency mode). Pass a gold path
    for two-file scoring.
    """
    return _load(1).evaluate(predictions_path, gold_path)  # type: ignore[no-any-return]


def score_track2_from_file(
    predictions_path: Path,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    """Run the Track 2 scorer on a predictions file."""
    return _load(2).evaluate(predictions_path, gold_path)  # type: ignore[no-any-return]
