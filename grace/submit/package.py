"""Zip a prediction file into a Codabench-ready submission bundle.

Codabench expects a zip containing the prediction JSON. Naming convention:
``grace-2026-track{N}-<timestamp>.zip``. Called by ``scripts/submit.py``.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def package_submission(
    prediction_path: Path,
    track: int,
    output_dir: Path,
) -> Path:
    """Zip ``prediction_path`` into ``output_dir`` with a timestamped name.

    Returns the path to the created zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    zip_path = output_dir / f"grace-2026-track{track}-{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname=prediction_path.name)
    return zip_path
