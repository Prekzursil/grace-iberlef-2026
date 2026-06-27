"""Coverage for grace.submit.package.package_submission."""

import zipfile
from pathlib import Path

from grace.submit.package import package_submission


def test_package_submission_creates_named_zip(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.json"
    pred.write_text("[]", encoding="utf-8")
    out_dir = tmp_path / "bundles"
    zip_path = package_submission(pred, track=1, output_dir=out_dir)

    assert zip_path.exists()
    assert zip_path.parent == out_dir
    assert zip_path.name.startswith("grace-2026-track1-")
    assert zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["predictions.json"]


def test_package_submission_creates_output_dir(tmp_path: Path) -> None:
    pred = tmp_path / "p.json"
    pred.write_text("[]", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    zip_path = package_submission(pred, track=2, output_dir=nested)
    assert zip_path.exists()
    assert "track2" in zip_path.name
