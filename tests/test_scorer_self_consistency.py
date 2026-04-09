"""Prove that round-tripped gold predictions score 1.0 via the official scorers.

This is the single most important correctness test in the whole project. If it
passes, our saved files are wire-compatible with Codabench and any "improvement"
we see on our dev scorer translates 1:1 to the leaderboard. If it fails, fix
``grace/io/loaders.py::save_predictions`` before doing anything else.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

from grace.io.loaders import load_track1, load_track2, save_predictions

_SCORER_T1 = (
    Path("downloaded_data")
    / "track1_scoring_program"
    / "track1_scoring_program"
    / "track1_scoring_program.py"
)
_SCORER_T2 = (
    Path("downloaded_data")
    / "track2_scoring_program"
    / "track2_scoring_program"
    / "track2_scoring_program.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_track1_annotations_to_predictions(src: Path, dst: Path) -> None:
    """The Track 1 scorer expects a 'predictions' block. Copy annotations into it."""
    data = json.loads(src.read_text(encoding="utf-8"))
    for case in data:
        case["predictions"] = {
            "entities": list(case["annotations"]["entities"]),
            "relations": list(case["annotations"]["relations"]),
        }
    dst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _copy_track2_annotations_to_predictions(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    for case in data:
        case["predictions"] = {
            "sentence_relevancy": list(case["annotations"]["sentence_relevancy"]),
            "entities": list(case["annotations"]["entities"]),
            "relations": list(case["annotations"]["relations"]),
        }
    dst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_track1_identity_predictions_score_perfect() -> None:
    mod = _load_module(_SCORER_T1, "grace_scorer_t1_self")
    dev = Path("downloaded_data/public_data/public_data/track_1_dev.json")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t1_self.json"
        _copy_track1_annotations_to_predictions(dev, out)
        results = mod.evaluate(out)
    assert results["subtask1"]["official_score"] == 1.0, results["subtask1"]
    assert results["subtask2"]["official_score"] == 1.0, results["subtask2"]


def test_track2_identity_predictions_score_perfect() -> None:
    mod = _load_module(_SCORER_T2, "grace_scorer_t2_self")
    dev = Path("downloaded_data/public_data/public_data/track_2_dev.json")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t2_self.json"
        _copy_track2_annotations_to_predictions(dev, out)
        results = mod.evaluate(out)
    assert results["subtask1"]["official_score"] == 1.0, results["subtask1"]
    assert results["subtask2"]["official_score"] == 1.0, results["subtask2"]
    assert results["subtask3"]["official_score"] == 1.0, results["subtask3"]


def test_track1_saved_by_our_loader_passes_scorer() -> None:
    """End-to-end: our loader → save_predictions → scorer must score 1.0."""
    mod = _load_module(_SCORER_T1, "grace_scorer_t1_save")
    dev_path = Path("downloaded_data/public_data/public_data/track_1_dev.json")
    cases = load_track1(dev_path)
    with tempfile.TemporaryDirectory() as d:
        saved = Path(d) / "saved.json"
        save_predictions(cases, saved, track=1)
        # Copy annotations → predictions so the scorer has a prediction block
        with_preds = Path(d) / "with_preds.json"
        _copy_track1_annotations_to_predictions(saved, with_preds)
        results = mod.evaluate(with_preds)
    assert results["subtask1"]["official_score"] == 1.0, results["subtask1"]
    assert results["subtask2"]["official_score"] == 1.0, results["subtask2"]


def test_track2_saved_by_our_loader_passes_scorer() -> None:
    mod = _load_module(_SCORER_T2, "grace_scorer_t2_save")
    dev_path = Path("downloaded_data/public_data/public_data/track_2_dev.json")
    cases = load_track2(dev_path)
    with tempfile.TemporaryDirectory() as d:
        saved = Path(d) / "saved.json"
        save_predictions(cases, saved, track=2)
        with_preds = Path(d) / "with_preds.json"
        _copy_track2_annotations_to_predictions(saved, with_preds)
        results = mod.evaluate(with_preds)
    assert results["subtask1"]["official_score"] == 1.0, results["subtask1"]
    assert results["subtask2"]["official_score"] == 1.0, results["subtask2"]
    assert results["subtask3"]["official_score"] == 1.0, results["subtask3"]
