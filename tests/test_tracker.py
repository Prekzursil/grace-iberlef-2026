"""Tests for grace.eval.tracker append-only ledger."""

import json
import tempfile
from pathlib import Path

from grace.eval.tracker import LedgerEntry, append_entry, sha256_file


def test_sha256_file_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "sample.txt"
        f.write_text("hello", encoding="utf-8")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
    assert h1 == h2
    assert len(h1) == 64


def test_append_writes_one_jsonl_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        ledger = Path(d) / "ledger.jsonl"
        entry = LedgerEntry(
            tag="test-run",
            git_sha="abc123",
            track=1,
            subtask="both",
            backbone="xlm-roberta-base",
            config_path="configs/track1/xlmr_base.yaml",
            scorer_sha256="deadbeef",
            dev_metrics={"overall": 0.4},
        )
        append_entry(ledger, entry)
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["tag"] == "test-run"
    assert row["track"] == 1
    assert row["dev_metrics"]["overall"] == 0.4
    assert "timestamp" in row


def test_append_multiple_entries_stays_append_only() -> None:
    with tempfile.TemporaryDirectory() as d:
        ledger = Path(d) / "ledger.jsonl"
        for i in range(3):
            append_entry(
                ledger,
                LedgerEntry(
                    tag=f"run-{i}",
                    git_sha="x",
                    track=1,
                    subtask="s1",
                    backbone="b",
                    config_path="c",
                    scorer_sha256="s",
                    dev_metrics={"overall": i / 10},
                ),
            )
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert [r["tag"] for r in rows] == ["run-0", "run-1", "run-2"]


def test_ledger_entry_data_sha256_fields_default_to_empty() -> None:
    entry = LedgerEntry(
        tag="t",
        git_sha="g",
        track=1,
        subtask="s",
        backbone="b",
        config_path="c",
        scorer_sha256="s",
        dev_metrics={},
    )
    assert entry.train_data_sha256 == ""
    assert entry.dev_data_sha256 == ""
