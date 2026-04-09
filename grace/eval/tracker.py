"""Append-only JSONL experiment ledger.

Every training run writes exactly one line to ``experiments/ledger.jsonl``.
The paper's results table is a ``jq`` query against this file. The ledger is
append-only, always committed, and includes reproducibility fingerprints
(git SHA, scorer SHA, train/dev data SHA) so any past run can be reproduced
by checking out the recorded git SHA.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Compute the SHA256 of a file's contents for reproducibility fingerprints."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(slots=True)
class LedgerEntry:
    """One row in ``experiments/ledger.jsonl``."""

    tag: str
    git_sha: str
    track: int
    subtask: str
    backbone: str
    config_path: str
    scorer_sha256: str
    dev_metrics: dict[str, float]
    # Input-data fingerprints (design §5.3) — detects organizer data updates mid-competition
    train_data_sha256: str = ""
    dev_data_sha256: str = ""
    train_cost_minutes: float = 0.0
    llm_cost_usd: float = 0.0
    notes: str = ""
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())


def append_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    """Append one LedgerEntry as a single JSONL line. Never overwrites."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = asdict(entry)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
