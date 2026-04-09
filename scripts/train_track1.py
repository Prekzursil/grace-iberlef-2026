#!/usr/bin/env python3
"""Track 1 training CLI - XLM-R BIO tagger for GRACE @ IberLEF 2026.

Reads a YAML config, trains a ``ComponentTagger`` on Track 1 train data,
evaluates on dev, writes submission-ready predictions, runs the official
scorer, and appends a row to ``experiments/ledger.jsonl``.

Usage:
    python scripts/train_track1.py --config configs/track1/xlmr_base.yaml --seed 42
    python scripts/train_track1.py --config configs/track1/beto_base.yaml \\
        --seed 42 --out experiments/runs

Relation classifier is added in Phase 3 Task 3.4 (separate file). This
CLI currently runs Subtask 1 only; Subtask 2 scores will be 0 until the
relation classifier is wired in.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from grace.eval.scorer import score_track1_from_file, scorer_fingerprint
from grace.eval.tracker import LedgerEntry, append_entry, sha256_file
from grace.io.loaders import load_track1
from grace.submit.formatter import format_submission
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(config: Path, seed: int, out: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))
    _set_seed(seed)

    ts = dt.datetime.now(dt.UTC).isoformat().replace(":", "-")
    tag = f"{cfg['tag']}-seed{seed}"
    run_dir = out / f"{ts}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")

    train_path = Path(cfg["data"]["train"])
    dev_path = Path(cfg["data"]["dev"])
    train_cases = list(load_track1(train_path))
    dev_cases = list(load_track1(dev_path))
    print(f"Loaded {len(train_cases)} train cases, {len(dev_cases)} dev cases")

    tagger = ComponentTagger(
        ComponentTaggerConfig(
            backbone=cfg["backbone"],
            max_length=cfg["task"]["subtask1"]["max_length"],
            stride=cfg["task"]["subtask1"]["stride"],
            class_weights=cfg["task"]["subtask1"].get("class_weights"),
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tagger.model.to(device)
    print(f"Training on {device}")

    optim = torch.optim.AdamW(
        tagger.model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]
    for epoch in range(num_epochs):
        random.shuffle(train_cases)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_cases), batch_size):
            batch = train_cases[i : i + batch_size]
            optim.zero_grad()
            loss = tagger.train_step(batch, device=device)
            optim.step()
            epoch_loss += loss
            n_batches += 1
        print(f"epoch {epoch + 1}/{num_epochs}: mean loss = {epoch_loss / max(n_batches, 1):.4f}")

    # Predict + score on dev
    preds = tagger.predict(dev_cases)
    dev_pred_path = run_dir / "predictions_dev.json"
    format_submission(preds, dev_pred_path, track=1)
    results = score_track1_from_file(dev_pred_path)

    metrics = {
        "subtask1_official": results["subtask1"]["official_score"],
        "subtask2_official": results["subtask2"]["official_score"],
        "overall": (results["subtask1"]["official_score"] + results["subtask2"]["official_score"])
        / 2,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps({"dev": results, "summary": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save the trained model for the relation classifier phase to reuse
    tagger.save(str(run_dir / "checkpoint"))

    # Append ledger row
    append_entry(
        Path("experiments") / "ledger.jsonl",
        LedgerEntry(
            tag=tag,
            git_sha=_git_sha(),
            track=1,
            subtask="subtask1",  # relation classifier wired in separately
            backbone=cfg["backbone"],
            config_path=str(config),
            scorer_sha256=scorer_fingerprint(1),
            dev_metrics=metrics,
            train_data_sha256=sha256_file(train_path),
            dev_data_sha256=sha256_file(dev_path),
            notes=cfg.get("notes", ""),
        ),
    )
    print(f"run complete. metrics: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit(f"config does not exist: {args.config}")
    train(args.config, args.seed, args.out)


if __name__ == "__main__":
    main()
