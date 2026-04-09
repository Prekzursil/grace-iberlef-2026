#!/usr/bin/env python3
"""Kaggle P100 training entrypoint for GRACE Track 1.

This script is designed to run inside a Kaggle Notebook with GPU accelerator.
It reads a YAML config, trains the ComponentTagger, and saves checkpoints +
metrics to the Kaggle output directory for download.

Usage (in a Kaggle Notebook cell):
    !pip install -e /kaggle/input/grace-package/
    !python /kaggle/input/grace-package/scripts/kaggle_train.py \\
        --config /kaggle/input/grace-package/configs/track1/xlmr_large_base.yaml \\
        --seed 42 \\
        --out /kaggle/working/

Checkpoint/resume:
    If a previous run was interrupted (Kaggle 12h session limit), pass
    --resume /kaggle/working/<prev-run>/checkpoint/ to pick up from the
    last saved epoch. Checkpoints are saved after every epoch.

Workflow:
    1. Upload the grace/ package as a Kaggle Dataset named 'grace-package'
    2. Create a Kaggle Notebook, attach the dataset
    3. Enable P100 GPU accelerator
    4. Run this script via !python ...
    5. Download /kaggle/working/<run-dir>/checkpoint/ and metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from grace.io.loaders import load_track1
from grace.submit.formatter import format_submission
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_on_kaggle(config: Path, seed: int, out: Path, resume: Path | None) -> None:
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
    print(f"Loaded {len(train_cases)} train, {len(dev_cases)} dev cases")

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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} ({gpu_name})")

    optim = torch.optim.AdamW(
        tagger.model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    start_epoch = 0
    # Resume from checkpoint if provided
    if resume and resume.exists():
        ckpt = torch.load(resume / "training_state.pt", map_location=device, weights_only=False)
        tagger.model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
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
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}")

        # Save checkpoint every epoch (Kaggle sessions can die at 12h)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": tagger.model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_dir / "training_state.pt",
        )
        tagger.save(str(checkpoint_dir))

    # Final prediction + scoring on dev
    preds = tagger.predict(dev_cases)
    dev_pred_path = run_dir / "predictions_dev.json"
    format_submission(preds, dev_pred_path, track=1)

    # Import scorer if available (may not be on Kaggle if scorer files aren't uploaded)
    try:
        from grace.eval.scorer import score_track1_from_file

        results = score_track1_from_file(dev_pred_path)
        metrics = {
            "subtask1_official": results["subtask1"]["official_score"],
            "subtask2_official": results["subtask2"]["official_score"],
            "overall": (
                results["subtask1"]["official_score"] + results["subtask2"]["official_score"]
            )
            / 2,
        }
    except Exception as e:
        print(f"Scorer not available on Kaggle: {e}")
        metrics = {"note": "scorer not available, score locally after download"}

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Done. Metrics: {metrics}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Predictions: {dev_pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a previous checkpoint dir to resume from",
    )
    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    train_on_kaggle(args.config, args.seed, args.out, args.resume)


if __name__ == "__main__":
    main()
