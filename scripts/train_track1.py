#!/usr/bin/env python3
"""Track 1 training CLI - full pipeline for GRACE @ IberLEF 2026.

Phase A: Train ComponentTagger (BIO token classification)
Phase B: Train RelationClassifier (pairwise relation classification)
Phase C: Predict on dev + score both subtasks

The training loop preserves the exact dynamics that scored 0.4037 F1:
- optim.zero_grad() at batch boundaries only
- tagger/clf.train_step() calls loss.backward() per case inside
- optim.step() after the batch
- Plain AdamW, constant LR, NO scheduler, NO warmup, NO decay

Usage:
    python scripts/train_track1.py --config configs/track1/beto_base.yaml --seed 42
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
from grace.submit.formatter import format_predictions
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig
from grace.track1.nli_relation_classifier import (
    NLIRelationClassifier,
    NLIRelationConfig,
)
from grace.track1.relation_classifier import (
    RelationClassifier,
    RelationClassifierConfig,
)


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

    # Apply augmentation if configured
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("enabled") and aug_cfg.get("strategy") == "oversample":
        from grace.track1.augment import oversample_rare_classes

        train_cases = list(
            oversample_rare_classes(
                tuple(train_cases),
                majorclaim_factor=aug_cfg.get("majorclaim_factor", 5),
                attack_factor=aug_cfg.get("attack_factor", 3),
                partial_attack_factor=aug_cfg.get("partial_attack_factor", 2),
            )
        )
        print(f"After oversampling: {len(train_cases)} train cases")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]

    # ── Phase A: Train ComponentTagger ──────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase A: Training ComponentTagger (BIO entity detection)")
    print(f"{'=' * 60}")

    s1_cfg = cfg["task"]["subtask1"]
    tagger = ComponentTagger(
        ComponentTaggerConfig(
            backbone=cfg["backbone"],
            max_length=s1_cfg["max_length"],
            stride=s1_cfg["stride"],
            class_weights=s1_cfg.get("class_weights"),
            loss_type=s1_cfg.get("loss", "ce"),
            focal_gamma=s1_cfg.get("focal_gamma", 2.0),
            use_crf=s1_cfg.get("use_crf", False),
            use_bilstm=s1_cfg.get("use_bilstm", False),
            bilstm_hidden=s1_cfg.get("bilstm_hidden", 256),
        )
    )
    tagger.to(device)
    print(f"Training on {device}")

    tagger_optim = torch.optim.AdamW(
        tagger.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    for epoch in range(num_epochs):
        random.shuffle(train_cases)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_cases), batch_size):
            batch = train_cases[i : i + batch_size]
            tagger_optim.zero_grad()
            loss = tagger.train_step(batch, device=device)
            tagger_optim.step()
            epoch_loss += loss
            n_batches += 1
        print(
            f"  [tagger] epoch {epoch + 1}/{num_epochs}: loss={epoch_loss / max(n_batches, 1):.4f}"
        )

    tagger.save(str(run_dir / "tagger"))

    # ── Phase B: Train RelationClassifier ───────────────────────────
    s2_cfg = cfg["task"].get("subtask2", {})
    rel_method = s2_cfg.get("method", "pairwise")
    rel_epochs = min(num_epochs, s2_cfg.get("rel_epochs", 5))

    print(f"\n{'=' * 60}")
    print(f"Phase B: Training RelationClassifier [{rel_method}] ({rel_epochs} epochs)")
    print(f"{'=' * 60}")

    if rel_method == "nli":
        rel_clf = NLIRelationClassifier(
            NLIRelationConfig(
                nli_backbone=s2_cfg.get("nli_backbone", "cross-encoder/nli-deberta-v3-small"),
                max_length=s2_cfg.get("max_length", 512),
                class_weights=s2_cfg.get("class_weights"),
                negative_sampling_ratio=s2_cfg.get("negative_sampling_ratio", 3.0),
                partial_attack_threshold=s2_cfg.get("partial_attack_threshold", 0.3),
            )
        )
    else:
        rel_clf = RelationClassifier(
            RelationClassifierConfig(
                backbone=cfg["backbone"],
                max_length=s2_cfg.get("max_length", 512),
                class_weights=s2_cfg.get("class_weights"),
                negative_sampling_ratio=s2_cfg.get("negative_sampling_ratio", 3.0),
            )
        )

    rel_clf.model.to(device)

    rel_optim = torch.optim.AdamW(
        rel_clf.model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    for epoch in range(rel_epochs):
        random.shuffle(train_cases)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_cases), batch_size):
            batch = train_cases[i : i + batch_size]
            rel_optim.zero_grad()
            loss = rel_clf.train_step(batch, device=device)
            rel_optim.step()
            epoch_loss += loss
            n_batches += 1
        print(
            f"  [rel_clf] epoch {epoch + 1}/{rel_epochs}: loss={epoch_loss / max(n_batches, 1):.4f}"
        )

    rel_clf.save(str(run_dir / "relation"))

    # ── Phase C: Predict + Score ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase C: Predicting on dev + scoring")
    print(f"{'=' * 60}")

    # Step 1: Entity predictions (tagger outputs relations=())
    entity_preds = tagger.predict(dev_cases)
    print(f"  Entity predictions: {sum(len(c.entities) for c in entity_preds)} entities")

    # Step 2: Relation predictions on top of predicted entities
    full_preds = rel_clf.predict(entity_preds)
    print(f"  Relation predictions: {sum(len(c.relations) for c in full_preds)} relations")

    # Step 3: Format + score
    dev_pred_path = run_dir / "predictions_dev.json"
    format_predictions(full_preds, dev_pred_path, track=1)
    results = score_track1_from_file(dev_pred_path, gold_path=dev_path)

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

    # Append ledger row
    append_entry(
        Path("experiments") / "ledger.jsonl",
        LedgerEntry(
            tag=tag,
            git_sha=_git_sha(),
            track=1,
            subtask="subtask1+subtask2",
            backbone=cfg["backbone"],
            config_path=str(config),
            scorer_sha256=scorer_fingerprint(1),
            dev_metrics=metrics,
            train_data_sha256=sha256_file(train_path),
            dev_data_sha256=sha256_file(dev_path),
            notes=cfg.get("notes", ""),
        ),
    )
    print(f"\nResults: {metrics}")
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
