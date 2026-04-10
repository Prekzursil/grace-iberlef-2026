#!/usr/bin/env python3
"""Track 2 training CLI - full pipeline for GRACE @ IberLEF 2026.

Phase A: Train SentenceClassifier (binary relevance detection)
Phase B: Entity span detection (BIO tagger, reuses ComponentTagger architecture)
Phase C: Relation classification (pairwise or NLI-based)
Phase D: Predict on dev + score both subtasks

Training loop preserves the same gradient dynamics as Track 1.

Usage:
    python scripts/train_track2.py --config configs/track2/beto_base.yaml --seed 42
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

from grace.eval.scorer import score_track2_from_file
from grace.io.loaders import load_track2
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
from grace.track2.sentence_clf import SentenceClassifier, SentenceClassifierConfig


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


# Track 2 BIO labels (no MajorClaim)
_T2_LABELS = ("O", "B-Premise", "I-Premise", "B-Claim", "I-Claim")


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
    train_cases = list(load_track2(train_path))
    dev_cases = list(load_track2(dev_path))
    print(f"Loaded {len(train_cases)} train cases, {len(dev_cases)} dev cases")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]

    # ── Phase A: Train SentenceClassifier ───────────────────────────
    print(f"\n{'='*60}")
    print("Phase A: Training SentenceClassifier (binary relevance)")
    print(f"{'='*60}")

    sent_cfg = cfg["task"].get("subtask1", {})
    sent_clf = SentenceClassifier(
        SentenceClassifierConfig(
            backbone=cfg["backbone"],
            max_length=sent_cfg.get("max_length", 256),
            include_correct_option=sent_cfg.get("include_correct_option", True),
        )
    )
    sent_clf.model.to(device)
    print(f"Training on {device}")

    sent_epochs = sent_cfg.get("epochs", num_epochs)
    sent_optim = torch.optim.AdamW(
        sent_clf.model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    for epoch in range(sent_epochs):
        random.shuffle(train_cases)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_cases), batch_size):
            batch = train_cases[i : i + batch_size]
            sent_optim.zero_grad()
            loss = sent_clf.train_step(batch, device=device)
            sent_optim.step()
            epoch_loss += loss
            n_batches += 1
        print(
            f"  [sent_clf] epoch {epoch + 1}/{sent_epochs}: "
            f"loss={epoch_loss / max(n_batches, 1):.4f}"
        )

    # ── Phase B: Train Entity Tagger ────────────────────────────────
    s2_cfg = cfg["task"].get("subtask2", {})
    tagger_epochs = s2_cfg.get("tagger_epochs", num_epochs)

    print(f"\n{'='*60}")
    print(f"Phase B: Training Entity Tagger ({tagger_epochs} epochs)")
    print(f"{'='*60}")

    tagger = ComponentTagger(
        ComponentTaggerConfig(
            backbone=cfg["backbone"],
            max_length=s2_cfg.get("max_length", 512),
            stride=s2_cfg.get("stride", 128),
            class_weights=s2_cfg.get("class_weights"),
            loss_type=s2_cfg.get("loss", "ce"),
            focal_gamma=s2_cfg.get("focal_gamma", 2.0),
            use_crf=s2_cfg.get("use_crf", False),
            use_bilstm=s2_cfg.get("use_bilstm", False),
            bilstm_hidden=s2_cfg.get("bilstm_hidden", 256),
        )
    )
    tagger.to(device)

    tagger_optim = torch.optim.AdamW(
        tagger.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    for epoch in range(tagger_epochs):
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
            f"  [tagger] epoch {epoch + 1}/{tagger_epochs}: "
            f"loss={epoch_loss / max(n_batches, 1):.4f}"
        )

    # ── Phase C: Train Relation Classifier ──────────────────────────
    rel_method = s2_cfg.get("rel_method", "pairwise")
    rel_epochs = min(num_epochs, s2_cfg.get("rel_epochs", 5))

    print(f"\n{'='*60}")
    print(f"Phase C: Training RelationClassifier [{rel_method}] ({rel_epochs} epochs)")
    print(f"{'='*60}")

    if rel_method == "nli":
        rel_clf = NLIRelationClassifier(
            NLIRelationConfig(
                nli_backbone=s2_cfg.get("nli_backbone", "cross-encoder/nli-deberta-v3-small"),
                max_length=s2_cfg.get("max_length", 512),
                class_weights=s2_cfg.get("rel_class_weights"),
                negative_sampling_ratio=s2_cfg.get("negative_sampling_ratio", 3.0),
            )
        )
    else:
        rel_clf = RelationClassifier(
            RelationClassifierConfig(
                backbone=cfg["backbone"],
                max_length=s2_cfg.get("max_length", 512),
                class_weights=s2_cfg.get("rel_class_weights"),
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
            f"  [rel_clf] epoch {epoch + 1}/{rel_epochs}: "
            f"loss={epoch_loss / max(n_batches, 1):.4f}"
        )

    # ── Phase D: Predict + Score ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase D: Predicting on dev + scoring")
    print(f"{'='*60}")

    # Step 1: Sentence relevancy predictions
    sent_preds = sent_clf.predict(dev_cases)
    n_relevant = sum(sum(1 for s in c.sentence_relevancy if s == "relevant") for c in sent_preds)
    print(f"  Sentence predictions: {n_relevant} relevant sentences")

    # Step 2: Entity predictions
    entity_preds = tagger.predict(sent_preds)
    print(f"  Entity predictions: {sum(len(c.entities) for c in entity_preds)} entities")

    # Step 3: Relation predictions
    full_preds = rel_clf.predict(entity_preds)
    print(f"  Relation predictions: {sum(len(c.relations) for c in full_preds)} relations")

    # Step 4: Format + score
    dev_pred_path = run_dir / "predictions_dev.json"
    format_predictions(full_preds, dev_pred_path, track=2)

    try:
        results = score_track2_from_file(dev_pred_path, gold_path=dev_path)
        metrics = {
            "subtask1_official": results.get("subtask1", {}).get("official_score", 0.0),
            "subtask2_official": results.get("subtask2", {}).get("official_score", 0.0),
            "overall": (
                results.get("subtask1", {}).get("official_score", 0.0)
                + results.get("subtask2", {}).get("official_score", 0.0)
            )
            / 2,
        }
    except Exception as e:
        print(f"  WARNING: Scoring failed ({e}). Recording zeros.")
        metrics = {"subtask1_official": 0.0, "subtask2_official": 0.0, "overall": 0.0}
        results = {}

    (run_dir / "metrics.json").write_text(
        json.dumps({"dev": results, "summary": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
