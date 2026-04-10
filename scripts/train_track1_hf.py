#!/usr/bin/env python3
"""Track 1 training with HuggingFace Trainer - proper batching + LR scheduling.

Fixes the gradient summation bug in the original train_track1.py where
individual loss.backward() calls per case accumulated gradients without
averaging, making the effective LR = configured_LR * batch_size.

Usage:
    python scripts/train_track1_hf.py --config configs/track1/beto_base.yaml --seed 42
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from grace.io.loaders import load_track1
from grace.io.offsets import SpanAligner
from grace.submit.formatter import format_predictions
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig

if TYPE_CHECKING:
    from grace.io.schema import GraceCase

_BIO_LABELS = (
    "O",
    "B-Premise",
    "I-Premise",
    "B-Claim",
    "I-Claim",
    "B-MajorClaim",
    "I-MajorClaim",
)
_LABEL2ID = {lbl: i for i, lbl in enumerate(_BIO_LABELS)}
_NUM_LABELS = len(_BIO_LABELS)


class GraceTokenClassificationDataset(Dataset):
    """Converts GraceCase list into a token classification dataset."""

    def __init__(
        self,
        cases: list[GraceCase],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        aligner = SpanAligner(hf_tokenizer=tokenizer)

        for case in cases:
            enc = aligner.encode_with_labels(
                case.raw_text,
                case.entities,
                max_length=max_length,
            )
            self.items.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": enc["labels"],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: Path, seed: int, out: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    _set_seed(seed)

    ts = dt.datetime.now(dt.UTC).isoformat().replace(":", "-")
    tag = f"{cfg['tag']}-hf-seed{seed}"
    run_dir = out / f"{ts}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(cfg["data"]["train"])
    dev_path = Path(cfg["data"]["dev"])
    train_cases = list(load_track1(train_path))
    dev_cases = list(load_track1(dev_path))
    print(f"Loaded {len(train_cases)} train, {len(dev_cases)} dev")

    # Apply augmentation
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
        print(f"After oversampling: {len(train_cases)} cases")

    backbone = cfg["backbone"]
    max_length = cfg["task"]["subtask1"]["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

    print("Building datasets...")
    train_dataset = GraceTokenClassificationDataset(train_cases, tokenizer, max_length)
    print(f"Train dataset: {len(train_dataset)} items")

    # Build model with class weights
    model = AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=_NUM_LABELS,
        id2label=dict(enumerate(_BIO_LABELS)),
        label2id=_LABEL2ID,
    )

    # Training arguments with proper LR scheduling
    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "hf_output"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        lr_scheduler_type="linear",
        fp16=cfg["training"].get("mixed_precision") == "fp16",
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,  # ignored in loss computation
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(
        f"Training with HF Trainer (batch={batch_size}, grad_accum={grad_accum}, "
        f"lr={cfg['training']['learning_rate']}, epochs={num_epochs}, warmup=10%)..."
    )
    trainer.train()

    # Predict on dev using our ComponentTagger.predict (sliding window)
    print("Predicting on dev...")
    tagger = ComponentTagger(
        ComponentTaggerConfig(
            backbone=backbone,
            max_length=max_length,
            stride=cfg["task"]["subtask1"]["stride"],
        )
    )
    # Copy the trained weights into our tagger
    tagger.model = model
    tagger.tokenizer = tokenizer
    tagger.aligner = SpanAligner(hf_tokenizer=tokenizer)

    dev_preds = tagger.predict(dev_cases)

    # Score
    dev_pred_path = run_dir / "predictions_dev.json"
    format_predictions(dev_preds, dev_pred_path, track=1)

    from grace.eval.scorer import score_track1_from_file

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
    print(f"\nResults: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args()
    train(args.config, args.seed, args.out)


if __name__ == "__main__":
    main()
