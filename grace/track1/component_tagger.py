"""Track 1 component tagger - BIO token classification over Premise/Claim/MajorClaim.

Uses a HuggingFace ``AutoModelForTokenClassification`` with 7 output classes
(O + B/I x 3 entity types) and a class-weighted cross-entropy loss to
compensate for the severe rare-class imbalance in the GRACE 2026 training
data (MajorClaim ~3%, see design doc Appendix A).

Inference uses sliding-window decoding with stride=128 and mean-logit
pooling across overlap regions so long abstracts (up to ~1400 tokens) are
not silently truncated. The logit averaging approach preserves BIO
adjacency across window boundaries - see design doc section 3.2 and the
plan-review-gate round 2 fix for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoTokenizer

from grace.io.offsets import SpanAligner
from grace.io.schema import GraceCase

if TYPE_CHECKING:
    from collections.abc import Sequence


_LABELS: tuple[str, ...] = (
    "O",
    "B-Premise",
    "I-Premise",
    "B-Claim",
    "I-Claim",
    "B-MajorClaim",
    "I-MajorClaim",
)
_LABEL2ID = {lbl: i for i, lbl in enumerate(_LABELS)}


@dataclass(slots=True)
class ComponentTaggerConfig:
    """Configuration for :class:`ComponentTagger`."""

    backbone: str = "xlm-roberta-base"
    max_length: int = 512
    stride: int = 128
    class_weights: dict[str, float] | None = None


class ComponentTagger:
    """Fine-tunes a HuggingFace token classification model for Track 1 BIO tagging."""

    num_labels = len(_LABELS)

    def __init__(self, cfg: ComponentTaggerConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
        self.aligner = SpanAligner(hf_tokenizer=self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(
            cfg.backbone,
            num_labels=self.num_labels,
            id2label=dict(enumerate(_LABELS)),
            label2id=_LABEL2ID,
        )

    def _class_weights_tensor(self, device: torch.device) -> torch.Tensor:
        if not self.cfg.class_weights:
            return torch.ones(self.num_labels, device=device)
        weights = torch.ones(self.num_labels, device=device)
        for i, lbl in enumerate(_LABELS):
            if lbl == "O":
                weights[i] = self.cfg.class_weights.get("O", 1.0)
            else:
                etype = lbl.split("-", 1)[1]
                weights[i] = self.cfg.class_weights.get(etype, 1.0)
        return weights

    def train_step(
        self,
        cases: Sequence[GraceCase],
        device: torch.device,
    ) -> float:
        """Run one forward+backward pass over a batch of cases. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        for case in cases:
            enc = self.aligner.encode_with_labels(
                case.raw_text,
                case.entities,
                max_length=self.cfg.max_length,
            )
            input_ids = torch.tensor([enc["input_ids"]], device=device)
            attn = torch.tensor([enc["attention_mask"]], device=device)
            labels = torch.tensor([enc["labels"]], device=device)
            outputs = self.model(input_ids=input_ids, attention_mask=attn)
            loss_fn = CrossEntropyLoss(weight=self._class_weights_tensor(device))
            loss = loss_fn(
                outputs.logits.view(-1, self.num_labels),
                labels.view(-1),
            )
            loss.backward()
            total_loss += float(loss.item())
        return total_loss / max(len(cases), 1)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Sliding-window inference with mean-logit pooling across overlaps."""
        self.model.train(False)  # switch to inference mode (no dropout)
        out: list[GraceCase] = []
        device = next(self.model.parameters()).device

        for case in cases:
            enc = self.tokenizer(
                case.raw_text,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=self.cfg.stride,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # [num_windows, seq_len, num_labels]

            # Aggregate logits per (char_start, char_end) across windows
            summed: dict[tuple[int, int], torch.Tensor] = {}
            counts: dict[tuple[int, int], int] = {}
            num_windows, seq_len, _ = logits.shape
            for w in range(num_windows):
                for pos in range(seq_len):
                    s = int(enc["offset_mapping"][w][pos][0])
                    t = int(enc["offset_mapping"][w][pos][1])
                    if s == 0 and t == 0:
                        continue
                    if int(enc["attention_mask"][w][pos]) == 0:
                        continue
                    key = (s, t)
                    tok_logits = logits[w][pos].detach().cpu()
                    if key in summed:
                        summed[key] = summed[key] + tok_logits
                        counts[key] += 1
                    else:
                        summed[key] = tok_logits
                        counts[key] = 1

            if not summed:
                out.append(
                    GraceCase(
                        id=case.id,
                        raw_text=case.raw_text,
                        track=1,
                        entities=(),
                        relations=case.relations,
                    )
                )
                continue

            ordered_keys = sorted(summed.keys())
            averaged = torch.stack([summed[k] / counts[k] for k in ordered_keys])
            label_ids = averaged.argmax(dim=-1).tolist()
            offset_mapping = list(ordered_keys)

            entities = self.aligner.decode_bio_to_entities(
                case.raw_text,
                label_ids,
                offset_mapping,
            )
            out.append(
                GraceCase(
                    id=case.id,
                    raw_text=case.raw_text,
                    track=1,
                    entities=entities,
                    relations=case.relations,
                )
            )
        return tuple(out)

    def save(self, save_dir: str) -> None:
        """Save the model + tokenizer to ``save_dir`` for resume / inference."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir: str, cfg: ComponentTaggerConfig) -> ComponentTagger:
        """Load a previously-saved model into a fresh tagger instance."""
        tagger = cls(cfg)
        tagger.model = AutoModelForTokenClassification.from_pretrained(save_dir)
        tagger.tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        tagger.aligner = SpanAligner(hf_tokenizer=tagger.tokenizer)
        return tagger
