"""Track 1 Subtask 2: pairwise relation classifier.

Classifies directed relations between argumentative components as
Support / Attack / Partial-Attack / no-relation. Uses typed marker
tokens to encode entity pairs within their context.

Input encoding (marker_typed scheme):
  [CLS] <raw_text_context> [SEP] <premise> span </premise> ... <claim> span </claim> [SEP]

The 4-class head uses class-weighted cross-entropy to compensate for
the severe imbalance: Support 86%, Partial-Attack 12%, Attack 2.5%.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grace.io.schema import GraceCase, GraceRelation

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)

_LABELS: tuple[str, ...] = ("no-relation", "Support", "Attack", "Partial-Attack")
_LABEL2ID = {lbl: i for i, lbl in enumerate(_LABELS)}
_ID2LABEL = dict(enumerate(_LABELS))


@dataclass(slots=True)
class RelationClassifierConfig:
    """Configuration for the pairwise relation classifier."""

    backbone: str = "xlm-roberta-base"
    max_length: int = 512
    class_weights: dict[str, float] | None = None
    negative_sampling_ratio: float = 3.0


class RelationClassifier:
    """Pairwise relation classifier for Track 1 Subtask 2."""

    num_labels = len(_LABELS)

    def __init__(self, cfg: RelationClassifierConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.backbone,
            num_labels=self.num_labels,
            id2label=_ID2LABEL,
            label2id=_LABEL2ID,
        )

    def _encode_pair(
        self,
        raw_text: str,
        e1_text: str,
        e1_type: str,
        e2_text: str,
        e2_type: str,
    ) -> dict[str, Any]:
        """Encode a component pair with typed markers."""
        left = f"<{e1_type.lower()}> {e1_text} </{e1_type.lower()}>"
        right = f"<{e2_type.lower()}> {e2_text} </{e2_type.lower()}>"
        return self.tokenizer(
            raw_text + " " + left,
            right,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

    def _pair_label(
        self,
        rels: tuple[GraceRelation, ...],
        e1_id: str,
        e2_id: str,
    ) -> int:
        """Look up the gold relation label for an entity pair."""
        for r in rels:
            if r.arg1_id == e1_id and r.arg2_id == e2_id:
                return _LABEL2ID[r.relation_type]
        return _LABEL2ID["no-relation"]

    def _class_weights_tensor(self, device: torch.device) -> torch.Tensor:
        if not self.cfg.class_weights:
            return torch.ones(self.num_labels, device=device)
        return torch.tensor(
            [self.cfg.class_weights.get(lbl, 1.0) for lbl in _LABELS],
            device=device,
        )

    def train_step(
        self,
        cases: Sequence[GraceCase],
        device: torch.device,
    ) -> float:
        """One forward+backward pass over entity pairs in the batch."""
        self.model.train()
        loss_fn = CrossEntropyLoss(weight=self._class_weights_tensor(device))
        total = 0.0
        n = 0

        for case in cases:
            # Build positive and negative pairs
            positives: list[tuple[str, str, str, str, int]] = (
                []
            )  # e1_text, e1_type, e2_text, e2_type, label
            negatives: list[tuple[str, str, str, str, int]] = []

            for e1 in case.entities:
                for e2 in case.entities:
                    if e1.id == e2.id:
                        continue
                    lbl = self._pair_label(case.relations, e1.id, e2.id)
                    item = (e1.text, e1.type, e2.text, e2.type, lbl)
                    if lbl == _LABEL2ID["no-relation"]:
                        negatives.append(item)
                    else:
                        positives.append(item)

            # Negative sampling
            random.shuffle(negatives)
            neg_keep = int(self.cfg.negative_sampling_ratio * max(1, len(positives)))
            pairs = positives + negatives[:neg_keep]

            for e1_text, e1_type, e2_text, e2_type, lbl in pairs:
                enc = self._encode_pair(case.raw_text, e1_text, e1_type, e2_text, e2_type)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = self.model(**enc)
                loss = loss_fn(out.logits, torch.tensor([lbl], device=device))
                loss.backward()
                total += float(loss.item())
                n += 1

        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Predict relations for all entity pairs in each case."""
        self.model.train(False)
        out: list[GraceCase] = []

        for case in cases:
            new_rels: list[GraceRelation] = []
            counter = 0
            for e1 in case.entities:
                for e2 in case.entities:
                    if e1.id == e2.id:
                        continue
                    enc = self._encode_pair(case.raw_text, e1.text, e1.type, e2.text, e2.type)
                    logits = self.model(**enc).logits
                    pred_id = int(logits.argmax(-1).item())
                    lbl = _ID2LABEL[pred_id]
                    if lbl == "no-relation":
                        continue
                    counter += 1
                    new_rels.append(
                        GraceRelation(
                            id=f"R{counter}",
                            arg1_id=e1.id,
                            arg2_id=e2.id,
                            relation_type=lbl,  # type: ignore[arg-type]
                        )
                    )
            out.append(
                GraceCase(
                    id=case.id,
                    raw_text=case.raw_text,
                    track=1,
                    entities=case.entities,
                    relations=tuple(new_rels),
                )
            )
        return tuple(out)
