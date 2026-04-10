"""Track 1 Subtask 2: pairwise relation classifier.

Classifies directed relations between argumentative components as
Support / Attack / Partial-Attack / no-relation. Uses typed marker
tokens to encode entity pairs within their context.

Input encoding (marker_typed scheme):
  [CLS] <raw_text_context> [SEP] <premise> span </premise> ... <claim> span </claim> [SEP]

The 4-class head uses class-weighted cross-entropy to compensate for
the severe imbalance: Support 86%, Partial-Attack 12%, Attack 2.5%.

Batched encoding: all entity pairs for a case are tokenized in one
call with padding, then a single forward pass replaces the O(N^2)
per-pair loop that was ~25 min on H100 for the full dev set.
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

    def _encode_pairs_batched(
        self,
        raw_text: str,
        pairs: Sequence[tuple[str, str, str, str]],
    ) -> dict[str, torch.Tensor]:
        """Batch-encode all entity pairs for a case in one tokenizer call.

        Each pair is (e1_text, e1_type, e2_text, e2_type).
        Returns padded tensors ready for a single forward pass.
        """
        if not pairs:
            return {}
        texts_a: list[str] = []
        texts_b: list[str] = []
        for e1_text, e1_type, e2_text, e2_type in pairs:
            left = f"<{e1_type.lower()}> {e1_text} </{e1_type.lower()}>"
            right = f"<{e2_type.lower()}> {e2_text} </{e2_type.lower()}>"
            texts_a.append(raw_text + " " + left)
            texts_b.append(right)
        return self.tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=True,
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
        """One forward+backward pass over entity pairs in the batch.

        Uses batched encoding per case for speed, but calls loss.backward()
        per case to preserve the gradient accumulation dynamics that proved
        optimal in the 13-experiment sweep.
        """
        self.model.train()
        loss_fn = CrossEntropyLoss(weight=self._class_weights_tensor(device))
        total = 0.0
        n = 0

        for case in cases:
            # Build positive and negative pairs
            positives: list[tuple[str, str, str, str, int]] = []
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
            pairs_with_labels = positives + negatives[:neg_keep]

            if not pairs_with_labels:
                continue

            # Batch encode all pairs for this case
            pair_tuples = [(e1t, e1ty, e2t, e2ty) for e1t, e1ty, e2t, e2ty, _ in pairs_with_labels]
            labels = torch.tensor(
                [lbl for _, _, _, _, lbl in pairs_with_labels],
                device=device,
            )
            enc = self._encode_pairs_batched(case.raw_text, pair_tuples)
            enc = {k: v.to(device) for k, v in enc.items()}

            # Single forward pass for all pairs in this case
            out = self.model(**enc)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            total += float(loss.item()) * len(pairs_with_labels)
            n += len(pairs_with_labels)

        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Predict relations for all entity pairs in each case (batched)."""
        self.model.train(False)
        device = next(self.model.parameters()).device
        out: list[GraceCase] = []

        for case in cases:
            # Build all directed pairs
            pair_info: list[tuple[str, str, str, str, str, str]] = []
            for e1 in case.entities:
                for e2 in case.entities:
                    if e1.id == e2.id:
                        continue
                    pair_info.append((e1.id, e2.id, e1.text, e1.type, e2.text, e2.type))

            if not pair_info:
                out.append(
                    GraceCase(
                        id=case.id,
                        raw_text=case.raw_text,
                        track=case.track,
                        entities=case.entities,
                        relations=(),
                    )
                )
                continue

            pair_tuples = [(e1t, e1ty, e2t, e2ty) for _, _, e1t, e1ty, e2t, e2ty in pair_info]
            enc = self._encode_pairs_batched(case.raw_text, pair_tuples)
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = self.model(**enc).logits
            pred_ids = logits.argmax(dim=-1).tolist()

            new_rels: list[GraceRelation] = []
            counter = 0
            for idx, (e1_id, e2_id, *_) in enumerate(pair_info):
                lbl = _ID2LABEL[pred_ids[idx]]
                if lbl == "no-relation":
                    continue
                counter += 1
                new_rels.append(
                    GraceRelation(
                        id=f"R{counter}",
                        arg1_id=e1_id,
                        arg2_id=e2_id,
                        relation_type=lbl,  # type: ignore[arg-type]
                    )
                )

            out.append(
                GraceCase(
                    id=case.id,
                    raw_text=case.raw_text,
                    track=case.track,
                    entities=case.entities,
                    relations=tuple(new_rels),
                )
            )
        return tuple(out)

    def save(self, save_dir: str) -> None:
        """Save the model + tokenizer to ``save_dir`` for resume / inference."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir: str, cfg: RelationClassifierConfig) -> RelationClassifier:
        """Load a previously-saved model into a fresh classifier instance."""
        clf = cls(cfg)
        clf.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        clf.tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        return clf
