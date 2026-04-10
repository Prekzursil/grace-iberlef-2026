"""NLI-based relation classifier for Track 1 Subtask 2.

Reframes Support/Attack classification as Natural Language Inference:
  Support      → entailment
  Attack       → contradiction
  Partial-Attack → contradiction (with lower confidence threshold)
  no-relation  → neutral

Pre-trained NLI models (trained on SNLI+MNLI+FEVER+ANLI) already encode
argumentative reasoning, giving a massive head start over randomly
initialized 4-class classifiers — especially with only 36 Attack examples.

Validated by HiTZ (GRACE organizers) on casiMedicos (Urruela et al. 2025).

Input encoding:
  Premise: "<e1_type> e1_text </e1_type> [SEP context]"
  Hypothesis: "<e2_type> e2_text </e2_type> supports or contradicts this."
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grace.io.schema import GraceCase, GraceRelation

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)

# NLI model output indices (standard for cross-encoder NLI models)
_NLI_CONTRADICTION = 0
_NLI_NEUTRAL = 1
_NLI_ENTAILMENT = 2

# GRACE relation labels (same as pairwise classifier for scoring compat)
_GRACE_LABELS: tuple[str, ...] = ("no-relation", "Support", "Attack", "Partial-Attack")
_GRACE_LABEL2ID = {lbl: i for i, lbl in enumerate(_GRACE_LABELS)}

# Mapping GRACE → NLI for training
_GRACE_TO_NLI = {
    "Support": _NLI_ENTAILMENT,
    "Attack": _NLI_CONTRADICTION,
    "Partial-Attack": _NLI_CONTRADICTION,  # trained as contradiction
    "no-relation": _NLI_NEUTRAL,
}


@dataclass(slots=True)
class NLIRelationConfig:
    """Configuration for the NLI-based relation classifier."""

    nli_backbone: str = "cross-encoder/nli-deberta-v3-small"
    max_length: int = 512
    # Threshold: contradiction logit must exceed this to predict Attack/PA
    partial_attack_threshold: float = 0.3
    # Class weights for NLI fine-tuning (3 classes: contradiction, neutral, entailment)
    class_weights: dict[str, float] | None = None
    negative_sampling_ratio: float = 3.0


class NLIRelationClassifier:
    """Relation classifier using NLI reframing (HiTZ method)."""

    num_nli_labels = 3  # contradiction, neutral, entailment

    def __init__(self, cfg: NLIRelationConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.nli_backbone, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.nli_backbone,
            num_labels=self.num_nli_labels,
        )

    def _format_premise_hypothesis(
        self,
        raw_text: str,
        e1_text: str,
        e1_type: str,
        e2_text: str,
        e2_type: str,
    ) -> tuple[str, str]:
        """Format entity pair as NLI premise-hypothesis.

        Premise: the source entity in context
        Hypothesis: whether the target entity supports or contradicts
        """
        premise = f"<{e1_type.lower()}> {e1_text} </{e1_type.lower()}> {raw_text}"
        hypothesis = (
            f'The {e2_type.lower()} "{e2_text}" supports or contradicts ' f"the {e1_type.lower()}."
        )
        return premise, hypothesis

    def _encode_pairs_batched(
        self,
        raw_text: str,
        pairs: Sequence[tuple[str, str, str, str]],
    ) -> dict[str, torch.Tensor]:
        """Batch-encode entity pairs as NLI premise-hypothesis pairs."""
        if not pairs:
            return {}
        premises: list[str] = []
        hypotheses: list[str] = []
        for e1_text, e1_type, e2_text, e2_type in pairs:
            p, h = self._format_premise_hypothesis(raw_text, e1_text, e1_type, e2_text, e2_type)
            premises.append(p)
            hypotheses.append(h)
        return self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=True,
            return_tensors="pt",
        )

    def _nli_class_weights(self, device: torch.device) -> torch.Tensor:
        """Class weights for NLI fine-tuning (3 classes)."""
        if not self.cfg.class_weights:
            return torch.ones(self.num_nli_labels, device=device)
        # Map GRACE relation weights to NLI classes
        w_contra = max(
            self.cfg.class_weights.get("Attack", 1.0),
            self.cfg.class_weights.get("Partial-Attack", 1.0),
        )
        w_neutral = self.cfg.class_weights.get("no-relation", 1.0)
        w_entail = self.cfg.class_weights.get("Support", 1.0)
        return torch.tensor([w_contra, w_neutral, w_entail], device=device)

    def train_step(
        self,
        cases: Sequence[GraceCase],
        device: torch.device,
    ) -> float:
        """One forward+backward pass. Relabels GRACE relations to NLI format."""
        self.model.train()
        loss_fn = CrossEntropyLoss(weight=self._nli_class_weights(device))
        total = 0.0
        n = 0

        for case in cases:
            positives: list[tuple[str, str, str, str, int]] = []
            negatives: list[tuple[str, str, str, str, int]] = []

            for e1 in case.entities:
                for e2 in case.entities:
                    if e1.id == e2.id:
                        continue
                    # Find gold relation
                    grace_lbl = "no-relation"
                    for r in case.relations:
                        if r.arg1_id == e1.id and r.arg2_id == e2.id:
                            grace_lbl = r.relation_type
                            break
                    nli_lbl = _GRACE_TO_NLI[grace_lbl]
                    item = (e1.text, e1.type, e2.text, e2.type, nli_lbl)
                    if grace_lbl == "no-relation":
                        negatives.append(item)
                    else:
                        positives.append(item)

            # Negative sampling
            random.shuffle(negatives)
            neg_keep = int(self.cfg.negative_sampling_ratio * max(1, len(positives)))
            pairs_with_labels = positives + negatives[:neg_keep]

            if not pairs_with_labels:
                continue

            pair_tuples = [(e1t, e1ty, e2t, e2ty) for e1t, e1ty, e2t, e2ty, _ in pairs_with_labels]
            labels = torch.tensor(
                [lbl for _, _, _, _, lbl in pairs_with_labels],
                device=device,
            )
            enc = self._encode_pairs_batched(case.raw_text, pair_tuples)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = self.model(**enc)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            total += float(loss.item()) * len(pairs_with_labels)
            n += len(pairs_with_labels)

        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Predict relations using NLI → GRACE label mapping."""
        self.model.train(False)
        device = next(self.model.parameters()).device
        out: list[GraceCase] = []

        for case in cases:
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
            probs = torch.softmax(logits, dim=-1)

            new_rels: list[GraceRelation] = []
            counter = 0
            for idx, (e1_id, e2_id, *_) in enumerate(pair_info):
                p_contra = float(probs[idx, _NLI_CONTRADICTION])
                p_entail = float(probs[idx, _NLI_ENTAILMENT])

                if p_entail > p_contra and p_entail > 0.5:
                    rel_type = "Support"
                elif p_contra > p_entail and p_contra > 0.5:
                    # High contradiction → Attack, moderate → Partial-Attack
                    if p_contra > (0.5 + self.cfg.partial_attack_threshold):
                        rel_type = "Attack"
                    else:
                        rel_type = "Partial-Attack"
                else:
                    continue  # neutral → no relation

                counter += 1
                new_rels.append(
                    GraceRelation(
                        id=f"R{counter}",
                        arg1_id=e1_id,
                        arg2_id=e2_id,
                        relation_type=rel_type,  # type: ignore[arg-type]
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
        """Save model + tokenizer."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir: str, cfg: NLIRelationConfig) -> NLIRelationClassifier:
        """Load a previously-saved NLI classifier."""
        clf = cls(cfg)
        clf.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        clf.tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        return clf
