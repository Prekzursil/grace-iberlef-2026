"""Track 2 Subtask 1: BETO sentence relevance classifier.

Binary classifier that predicts whether each sentence in a MIR clinical
case is 'relevant' (contains evidence) or 'not-relevant'. Input encoding
conditions on the correct answer option text — a sentence is 'relevant'
if it contains evidence FOR the correct diagnosis, so knowing the answer
is the strongest signal available.

Input: [CLS] <context> [SEP] <sentence> [SEP] <correct_option_text> [SEP]
Output: binary logit (relevant / not-relevant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grace.io.schema import GraceCase

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class SentenceClassifierConfig:
    """Configuration for the BETO sentence relevance classifier."""

    backbone: str = "dccuchile/bert-base-spanish-wwm-cased"
    max_length: int = 256
    include_correct_option: bool = True


class SentenceClassifier:
    """Binary classifier for Track 2 sentence relevance detection."""

    def __init__(self, cfg: SentenceClassifierConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.backbone,
            num_labels=1,  # binary classification via BCEWithLogitsLoss
        )

    def _encode_sentence(
        self,
        case: GraceCase,
        sentence_idx: int,
    ) -> dict:
        """Encode one sentence with optional correct-option conditioning."""
        sentence = case.context_sentences[sentence_idx]

        if self.cfg.include_correct_option and case.correct_choice_id:
            # Find the correct option text
            correct_text = ""
            for ch in case.choices:
                if ch.id == case.correct_choice_id:
                    correct_text = ch.text
                    break
            text_a = sentence.sentence
            text_b = correct_text
        else:
            text_a = sentence.sentence
            text_b = None

        return self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

    def train_step(
        self,
        cases: Sequence[GraceCase],
        device: torch.device,
    ) -> float:
        """One forward+backward pass over all sentences in the batch of cases."""
        self.model.train()
        loss_fn = BCEWithLogitsLoss()
        total_loss = 0.0
        n = 0

        for case in cases:
            for i, label_str in enumerate(case.sentence_relevancy):
                enc = self._encode_sentence(case, i)
                enc = {k: v.to(device) for k, v in enc.items()}
                label = torch.tensor(
                    [[1.0 if label_str == "relevant" else 0.0]],
                    device=device,
                )
                logits = self.model(**enc).logits  # [1, 1]
                loss = loss_fn(logits, label)
                loss.backward()
                total_loss += float(loss.item())
                n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict(self, cases: Sequence[GraceCase]) -> tuple[GraceCase, ...]:
        """Predict sentence relevancy for each case.

        Returns new GraceCase objects with the ``sentence_relevancy``
        tuple filled in from model predictions. All other fields are
        preserved from the input cases.
        """
        self.model.train(False)
        out: list[GraceCase] = []

        for case in cases:
            labels: list[str] = []
            for i in range(len(case.context_sentences)):
                enc = self._encode_sentence(case, i)
                logits = self.model(**enc).logits  # [1, 1]
                prob = torch.sigmoid(logits).item()
                labels.append("relevant" if prob >= 0.5 else "not-relevant")

            out.append(
                GraceCase(
                    id=case.id,
                    raw_text=case.raw_text,
                    track=2,
                    metadata_context=case.metadata_context,
                    context_sentences=case.context_sentences,
                    choices=case.choices,
                    correct_choice_id=case.correct_choice_id,
                    sentence_relevancy=tuple(labels),
                    entities=case.entities,
                    relations=case.relations,
                )
            )
        return tuple(out)
