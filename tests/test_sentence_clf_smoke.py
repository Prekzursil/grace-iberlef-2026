"""Smoke tests for Track 2 BETO sentence classifier."""

from pathlib import Path

import torch

from grace.io.loaders import load_track2
from grace.track2.sentence_clf import SentenceClassifier, SentenceClassifierConfig

_SMOKE_BACKBONE = "distilbert-base-multilingual-cased"
_DEV = Path("downloaded_data/public_data/public_data/track_2_dev.json")


def test_sentence_clf_instantiates() -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    assert clf.model is not None


def test_sentence_clf_train_step_runs() -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    clf.model.to("cpu")
    cases = load_track2(_DEV)[:2]
    loss = clf.train_step(cases, device=torch.device("cpu"))
    assert loss > 0


def test_sentence_clf_predict_returns_correct_labels() -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    clf.model.to("cpu")
    cases = load_track2(_DEV)[:2]
    preds = clf.predict(cases)
    assert len(preds) == 2
    for p, orig in zip(preds, cases, strict=False):
        # Same number of sentence labels as context_sentences
        assert len(p.sentence_relevancy) == len(orig.context_sentences)
        # All labels are valid
        assert all(lbl in {"relevant", "not-relevant"} for lbl in p.sentence_relevancy)
        # Other fields preserved
        assert p.id == orig.id
        assert p.track == 2
        assert p.correct_choice_id == orig.correct_choice_id
