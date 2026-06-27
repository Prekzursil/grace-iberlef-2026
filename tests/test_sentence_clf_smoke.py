"""Smoke tests for Track 2 BETO sentence classifier (synthetic fixtures)."""

import torch

from grace.track2.sentence_clf import SentenceClassifier, SentenceClassifierConfig

_SMOKE_BACKBONE = "distilbert-base-multilingual-cased"


def test_sentence_clf_instantiates() -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    assert clf.model is not None


def test_sentence_clf_train_step_runs(track2_cases) -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    clf.model.to("cpu")
    loss = clf.train_step(track2_cases, device=torch.device("cpu"))
    assert loss > 0


def test_sentence_clf_train_step_without_correct_option(track2_cases) -> None:
    """include_correct_option=False exercises the text_b=None encoding branch."""
    cfg = SentenceClassifierConfig(
        backbone=_SMOKE_BACKBONE, max_length=128, include_correct_option=False
    )
    clf = SentenceClassifier(cfg)
    clf.model.to("cpu")
    loss = clf.train_step(track2_cases, device=torch.device("cpu"))
    assert loss >= 0.0


def test_sentence_clf_predict_returns_correct_labels(track2_cases) -> None:
    cfg = SentenceClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = SentenceClassifier(cfg)
    clf.model.to("cpu")
    preds = clf.predict(track2_cases)
    assert len(preds) == len(track2_cases)
    for p, orig in zip(preds, track2_cases, strict=True):
        assert len(p.sentence_relevancy) == len(orig.context_sentences)
        assert all(lbl in {"relevant", "not-relevant"} for lbl in p.sentence_relevancy)
        assert p.id == orig.id
        assert p.track == 2
        assert p.correct_choice_id == orig.correct_choice_id
