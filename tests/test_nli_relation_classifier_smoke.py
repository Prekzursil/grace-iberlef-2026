"""Smoke tests for the NLI-based relation classifier."""

import torch

from grace.io.schema import GraceCase, GraceEntity, GraceRelation
from grace.track1.nli_relation_classifier import (
    NLIRelationClassifier,
    NLIRelationConfig,
)

# Small NLI-capable model for smoke tests
_SMOKE_BACKBONE = "cross-encoder/nli-distilroberta-base"


def _tiny_case() -> GraceCase:
    text = "El cancer es grave. Los pacientes mueren rapidamente."
    return GraceCase(
        id="smoke",
        raw_text=text,
        track=1,
        entities=(
            GraceEntity(id="T1", text="cancer", start=3, end=9, type="Premise"),
            GraceEntity(
                id="T2",
                text="Los pacientes mueren rapidamente",
                start=20,
                end=52,
                type="Claim",
            ),
        ),
        relations=(GraceRelation(id="R1", arg1_id="T1", arg2_id="T2", relation_type="Support"),),
    )


def test_nli_classifier_instantiates() -> None:
    cfg = NLIRelationConfig(nli_backbone=_SMOKE_BACKBONE, max_length=128)
    clf = NLIRelationClassifier(cfg)
    assert clf.num_nli_labels == 3


def test_nli_classifier_train_step() -> None:
    cfg = NLIRelationConfig(nli_backbone=_SMOKE_BACKBONE, max_length=128)
    clf = NLIRelationClassifier(cfg)
    clf.model.to("cpu")
    loss = clf.train_step([_tiny_case()], device=torch.device("cpu"))
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_nli_classifier_predict() -> None:
    cfg = NLIRelationConfig(nli_backbone=_SMOKE_BACKBONE, max_length=128)
    clf = NLIRelationClassifier(cfg)
    clf.model.to("cpu")
    preds = clf.predict([_tiny_case()])
    assert len(preds) == 1
    assert isinstance(preds[0].relations, tuple)
    # Entities should be preserved from input
    assert len(preds[0].entities) == 2


def test_nli_classifier_predict_no_entities() -> None:
    """No entities → no relations, no crash."""
    case = GraceCase(id="empty", raw_text="No hay nada.", track=1)
    cfg = NLIRelationConfig(nli_backbone=_SMOKE_BACKBONE, max_length=128)
    clf = NLIRelationClassifier(cfg)
    clf.model.to("cpu")
    preds = clf.predict([case])
    assert len(preds) == 1
    assert preds[0].relations == ()


def test_nli_class_weights() -> None:
    cfg = NLIRelationConfig(
        nli_backbone=_SMOKE_BACKBONE,
        max_length=128,
        class_weights={
            "no-relation": 1.0,
            "Support": 1.0,
            "Attack": 15.0,
            "Partial-Attack": 5.0,
        },
    )
    clf = NLIRelationClassifier(cfg)
    weights = clf._nli_class_weights(torch.device("cpu"))
    assert weights.shape == (3,)
    # contradiction weight = max(Attack=15, Partial-Attack=5) = 15
    assert float(weights[0]) == 15.0
    # neutral = no-relation = 1.0
    assert float(weights[1]) == 1.0
    # entailment = Support = 1.0
    assert float(weights[2]) == 1.0
