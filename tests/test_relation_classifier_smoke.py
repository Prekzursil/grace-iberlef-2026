"""Smoke tests for Track 1 pairwise relation classifier."""

import torch

from grace.io.schema import GraceCase, GraceEntity, GraceRelation
from grace.track1.relation_classifier import (
    RelationClassifier,
    RelationClassifierConfig,
)

_SMOKE_BACKBONE = "distilbert-base-multilingual-cased"


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


def test_relation_classifier_instantiates() -> None:
    cfg = RelationClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = RelationClassifier(cfg)
    assert clf.num_labels == 4  # no-relation, Support, Attack, Partial-Attack


def test_relation_classifier_train_step() -> None:
    cfg = RelationClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = RelationClassifier(cfg)
    clf.model.to("cpu")
    loss = clf.train_step([_tiny_case()], device=torch.device("cpu"))
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_relation_classifier_predict() -> None:
    cfg = RelationClassifierConfig(backbone=_SMOKE_BACKBONE, max_length=128)
    clf = RelationClassifier(cfg)
    clf.model.to("cpu")
    preds = clf.predict([_tiny_case()])
    assert len(preds) == 1
    assert isinstance(preds[0].relations, tuple)
    # Entities should be preserved from input
    assert len(preds[0].entities) == 2


def test_class_weights_applied() -> None:
    cfg = RelationClassifierConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=128,
        class_weights={
            "no-relation": 1.0,
            "Support": 1.0,
            "Attack": 15.0,
            "Partial-Attack": 5.0,
        },
    )
    clf = RelationClassifier(cfg)
    weights = clf._class_weights_tensor(torch.device("cpu"))
    assert weights.shape == (4,)
    assert float(weights[2]) == 15.0  # Attack
    assert float(weights[3]) == 5.0  # Partial-Attack
