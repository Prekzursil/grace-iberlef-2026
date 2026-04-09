"""Smoke tests for the Track 1 component tagger.

These tests DO NOT train to convergence. They verify that:
- the module can be imported and instantiated
- a single training step runs without crashing
- prediction produces well-formed GraceCase objects

A heavier convergence test is only justified when we have a trained
checkpoint to compare against.
"""

from __future__ import annotations

import pytest
import torch

from grace.io.schema import GraceCase, GraceEntity
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig

# Small multilingual model for fast smoke tests — about 50 MB vs XLM-R-large's 2.2 GB
_SMOKE_BACKBONE = "distilbert-base-multilingual-cased"


@pytest.fixture
def tiny_case() -> GraceCase:
    text = "El cáncer es grave. Los pacientes mueren rápidamente."
    return GraceCase(
        id="smoke",
        raw_text=text,
        track=1,
        entities=(
            GraceEntity(id="T1", text="cáncer", start=3, end=9, type="Premise"),
            GraceEntity(
                id="T2",
                text="Los pacientes mueren rápidamente",
                start=20,
                end=52,
                type="Claim",
            ),
        ),
    )


def test_tagger_instantiates_from_config() -> None:
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    assert tagger.num_labels == 7  # O + B/I x 3


def test_tagger_one_training_step_does_not_crash(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    device = torch.device("cpu")
    tagger.model.to(device)
    loss = tagger.train_step([tiny_case], device=device)
    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))


def test_tagger_predict_returns_cases_with_entities_tuple(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    tagger.model.to("cpu")
    preds = tagger.predict([tiny_case])
    assert len(preds) == 1
    assert isinstance(preds[0].entities, tuple)
    assert preds[0].id == tiny_case.id
    assert preds[0].track == 1


def test_class_weights_applied_when_configured() -> None:
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        class_weights={"O": 1.0, "Premise": 2.0, "Claim": 3.0, "MajorClaim": 8.0},
    )
    tagger = ComponentTagger(cfg)
    weights = tagger._class_weights_tensor(torch.device("cpu"))
    assert weights.shape == (7,)
    # B-MajorClaim and I-MajorClaim should both be 8.0
    assert float(weights[5]) == 8.0  # B-MajorClaim
    assert float(weights[6]) == 8.0  # I-MajorClaim


def test_class_weights_default_to_ones() -> None:
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    weights = tagger._class_weights_tensor(torch.device("cpu"))
    assert torch.all(weights == 1.0)
