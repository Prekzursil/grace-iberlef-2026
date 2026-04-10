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
from grace.track1.component_tagger import (
    ComponentTagger,
    ComponentTaggerConfig,
    FocalLoss,
)

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


def test_tagger_predict_outputs_empty_relations(tiny_case: GraceCase) -> None:
    """Tagger must output relations=() — never leak gold relations from input."""
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    tagger.model.to("cpu")
    preds = tagger.predict([tiny_case])
    assert preds[0].relations == ()


def test_focal_loss_runs_and_returns_finite() -> None:
    """Focal loss should produce finite loss values."""
    logits = torch.randn(10, 7)
    targets = torch.randint(0, 7, (10,))
    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_focal_loss_with_gamma_zero_matches_ce() -> None:
    """FocalLoss(gamma=0) should behave like cross-entropy."""
    torch.manual_seed(42)
    logits = torch.randn(20, 7)
    targets = torch.randint(0, 7, (20,))
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    assert abs(fl(logits, targets).item() - ce(logits, targets).item()) < 1e-5


def test_tagger_with_focal_loss_trains(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        loss_type="focal",
        focal_gamma=2.5,
        class_weights={"O": 0.25, "Premise": 1.0, "Claim": 2.0, "MajorClaim": 8.0},
    )
    tagger = ComponentTagger(cfg)
    tagger.to(torch.device("cpu"))
    loss = tagger.train_step([tiny_case], device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))


def test_tagger_with_crf_trains(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        use_crf=True,
    )
    tagger = ComponentTagger(cfg)
    tagger.to(torch.device("cpu"))
    loss = tagger.train_step([tiny_case], device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))


def test_tagger_with_crf_predicts(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        use_crf=True,
    )
    tagger = ComponentTagger(cfg)
    tagger.to(torch.device("cpu"))
    preds = tagger.predict([tiny_case])
    assert len(preds) == 1
    assert isinstance(preds[0].entities, tuple)
    assert preds[0].relations == ()


def test_tagger_with_bilstm_trains(tiny_case: GraceCase) -> None:
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        use_bilstm=True,
        bilstm_hidden=32,
    )
    tagger = ComponentTagger(cfg)
    tagger.to(torch.device("cpu"))
    loss = tagger.train_step([tiny_case], device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))


def test_tagger_with_crf_and_bilstm_and_focal(tiny_case: GraceCase) -> None:
    """Full stack: focal loss + BiLSTM + CRF."""
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        loss_type="focal",
        focal_gamma=2.5,
        use_crf=True,
        use_bilstm=True,
        bilstm_hidden=32,
    )
    tagger = ComponentTagger(cfg)
    tagger.to(torch.device("cpu"))
    loss = tagger.train_step([tiny_case], device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))
    preds = tagger.predict([tiny_case])
    assert len(preds) == 1
    assert preds[0].relations == ()


def test_tagger_parameters_includes_custom_head() -> None:
    """When CRF/BiLSTM enabled, parameters() should include all components."""
    cfg = ComponentTaggerConfig(
        backbone=_SMOKE_BACKBONE,
        max_length=64,
        stride=16,
        use_crf=True,
        use_bilstm=True,
        bilstm_hidden=32,
    )
    tagger = ComponentTagger(cfg)
    params = tagger.parameters()
    # Should include encoder + BiLSTM + classifier + CRF params
    param_count = sum(p.numel() for p in params)
    # encoder-only param count
    encoder_count = sum(p.numel() for p in tagger.model.parameters())
    assert param_count > encoder_count


def test_class_weights_default_to_ones() -> None:
    cfg = ComponentTaggerConfig(backbone=_SMOKE_BACKBONE, max_length=64, stride=16)
    tagger = ComponentTagger(cfg)
    weights = tagger._class_weights_tensor(torch.device("cpu"))
    assert torch.all(weights == 1.0)
