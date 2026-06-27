"""Targeted coverage for model-layer branches not hit by the smoke tests.

Uses the small ``distilbert-base-multilingual-cased`` backbone everywhere
(fast, already cached by the smoke tests) and deterministic fake models for
prediction-threshold branches that depend on specific logits.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from grace.io.offsets import SpanAligner
from grace.io.schema import GraceCase, GraceEntity, GraceSentence

if TYPE_CHECKING:
    import pytest
from grace.track1.component_tagger import ComponentTagger, ComponentTaggerConfig
from grace.track1.nli_relation_classifier import NLIRelationClassifier, NLIRelationConfig
from grace.track1.relation_classifier import RelationClassifier, RelationClassifierConfig
from grace.track2.premise_extractor import PremiseExtractor
from grace.track2.sentence_clf import SentenceClassifier, SentenceClassifierConfig

_BACKBONE = "distilbert-base-multilingual-cased"
_CPU = torch.device("cpu")


def _t1_case(text: str) -> GraceCase:
    return GraceCase(
        id="m",
        raw_text=text,
        track=1,
        entities=(
            GraceEntity(id="T1", text=text[3:9], start=3, end=9, type="Premise"),
            GraceEntity(id="T2", text=text[10:20], start=10, end=20, type="Claim"),
        ),
    )


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Minimal stand-in exposing the surface predict() touches."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def train(self, _mode: bool) -> None:
        return None

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, **_enc) -> _FakeOutput:
        return _FakeOutput(self._logits)


# ── component_tagger: parameters() standard path (173->175, 175->177, 177->179) ──


def test_tagger_parameters_standard_model_only() -> None:
    tagger = ComponentTagger(ComponentTaggerConfig(backbone=_BACKBONE, max_length=32))
    params = tagger.parameters()
    assert sum(p.numel() for p in params) == sum(p.numel() for p in tagger.model.parameters())


# ── component_tagger: empty-text predict (CRF -> 334-343, non-CRF -> 384-393) ──


def test_tagger_predict_empty_text_non_crf() -> None:
    tagger = ComponentTagger(ComponentTaggerConfig(backbone=_BACKBONE, max_length=32, stride=8))
    tagger.to(_CPU)
    preds = tagger.predict([GraceCase(id="e", raw_text="", track=1)])
    assert preds[0].entities == ()


def test_tagger_predict_empty_text_crf() -> None:
    tagger = ComponentTagger(
        ComponentTaggerConfig(backbone=_BACKBONE, max_length=32, stride=8, use_crf=True)
    )
    tagger.to(_CPU)
    preds = tagger.predict([GraceCase(id="e", raw_text="", track=1)])
    assert preds[0].entities == ()


# ── component_tagger: multi-window predict (373, 377-378) and CRF multi (400-402) ──

_LONG = (
    "El cancer de mama es una enfermedad grave que afecta a muchas personas en todo el mundo "
    "y requiere tratamiento temprano para mejorar la supervivencia de los pacientes afectados."
)


def test_tagger_predict_multi_window_non_crf() -> None:
    tagger = ComponentTagger(ComponentTaggerConfig(backbone=_BACKBONE, max_length=16, stride=8))
    tagger.to(_CPU)
    preds = tagger.predict([GraceCase(id="long", raw_text=_LONG, track=1)])
    assert isinstance(preds[0].entities, tuple)


def test_tagger_predict_multi_window_crf() -> None:
    tagger = ComponentTagger(
        ComponentTaggerConfig(backbone=_BACKBONE, max_length=16, stride=8, use_crf=True)
    )
    tagger.to(_CPU)
    preds = tagger.predict([GraceCase(id="long", raw_text=_LONG, track=1)])
    assert isinstance(preds[0].entities, tuple)


# ── component_tagger: save/load standard + custom-head (426-478) ──


def test_tagger_save_load_standard(tmp_path) -> None:
    tagger = ComponentTagger(ComponentTaggerConfig(backbone=_BACKBONE, max_length=32))
    tagger.save(str(tmp_path / "std"))
    loaded = ComponentTagger.load(
        str(tmp_path / "std"), ComponentTaggerConfig(backbone=_BACKBONE, max_length=32)
    )
    assert loaded.model is not None
    assert (tmp_path / "std" / "tagger_config.json").exists()


def test_tagger_save_load_custom_head(tmp_path) -> None:
    cfg = ComponentTaggerConfig(
        backbone=_BACKBONE, max_length=32, use_crf=True, use_bilstm=True, bilstm_hidden=16
    )
    tagger = ComponentTagger(cfg)
    save_dir = str(tmp_path / "custom")
    tagger.save(save_dir)
    assert (tmp_path / "custom" / "classifier.pt").exists()
    assert (tmp_path / "custom" / "bilstm.pt").exists()
    assert (tmp_path / "custom" / "crf.pt").exists()
    loaded = ComponentTagger.load(save_dir, cfg)
    assert loaded.crf is not None
    assert loaded.bilstm is not None


def test_tagger_save_load_crf_only(tmp_path) -> None:
    """CRF without BiLSTM: save skips bilstm.pt; load skips the bilstm branch."""
    cfg = ComponentTaggerConfig(backbone=_BACKBONE, max_length=32, use_crf=True)
    save_dir = str(tmp_path / "crf_only")
    ComponentTagger(cfg).save(save_dir)
    assert not (tmp_path / "crf_only" / "bilstm.pt").exists()
    assert (tmp_path / "crf_only" / "crf.pt").exists()
    loaded = ComponentTagger.load(save_dir, cfg)
    assert loaded.crf is not None
    assert loaded.bilstm is None


def test_tagger_save_load_bilstm_only(tmp_path) -> None:
    """BiLSTM without CRF: save skips crf.pt; load skips the crf branch."""
    cfg = ComponentTaggerConfig(
        backbone=_BACKBONE, max_length=32, use_bilstm=True, bilstm_hidden=16
    )
    save_dir = str(tmp_path / "bilstm_only")
    ComponentTagger(cfg).save(save_dir)
    assert (tmp_path / "bilstm_only" / "bilstm.pt").exists()
    assert not (tmp_path / "bilstm_only" / "crf.pt").exists()
    loaded = ComponentTagger.load(save_dir, cfg)
    assert loaded.bilstm is not None
    assert loaded.crf is None


def test_tagger_load_custom_head_missing_classifier_file(tmp_path) -> None:
    """Custom-head load tolerates a missing classifier.pt (465->469 false arc)."""
    cfg = ComponentTaggerConfig(
        backbone=_BACKBONE, max_length=32, use_crf=True, use_bilstm=True, bilstm_hidden=16
    )
    save_dir = tmp_path / "no_clf"
    ComponentTagger(cfg).save(str(save_dir))
    (save_dir / "classifier.pt").unlink()
    loaded = ComponentTagger.load(str(save_dir), cfg)
    assert loaded.classifier is not None  # falls back to fresh init


# ── offsets: encode_with_labels inner loop completing without break (129->127) ──


def test_encode_with_labels_entity_at_text_end() -> None:
    tok = AutoTokenizer.from_pretrained(_BACKBONE, use_fast=True)
    aligner = SpanAligner(hf_tokenizer=tok)
    text = "El cancer"  # entity covers the final token -> no token starts after e.end
    ent = GraceEntity(id="T1", text="cancer", start=3, end=9, type="Premise")
    enc = aligner.encode_with_labels(text, [ent], max_length=32)
    assert "labels" in enc


# ── nli_relation_classifier ──


def test_nli_encode_pairs_empty_returns_empty_dict() -> None:
    clf = NLIRelationClassifier(NLIRelationConfig(nli_backbone=_BACKBONE, max_length=32))
    assert clf._encode_pairs_batched("hi", []) == {}


def test_nli_train_step_single_entity_skips() -> None:
    clf = NLIRelationClassifier(NLIRelationConfig(nli_backbone=_BACKBONE, max_length=32))
    clf.model.to(_CPU)
    case = GraceCase(
        id="solo",
        raw_text="Una sola entidad.",
        track=1,
        entities=(GraceEntity(id="T1", text="Una", start=0, end=3, type="Premise"),),
    )
    assert clf.train_step([case], device=_CPU) == 0.0


def test_nli_predict_decision_branches() -> None:
    clf = NLIRelationClassifier(NLIRelationConfig(nli_backbone=_BACKBONE, max_length=32))
    case = GraceCase(
        id="trip",
        raw_text="A B C tres entidades aqui.",
        track=1,
        entities=(
            GraceEntity(id="T1", text="A", start=0, end=1, type="Premise"),
            GraceEntity(id="T2", text="B", start=2, end=3, type="Claim"),
            GraceEntity(id="T3", text="C", start=4, end=5, type="Premise"),
        ),
    )
    # 3 entities -> 6 directed pairs. Craft outcomes: Support, Attack, PA, neutral, ...
    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0],  # entailment -> Support
            [5.0, 0.0, 0.0],  # strong contradiction -> Attack
            [1.0, 0.0, 0.5],  # moderate contradiction -> Partial-Attack
            [0.0, 5.0, 0.0],  # neutral -> skipped
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    clf.model = _FakeModel(logits)
    preds = clf.predict([case])
    rel_types = {r.relation_type for r in preds[0].relations}
    assert "Support" in rel_types
    assert "Attack" in rel_types
    assert "Partial-Attack" in rel_types
    # neutral pairs produced no relation
    assert len(preds[0].relations) == 3


def test_nli_save_load_round_trip(tmp_path) -> None:
    cfg = NLIRelationConfig(nli_backbone=_BACKBONE, max_length=32)
    clf = NLIRelationClassifier(cfg)
    clf.save(str(tmp_path / "nli"))
    loaded = NLIRelationClassifier.load(str(tmp_path / "nli"), cfg)
    assert loaded.model is not None


# ── relation_classifier ──


def test_relation_encode_pair_single() -> None:
    clf = RelationClassifier(RelationClassifierConfig(backbone=_BACKBONE, max_length=32))
    enc = clf._encode_pair("contexto", "premisa", "Premise", "claim", "Claim")
    assert "input_ids" in enc


def test_relation_encode_pairs_empty_returns_empty_dict() -> None:
    clf = RelationClassifier(RelationClassifierConfig(backbone=_BACKBONE, max_length=32))
    assert clf._encode_pairs_batched("hi", []) == {}


def test_relation_train_step_single_entity_skips() -> None:
    clf = RelationClassifier(RelationClassifierConfig(backbone=_BACKBONE, max_length=32))
    clf.model.to(_CPU)
    case = GraceCase(
        id="solo",
        raw_text="Una sola entidad.",
        track=1,
        entities=(GraceEntity(id="T1", text="Una", start=0, end=3, type="Premise"),),
    )
    assert clf.train_step([case], device=_CPU) == 0.0


def test_relation_predict_no_entities() -> None:
    clf = RelationClassifier(RelationClassifierConfig(backbone=_BACKBONE, max_length=32))
    clf.model.to(_CPU)
    case = GraceCase(id="empty", raw_text="Nada.", track=1)
    preds = clf.predict([case])
    assert preds[0].relations == ()


def test_relation_predict_filters_no_relation() -> None:
    clf = RelationClassifier(RelationClassifierConfig(backbone=_BACKBONE, max_length=32))
    case = _t1_case("El cancer es grave ahora")
    # 2 pairs: first -> no-relation (filtered), second -> Support.
    clf.model = _FakeModel(torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]))
    preds = clf.predict([case])
    assert len(preds[0].relations) == 1
    assert preds[0].relations[0].relation_type == "Support"


def test_relation_save_load_round_trip(tmp_path) -> None:
    cfg = RelationClassifierConfig(backbone=_BACKBONE, max_length=32)
    clf = RelationClassifier(cfg)
    clf.save(str(tmp_path / "rel"))
    loaded = RelationClassifier.load(str(tmp_path / "rel"), cfg)
    assert loaded.model is not None


# ── premise_extractor: Levenshtein import failure (127-129) + long needle (140) ──


def test_fuzzy_find_without_levenshtein(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "Levenshtein", None)
    extractor = PremiseExtractor(max_fuzzy_distance=2)
    case = GraceCase(
        id="t",
        raw_text="La fiebre fue de 39 grados.",
        track=2,
        context_sentences=(GraceSentence("La fiebre fue de 39 grados.", 0, 27),),
        sentence_relevancy=("relevant",),
    )
    # "fibre" needs fuzzy; with Levenshtein unavailable it is dropped.
    assert extractor.align_proposals(case, 0, ["fibre"]) == ()


def test_fuzzy_find_needle_longer_than_haystack() -> None:
    extractor = PremiseExtractor(max_fuzzy_distance=2)
    assert extractor._fuzzy_find("ab", "abcdefghij") is None


# ── sentence_clf: choice-loop branches in _encode_sentence (62->66, 63->62) ──


def _choice(cid: str, text: str):
    from grace.io.schema import GraceChoice

    return GraceChoice(id=cid, text=text, start=0, end=0)


def _t2_case(correct: str | None) -> GraceCase:
    return GraceCase(
        id="t2",
        raw_text="Frase relevante aqui.",
        track=2,
        context_sentences=(GraceSentence("Frase relevante aqui.", 0, 21),),
        choices=(_choice("A", "Opcion A"), _choice("B", "Opcion B")),
        correct_choice_id=correct,
        sentence_relevancy=("relevant",),
    )


def test_sentence_clf_encode_correct_is_later_choice() -> None:
    """correct_choice_id 'B' -> first choice skipped (63->62), second matches."""
    clf = SentenceClassifier(SentenceClassifierConfig(backbone=_BACKBONE, max_length=64))
    enc = clf._encode_sentence(_t2_case("B"), 0)
    assert "input_ids" in enc


def test_sentence_clf_encode_correct_not_in_choices() -> None:
    """correct_choice_id absent from choices -> loop exhausts (62->66)."""
    clf = SentenceClassifier(SentenceClassifierConfig(backbone=_BACKBONE, max_length=64))
    enc = clf._encode_sentence(_t2_case("ZZ"), 0)
    assert "input_ids" in enc
