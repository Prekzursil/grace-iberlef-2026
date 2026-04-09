"""HF tokenizer round-trip tests for SpanAligner BIO encoder/decoder.

Uses distilbert-base-multilingual-cased because it's small (<500MB),
has a fast tokenizer with offset_mapping support, and handles Spanish
accents correctly. ``transformers`` is a hard dependency so we import
it directly without ``pytest.importorskip``.
"""

import pytest
from transformers import AutoTokenizer

from grace.io.offsets import SpanAligner
from grace.io.schema import GraceEntity

_MODEL_NAME = "distilbert-base-multilingual-cased"


@pytest.fixture(scope="module")
def aligner() -> SpanAligner:
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)
    return SpanAligner(hf_tokenizer=tok)


def test_encode_with_labels_produces_bio_for_single_entity(aligner: SpanAligner) -> None:
    text = "El cáncer de mama es agresivo."
    entities = (GraceEntity(id="T1", text="cáncer de mama", start=3, end=17, type="Premise"),)
    enc = aligner.encode_with_labels(text, entities, max_length=64)
    assert "labels" in enc
    labels = aligner.id_to_label(enc["labels"])
    assert any(lbl == "B-Premise" for lbl in labels), labels
    assert any(lbl == "I-Premise" for lbl in labels), labels


def test_decode_bio_round_trip_recovers_entity(aligner: SpanAligner) -> None:
    """Encode gold entities as BIO, decode them back, verify text is preserved."""
    text = "El cáncer de mama es agresivo."
    gold = (GraceEntity(id="T1", text="cáncer de mama", start=3, end=17, type="Premise"),)
    enc = aligner.encode_with_labels(text, gold, max_length=64)
    recovered = aligner.decode_bio_to_entities(text, enc["labels"], enc["offset_mapping"])
    assert len(recovered) == 1
    rec = recovered[0]
    assert rec.type == "Premise"
    # Snap may adjust boundaries slightly; allow a small range but text must contain the gold
    assert "cáncer de mama" in text[rec.start : rec.end]


def test_encode_with_no_gold_entities_produces_all_o(aligner: SpanAligner) -> None:
    text = "El cáncer de mama es agresivo."
    enc = aligner.encode_with_labels(text, (), max_length=64)
    labels = aligner.id_to_label(enc["labels"])
    assert all(lbl == "O" for lbl in labels), labels


def test_num_labels_is_7(aligner: SpanAligner) -> None:
    assert aligner.num_labels == 7
