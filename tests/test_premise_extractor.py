"""Tests for grace.track2.premise_extractor snap-to-offset alignment."""

from grace.io.schema import GraceCase, GraceSentence
from grace.track2.premise_extractor import PremiseExtractor


def _make_case(text: str) -> GraceCase:
    return GraceCase(
        id="t",
        raw_text=text,
        track=2,
        context_sentences=(GraceSentence(text, 0, len(text)),),
        sentence_relevancy=("relevant",),
    )


def test_extractor_snaps_exact_match() -> None:
    text = "La fiebre fue de 39 grados. El paciente tosia."
    case = GraceCase(
        id="t",
        raw_text=text,
        track=2,
        context_sentences=(
            GraceSentence("La fiebre fue de 39 grados.", 0, 27),
            GraceSentence("El paciente tosia.", 28, 46),
        ),
        sentence_relevancy=("relevant", "not-relevant"),
    )
    extractor = PremiseExtractor()
    entities = extractor.align_proposals(case, 0, ["fiebre fue de 39 grados"])
    assert len(entities) == 1
    assert text[entities[0].start : entities[0].end] == "fiebre fue de 39 grados"
    assert entities[0].type == "Premise"


def test_extractor_drops_unmatched_phrase() -> None:
    case = _make_case("La fiebre fue de 39 grados.")
    extractor = PremiseExtractor()
    entities = extractor.align_proposals(case, 0, ["paracetamol 500 mg"])
    assert entities == ()


def test_extractor_handles_empty_phrases() -> None:
    case = _make_case("La fiebre fue de 39 grados.")
    extractor = PremiseExtractor()
    entities = extractor.align_proposals(case, 0, ["", "  ", "fiebre"])
    # Only "fiebre" should match
    assert len(entities) == 1
    assert entities[0].text == "fiebre"


def test_extractor_fuzzy_matches_within_distance() -> None:
    case = _make_case("La fiebre fue de 39 grados.")
    extractor = PremiseExtractor(max_fuzzy_distance=2)
    # "fibre" is edit distance 2 from "fiebre" (missing 'e' and 'i'->missing)
    # Actually "fibre" vs "fiebre": delete 'e' at pos 2 = distance 1
    entities = extractor.align_proposals(case, 0, ["fibre"])
    # Should fuzzy-match to "fiebre" (distance 1 <= 2)
    assert len(entities) == 1


def test_extractor_assigns_sequential_ids() -> None:
    case = _make_case("fiebre alta y dolor abdominal intenso")
    extractor = PremiseExtractor()
    entities = extractor.align_proposals(case, 0, ["fiebre alta", "dolor abdominal"])
    assert len(entities) == 2
    assert entities[0].id == "P1"
    assert entities[1].id == "P2"
