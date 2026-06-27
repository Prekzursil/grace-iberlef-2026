"""Coverage for grace.eval.diagnose.build_diagnostics."""

from grace.eval.diagnose import build_diagnostics
from grace.io.schema import GraceCase, GraceEntity


def _case(cid: str, ents: tuple[GraceEntity, ...]) -> GraceCase:
    return GraceCase(id=cid, raw_text="x" * 50, track=1, entities=ents)


def test_build_diagnostics_matches_misses_and_fps() -> None:
    gold = (
        _case(
            "c1",
            (
                GraceEntity(id="g1", text="a", start=0, end=5, type="Premise"),
                GraceEntity(id="g2", text="b", start=10, end=15, type="Claim"),
            ),
        ),
    )
    pred = (
        _case(
            "c1",
            (
                # exact match for g1
                GraceEntity(id="p1", text="a", start=0, end=5, type="Premise"),
                # spurious false positive
                GraceEntity(id="p2", text="c", start=20, end=25, type="Premise"),
            ),
        ),
    )
    diag = build_diagnostics(gold, pred, track=1)
    assert diag["track"] == 1
    assert diag["num_cases"] == 1
    assert 0.0 < diag["corpus_f1_mean"] < 1.0
    # g1 matched (Premise->Premise), g2 missed (Claim->MISS)
    assert diag["per_type_confusion"]["Premise"]["Premise"] == 1
    assert diag["per_type_confusion"]["Claim"]["MISS"] == 1
    assert diag["worst_cases"][0]["case_id"] == "c1"
    assert diag["length_vs_score"][0]["text_len"] == 50
    assert diag["offset_error_histogram"]["count"] == 0


def test_build_diagnostics_no_shared_ids_is_empty() -> None:
    gold = (_case("only_gold", ()),)
    pred = (_case("only_pred", ()),)
    diag = build_diagnostics(gold, pred, track=2)
    assert diag["num_cases"] == 0
    assert diag["corpus_f1_mean"] == 0.0
    assert diag["worst_cases"] == []


def test_build_diagnostics_perfect_match() -> None:
    ents = (GraceEntity(id="g1", text="a", start=0, end=5, type="Premise"),)
    diag = build_diagnostics((_case("c1", ents),), (_case("c1", ents),), track=1)
    assert diag["corpus_f1_mean"] == 1.0
