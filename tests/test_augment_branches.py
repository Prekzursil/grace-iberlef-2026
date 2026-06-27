"""Branch coverage for grace.track1.augment reason-selection logic."""

from grace.io.schema import GraceCase, GraceEntity, GraceRelation
from grace.track1.augment import oversample_rare_classes


def _case(cid: str, ent_types: list[str], rel_types: list[str]) -> GraceCase:
    ents = tuple(
        GraceEntity(id=f"T{i}", text="a", start=0, end=1, type=t)  # type: ignore[arg-type]
        for i, t in enumerate(ent_types)
    )
    rels = tuple(
        GraceRelation(id=f"R{i}", arg1_id="T0", arg2_id="T0", relation_type=t)  # type: ignore[arg-type]
        for i, t in enumerate(rel_types)
    )
    return GraceCase(id=cid, raw_text="a", track=1, entities=ents, relations=rels)


def test_attack_without_majorclaim_sets_attack_reason() -> None:
    """Attack relation with no MajorClaim -> 'if not reason' true branch (line 72)."""
    case = _case("attack_only", ["Premise", "Claim"], ["Attack"])
    out = oversample_rare_classes([case], majorclaim_factor=5, attack_factor=3)
    # factor=3 -> 2 extra copies
    assert len(out) == 3


def test_partial_attack_with_majorclaim_skips_reason_assignment() -> None:
    """MajorClaim already set the reason; Partial-Attack hits the 77->79 skip."""
    case = _case("mc_and_pa", ["MajorClaim"], ["Partial-Attack"])
    out = oversample_rare_classes(
        [case], majorclaim_factor=5, attack_factor=3, partial_attack_factor=2
    )
    # MajorClaim factor (5) dominates -> 4 extra copies
    assert len(out) == 5


def test_no_rare_classes_is_identity() -> None:
    case = _case("plain", ["Premise"], ["Support"])
    out = oversample_rare_classes([case])
    assert len(out) == 1
