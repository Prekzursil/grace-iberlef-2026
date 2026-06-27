"""Tests for grace.track1.augment oversampling (synthetic fixtures)."""

from grace.io.loaders import load_track1
from grace.track1.augment import get_rare_class_stats, oversample_rare_classes
from tests.conftest import TRACK1_FIXTURE


def test_oversample_increases_case_count() -> None:
    train = load_track1(TRACK1_FIXTURE)
    augmented = oversample_rare_classes(train)
    assert len(augmented) > len(train)


def test_oversample_increases_majorclaim_count() -> None:
    train = load_track1(TRACK1_FIXTURE)
    before = get_rare_class_stats(train)
    augmented = oversample_rare_classes(train, majorclaim_factor=5)
    after = get_rare_class_stats(augmented)
    assert after["MajorClaim_entities"] > before["MajorClaim_entities"]


def test_oversample_preserves_original_cases() -> None:
    train = load_track1(TRACK1_FIXTURE)
    augmented = oversample_rare_classes(train)
    orig_ids = {c.id for c in train}
    aug_ids = {c.id for c in augmented}
    assert orig_ids == aug_ids


def test_oversample_factor_1_is_identity() -> None:
    train = load_track1(TRACK1_FIXTURE)
    augmented = oversample_rare_classes(
        train,
        majorclaim_factor=1,
        attack_factor=1,
        partial_attack_factor=1,
    )
    assert len(augmented) == len(train)


def test_oversample_attack_only_case_uses_attack_factor() -> None:
    """A case with an Attack relation but no MajorClaim picks the attack factor."""
    train = load_track1(TRACK1_FIXTURE)
    # majorclaim_factor=1 so only the Attack/Partial-Attack reasons drive copies.
    augmented = oversample_rare_classes(
        train, majorclaim_factor=1, attack_factor=4, partial_attack_factor=3
    )
    assert len(augmented) > len(train)


def test_get_rare_class_stats_matches_fixture_composition() -> None:
    train = load_track1(TRACK1_FIXTURE)
    stats = get_rare_class_stats(train)
    assert stats["total_cases"] == 2
    assert stats["MajorClaim_entities"] == 1
    assert stats["Attack_relations"] == 1
    # c1's dangling Partial-Attack is dropped at load; c2 keeps one.
    assert stats["Partial-Attack_relations"] == 1
    assert stats["cases_with_MajorClaim"] == 1
    assert stats["cases_with_Attack"] == 1
    assert stats["cases_with_Partial-Attack"] == 1
