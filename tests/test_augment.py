"""Tests for grace.track1.augment oversampling."""

from pathlib import Path

from grace.io.loaders import load_track1
from grace.track1.augment import get_rare_class_stats, oversample_rare_classes

_TRAIN = Path("downloaded_data/public_data/public_data/track_1_train.json")


def test_oversample_increases_case_count() -> None:
    train = load_track1(_TRAIN)
    augmented = oversample_rare_classes(train)
    assert len(augmented) > len(train)


def test_oversample_increases_majorclaim_count() -> None:
    train = load_track1(_TRAIN)
    before = get_rare_class_stats(train)
    augmented = oversample_rare_classes(train, majorclaim_factor=5)
    after = get_rare_class_stats(augmented)
    assert after["MajorClaim_entities"] > before["MajorClaim_entities"]


def test_oversample_preserves_original_cases() -> None:
    train = load_track1(_TRAIN)
    augmented = oversample_rare_classes(train)
    # Every original case should be present at least once
    orig_ids = {c.id for c in train}
    aug_ids = {c.id for c in augmented}
    assert orig_ids == aug_ids


def test_oversample_factor_1_is_identity() -> None:
    train = load_track1(_TRAIN)
    augmented = oversample_rare_classes(
        train,
        majorclaim_factor=1,
        attack_factor=1,
        partial_attack_factor=1,
    )
    assert len(augmented) == len(train)


def test_get_rare_class_stats_matches_audit() -> None:
    train = load_track1(_TRAIN)
    stats = get_rare_class_stats(train)
    # From our data audit: MajorClaim=64, Attack=36, Partial-Attack=169
    assert stats["MajorClaim_entities"] == 64
    assert stats["Attack_relations"] == 36
    # Note: Partial-Attack count may differ slightly due to dangling relation
    # drops, but should be close to 169
    assert stats["Partial-Attack_relations"] >= 165
