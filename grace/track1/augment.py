"""Track 1 data augmentation for rare classes.

Two strategies:
1. **Oversampling**: duplicate training cases that contain rare entity types
   (MajorClaim) or rare relation types (Attack, Partial-Attack) to balance
   the training distribution. Simple but effective baseline.

2. **Cross-lingual transfer** (Phase 5 full): translate rare-class instances
   from English/French/Italian AbstRCT to Spanish, project offsets, and
   filter with entailment. This is the paper's research contribution.

Strategy 1 is implemented here. Strategy 2 will be added in Task 5.2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grace.io.schema import GraceCase

_log = logging.getLogger(__name__)


def oversample_rare_classes(
    cases: Sequence[GraceCase],
    majorclaim_factor: int = 5,
    attack_factor: int = 3,
    partial_attack_factor: int = 2,
) -> tuple[GraceCase, ...]:
    """Duplicate cases containing rare entity/relation types.

    For each case that contains at least one MajorClaim entity, duplicate
    it ``majorclaim_factor`` times. Similarly for Attack and Partial-Attack
    relations. Cases matching multiple criteria get the highest factor.

    This is a simple baseline augmentation. It doesn't create new data —
    it just rebalances the training distribution so the model sees rare
    classes more often during training.

    Args:
        cases: Original training cases.
        majorclaim_factor: How many copies for MajorClaim cases.
        attack_factor: How many copies for Attack-relation cases.
        partial_attack_factor: How many copies for Partial-Attack cases.

    Returns:
        Augmented tuple of cases (original + duplicates).
    """
    augmented: list[GraceCase] = list(cases)
    stats = {"majorclaim": 0, "attack": 0, "partial_attack": 0}

    for case in cases:
        entity_types = {e.type for e in case.entities}
        relation_types = {r.relation_type for r in case.relations}

        # Determine the highest applicable factor
        factor = 1
        reason = ""

        if "MajorClaim" in entity_types:
            factor = max(factor, majorclaim_factor)
            reason = "MajorClaim"
            stats["majorclaim"] += 1

        if "Attack" in relation_types:
            factor = max(factor, attack_factor)
            if not reason:
                reason = "Attack"
            stats["attack"] += 1

        if "Partial-Attack" in relation_types:
            factor = max(factor, partial_attack_factor)
            if not reason:
                reason = "Partial-Attack"
            stats["partial_attack"] += 1

        # Add duplicates (factor - 1 copies since original is already in the list)
        if factor > 1:
            for _ in range(factor - 1):
                augmented.append(case)

    _log.info(
        "Oversampled: %d -> %d cases (MajorClaim: %d cases x%d, "
        "Attack: %d cases x%d, Partial-Attack: %d cases x%d)",
        len(cases),
        len(augmented),
        stats["majorclaim"],
        majorclaim_factor,
        stats["attack"],
        attack_factor,
        stats["partial_attack"],
        partial_attack_factor,
    )

    return tuple(augmented)


def get_rare_class_stats(cases: Sequence[GraceCase]) -> dict[str, int]:
    """Count rare class occurrences for reporting."""
    stats: dict[str, int] = {
        "total_cases": len(cases),
        "MajorClaim_entities": 0,
        "Attack_relations": 0,
        "Partial-Attack_relations": 0,
        "cases_with_MajorClaim": 0,
        "cases_with_Attack": 0,
        "cases_with_Partial-Attack": 0,
    }

    for case in cases:
        entity_types = {e.type for e in case.entities}
        relation_types = {r.relation_type for r in case.relations}

        stats["MajorClaim_entities"] += sum(1 for e in case.entities if e.type == "MajorClaim")
        stats["Attack_relations"] += sum(1 for r in case.relations if r.relation_type == "Attack")
        stats["Partial-Attack_relations"] += sum(
            1 for r in case.relations if r.relation_type == "Partial-Attack"
        )

        if "MajorClaim" in entity_types:
            stats["cases_with_MajorClaim"] += 1
        if "Attack" in relation_types:
            stats["cases_with_Attack"] += 1
        if "Partial-Attack" in relation_types:
            stats["cases_with_Partial-Attack"] += 1

    return stats
