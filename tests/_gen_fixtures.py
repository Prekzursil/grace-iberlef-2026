"""One-off generator for committed synthetic test fixtures.

Run with: python tests/_gen_fixtures.py

Builds schema-correct Track 1 / Track 2 sample JSON files so the test suite
can exercise the loaders, scorer wrapper, augmentation, and submission code
WITHOUT the organizer-provided (non-redistributable) ``downloaded_data/``.
Entity offsets are computed programmatically so they always match raw_text.
"""

from __future__ import annotations

import json
from pathlib import Path


def ent(eid: str, text: str, raw: str, etype: str, occ: int = 0) -> dict:
    idx = -1
    for _ in range(occ + 1):
        idx = raw.index(text, idx + 1)
    return {"id": eid, "text": text, "start": idx, "end": idx + len(text), "type": etype}


def sent(ctx: str, s: str) -> dict:
    i = ctx.index(s)
    return {"sentence": s, "start": i, "end": i + len(s)}


def build_track1() -> list[dict]:
    raw1 = (
        "El cancer de mama es una enfermedad grave. "
        "El cribado reduce la mortalidad. "
        "Los efectos secundarios son leves."
    )
    c1 = {
        "id": "t1c1",
        "raw_text": raw1,
        "metadata": {},
        "annotations": {
            "entities": [
                ent("T1", "El cancer de mama es una enfermedad grave", raw1, "Premise"),
                ent("T2", "El cribado reduce la mortalidad", raw1, "MajorClaim"),
                ent("T3", "Los efectos secundarios son leves", raw1, "Claim"),
            ],
            "relations": [
                {"id": "R1", "arg1_id": "T1", "arg2_id": "T2", "relation_type": "Support"},
                {"id": "R2", "arg1_id": "T3", "arg2_id": "T2", "relation_type": "Attack"},
                # dangling relation -> exercises _keep_valid_relations drop path
                {"id": "R3", "arg1_id": "T9", "arg2_id": "T2", "relation_type": "Partial-Attack"},
            ],
        },
    }
    raw2 = "La quimioterapia es eficaz. Sin embargo, algunos estudios la cuestionan."
    c2 = {
        "id": "t1c2",
        "raw_text": raw2,
        "metadata": {},
        "annotations": {
            "entities": [
                ent("T1", "La quimioterapia es eficaz", raw2, "Claim"),
                ent("T2", "algunos estudios la cuestionan", raw2, "Premise"),
            ],
            "relations": [
                {"id": "R1", "arg1_id": "T2", "arg2_id": "T1", "relation_type": "Partial-Attack"},
            ],
        },
    }
    return [c1, c2]


def build_track2() -> list[dict]:
    ctx = (
        "El paciente presenta fiebre alta. "
        "La radiografia muestra un infiltrado. "
        "No hay antecedentes relevantes."
    )
    s_list = [
        "El paciente presenta fiebre alta.",
        "La radiografia muestra un infiltrado.",
        "No hay antecedentes relevantes.",
    ]
    t2c1 = {
        "id": "t2c1",
        "raw_text": ctx,
        "metadata": {
            "context": ctx,
            "context_sentences": [sent(ctx, s) for s in s_list],
            "choices": [
                {"id": "A", "text": "Neumonia", "start": 0, "end": 0},
                {"id": "B", "text": "Gripe", "start": 0, "end": 0},
            ],
            "correct_choice_id": "A",
        },
        "annotations": {
            "sentence_relevancy": ["relevant", "relevant", "not-relevant"],
            "entities": [ent("T1", "fiebre alta", ctx, "Premise")],
            "relations": [],
        },
    }
    raw_t2b = "Otro caso clinico breve."
    t2c2 = {
        "id": "t2c2",
        "raw_text": raw_t2b,
        "metadata": {
            "context": None,
            "context_sentences": [],
            "choices": [],
            "correct_choice_id": None,
        },
        "annotations": {"sentence_relevancy": [], "entities": [], "relations": []},
    }
    return [t2c1, t2c2]


def main() -> None:
    fx = Path(__file__).parent / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    (fx / "track1_sample.json").write_text(
        json.dumps(build_track1(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (fx / "track2_sample.json").write_text(
        json.dumps(build_track2(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote fixtures to", fx)


if __name__ == "__main__":
    main()
