#!/usr/bin/env python3
"""Download and convert AbstRCT-ES (HiTZ) to GRACE JSON format.

AbstRCT-ES is a Spanish corpus of ~1,000 RCT abstracts with argument annotations
(Premise, Claim, Support, Attack). Created by HiTZ (GRACE organizers) via
translate-and-project from English AbstRCT.

Note: AbstRCT-ES does NOT have MajorClaim (GRACE-specific), so all entities
map to Premise or Claim only.

Usage:
    python scripts/convert_abstrct_es.py [--output downloaded_data/external/abstrct_es.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_abstrct_es(output: Path, split: str = "train") -> None:
    """Download AbstRCT-ES from HuggingFace and convert to GRACE JSON."""
    from datasets import load_dataset

    print(f"Loading HiTZ/AbstRCT-ES ({split} split)...")
    ds = load_dataset("HiTZ/AbstRCT-ES", split=split)

    cases: list[dict] = []
    skipped = 0

    for idx, item in enumerate(ds):
        case_id = f"abstrct_es_{split}_{idx}"
        raw_text = item.get("text", "")
        if not raw_text:
            skipped += 1
            continue

        # Parse entities
        entities = []
        ent_id_map: dict[str, str] = {}  # original_id → our_id

        if item.get("entities"):
            for ent_idx, ent in enumerate(item["entities"]):
                ent_id = f"T{ent_idx + 1}"
                ent_type = ent.get("type", ent.get("label", "Premise"))

                # Map AbstRCT entity types to GRACE types
                if ent_type in ("Premise", "Claim"):
                    grace_type = ent_type
                elif ent_type.lower() in ("premise", "evidence"):
                    grace_type = "Premise"
                elif ent_type.lower() in ("claim", "conclusion"):
                    grace_type = "Claim"
                else:
                    grace_type = "Premise"  # default fallback

                start = ent.get("start", ent.get("offset_start", 0))
                end = ent.get("end", ent.get("offset_end", 0))
                text = ent.get("text", raw_text[start:end])

                # Validate offset
                if raw_text[start:end] != text:
                    # Try to find the text in raw_text
                    found_idx = raw_text.find(text)
                    if found_idx >= 0:
                        start = found_idx
                        end = found_idx + len(text)
                    else:
                        continue  # skip entity with bad offsets

                orig_id = ent.get("id", str(ent_idx))
                ent_id_map[orig_id] = ent_id

                entities.append(
                    {
                        "id": ent_id,
                        "text": text,
                        "start": start,
                        "end": end,
                        "type": grace_type,
                    }
                )

        # Parse relations
        relations = []
        if item.get("relations"):
            for rel_idx, rel in enumerate(item["relations"]):
                rel_id = f"R{rel_idx + 1}"
                rel_type = rel.get("type", rel.get("label", "Support"))

                # Map AbstRCT relation types to GRACE types
                if rel_type in ("Support", "Attack", "Partial-Attack"):
                    grace_rel_type = rel_type
                elif rel_type.lower() == "support":
                    grace_rel_type = "Support"
                elif rel_type.lower() == "attack":
                    grace_rel_type = "Attack"
                else:
                    continue  # skip unknown relation types

                arg1_orig = str(rel.get("arg1", rel.get("head", "")))
                arg2_orig = str(rel.get("arg2", rel.get("tail", "")))

                arg1_id = ent_id_map.get(arg1_orig)
                arg2_id = ent_id_map.get(arg2_orig)

                if arg1_id and arg2_id:
                    relations.append(
                        {
                            "id": rel_id,
                            "arg1_id": arg1_id,
                            "arg2_id": arg2_id,
                            "relation_type": grace_rel_type,
                        }
                    )

        cases.append(
            {
                "id": case_id,
                "raw_text": raw_text,
                "entities": entities,
                "relations": relations,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Converted {len(cases)} cases ({skipped} skipped) → {output}")
    print(f"  Entities: {sum(len(c['entities']) for c in cases)}")
    print(f"  Relations: {sum(len(c['relations']) for c in cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloaded_data/external/abstrct_es.json"),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF dataset split to convert (default: train)",
    )
    args = parser.parse_args()
    convert_abstrct_es(args.output, args.split)


if __name__ == "__main__":
    main()
