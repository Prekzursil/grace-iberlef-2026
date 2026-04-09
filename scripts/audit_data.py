#!/usr/bin/env python3
"""Data audit CLI - computes schema and distribution stats for GRACE 2026.

Writes a Markdown report to experiments/audits/<timestamp>.md plus a JSON
summary to stdout. Intended to be re-runnable any time the organizers push
a data update.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grace.io.loaders import load_track1, load_track2

if TYPE_CHECKING:
    from grace.io.schema import GraceCase


def _stats(cases: tuple[GraceCase, ...]) -> dict[str, Any]:
    ent_types: collections.Counter[str] = collections.Counter()
    rel_types: collections.Counter[str] = collections.Counter()
    text_lens: list[int] = []
    n_ents: list[int] = []
    n_rels: list[int] = []
    sent_counts: list[int] = []
    choice_counts: collections.Counter[int] = collections.Counter()
    sent_relev: collections.Counter[str] = collections.Counter()

    for case in cases:
        text_lens.append(len(case.raw_text))
        n_ents.append(len(case.entities))
        n_rels.append(len(case.relations))
        for e in case.entities:
            ent_types[e.type] += 1
        for r in case.relations:
            rel_types[r.relation_type] += 1
        sent_counts.append(len(case.context_sentences))
        choice_counts[len(case.choices)] += 1
        for lbl in case.sentence_relevancy:
            sent_relev[lbl] += 1

    return {
        "cases": len(cases),
        "text_len_avg": round(statistics.mean(text_lens), 1) if text_lens else 0,
        "text_len_median": statistics.median(text_lens) if text_lens else 0,
        "text_len_max": max(text_lens) if text_lens else 0,
        "ents_per_case_avg": round(statistics.mean(n_ents), 2) if n_ents else 0,
        "rels_per_case_avg": round(statistics.mean(n_rels), 2) if n_rels else 0,
        "entity_types": dict(ent_types),
        "relation_types": dict(rel_types),
        "sentences_per_case_avg": round(statistics.mean(sent_counts), 2) if sent_counts else 0,
        "choices_per_case_hist": dict(choice_counts),
        "sentence_relevancy_distribution": dict(sent_relev),
    }


def audit(data_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    loaders: list[tuple[str, Any]] = [
        ("track_1_train.json", load_track1),
        ("track_1_dev.json", load_track1),
        ("track_2_train.json", load_track2),
        ("track_2_dev.json", load_track2),
    ]
    for fname, loader in loaders:
        path = data_dir / fname
        if not path.exists():
            print(f"SKIP {fname} (not found at {path})")
            continue
        results[fname] = _stats(loader(path))

    out_dir = Path("experiments") / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).isoformat().replace(":", "-")
    out_md = out_dir / f"{ts}.md"

    lines: list[str] = [f"# Data audit - {ts}", ""]
    for fname, stats in results.items():
        lines.append(f"## {fname}")
        lines.append(f"- cases: **{stats['cases']}**")
        lines.append(
            f"- text length: avg {stats['text_len_avg']}, "
            f"median {stats['text_len_median']}, "
            f"max {stats['text_len_max']}"
        )
        lines.append(f"- entities/case: {stats['ents_per_case_avg']}")
        lines.append(f"- relations/case: {stats['rels_per_case_avg']}")
        lines.append(f"- entity types: {stats['entity_types']}")
        lines.append(f"- relation types: {stats['relation_types']}")
        lines.append(f"- sentences/case: {stats['sentences_per_case_avg']}")
        lines.append(f"- choices/case histogram: {stats['choices_per_case_hist']}")
        lines.append(f"- sentence relevancy: {stats['sentence_relevancy_distribution']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report to {out_md}")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Path to public_data/public_data/")
    args = parser.parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"data_dir does not exist: {args.data_dir}")
    audit(args.data_dir)


if __name__ == "__main__":
    main()
