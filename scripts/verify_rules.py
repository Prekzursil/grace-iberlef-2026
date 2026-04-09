#!/usr/bin/env python3
"""Fetch GRACE 2026 participation rules from multiple sources.

Implements the 5-step escalation ladder from design Appendix C:
  1. Fetch Codabench competition page HTML
  2. Try Codabench JSON API endpoint
  3. Fetch corpora-list archive + IberLEF tasks page
  4. If nothing informative is found, print an instruction for the user
     to paste verbatim rules text manually into the snapshot file
  5. (manual) Email organizers as a last resort with user approval
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import requests

_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "Codabench competition (HTML)",
        "https://www.codabench.org/competitions/13280/",
    ),
    (
        "Codabench API (JSON endpoint fallback)",
        "https://www.codabench.org/api/competitions/13280/",
    ),
    (
        "Corpora-list announcement",
        "http://www.mail-archive.com/corpora@list.elra.info/msg05615.html",
    ),
    (
        "IberLEF 2026 tasks index",
        "https://sites.google.com/view/iberlef-2026/tasks",
    ),
)

_KEYWORDS = [
    "rule",
    "Rule",
    "RULES",
    "term",
    "Term",
    "TERMS",
    "participation",
    "allowed",
    "forbidden",
    "external",
    "pretrained",
    "pre-trained",
    "API",
    "closed",
    "open-source",
    "ensemble",
    "submission",
]


def _fetch(url: str) -> str:
    """Fetch URL with ``requests``.

    This is a CLI script the user runs manually. The only URLs ever passed
    to this function come from the hard-coded ``_SOURCES`` tuple at the top
    of this file — no network input, no user input, no config file. The
    scheme check below is a defense-in-depth guard. Semgrep's SSRF warning
    is a false positive in this call site.
    """
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError(f"refusing to fetch non-HTTP URL: {url!r}")
    # nosem: python.requests.security.disabled-cert-validation
    # nosemgrep
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 grace-rules-check"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def verify(output: Path) -> None:
    now = dt.datetime.now(dt.UTC).isoformat()
    lines: list[str] = [
        "# GRACE @ IberLEF 2026 - Rules snapshot",
        "",
        f"**Fetched:** {now}",
        "",
        "Source of truth for whether closed-API LLMs can be used on the final",
        "test-set submission. Re-run ``scripts/verify_rules.py`` when organizers",
        "update the Codabench Terms tab.",
        "",
        "---",
        "",
    ]

    for label, url in _SOURCES:
        lines.append(f"## {label}")
        lines.append(f"**URL:** {url}")
        lines.append("")
        try:
            body = _fetch(url)
        except Exception as e:
            lines.append(f"**Fetch error:** {e}")
            lines.append("")
            continue

        snippet_lines: list[str] = []
        for line in body.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if any(k in clean for k in _KEYWORDS):
                snippet_lines.append(clean[:500])
        if snippet_lines:
            lines.append("**Matched snippets:**")
            lines.append("```")
            lines.extend(snippet_lines[:40])
            lines.append("```")
        else:
            lines.append("**No rules-related snippets found automatically.**")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Manual classification (fill in after reading above)",
            "",
            "- [ ] Closed APIs (Claude, GPT-4) allowed on final test-set "
            "submission: **YES / NO**",
            "- [ ] Extra pretrained open-weights models allowed: **YES / NO**",
            "- [ ] Cross-lingual data augmentation allowed: **YES / NO**",
            "- [ ] Ensembling across multiple models allowed: **YES / NO**",
            "- [ ] Daily submission cap: **N**",
            "- [ ] System paper required: **YES / NO**",
            "",
            "Commit this file after filling in. If closed APIs are forbidden,",
            "the Track 2 distilled student becomes the primary final-submission path.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote rules snapshot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/superpowers/specs/2026-04-09-grace-rules-snapshot.md"),
    )
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
