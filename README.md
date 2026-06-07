# bionlp — ToolUniverse Workspace

Dedicated workspace for biomedical/scientific research using [ToolUniverse](https://github.com/mims-harvard/ToolUniverse) from Harvard Mims/Zitnik Lab.

## Projects in this workspace

This repo also hosts **GRACE @ IberLEF 2026** (Spanish clinical argument mining, Track 1: evidence component detection). Two distinct code areas — kept deliberately separate (no shared imports):

- **`grace/` + `tests/` + `configs/`** — the GRACE *reference library*: token-level BIO encoder, coverage-gated (`.coverage-thresholds.json`). Frozen reference implementation.
- **`FINAL/`** — the **canonical competition submission**: a sentence-level 4-class pipeline run as a 27-cell [Modal](https://modal.com) sweep (9 models × 3 seeds), softmax-probability ensemble (all-9 dev strict macro-F1 = 0.743), the blind-test submission (`FINAL/report_out/GRACE_track1_submission.json`), and the full report (`FINAL/report/` → PPTX / DOCX / PDF + figures). Start at [`FINAL/report/README.md`](FINAL/report/README.md).

The live results come from `FINAL/`; `grace/` is the earlier, separately-tested reference implementation.

## Layout

- `.venv/` — pinned Python 3.12 venv (gitignored)
- `.env` — API keys (gitignored, never committed)
- `.env.example` — template showing required keys + signup URLs
- `notebooks/` — Jupyter analysis notebooks
- `scripts/` — install/health/uninstall scripts + probe results
- `data/` — downloads, results, caches (gitignored)
- `docs/plans/` — design docs and implementation plans
- `requirements.lock` — frozen dependency list (reproducibility)

## Facts

- **2216 tools available** (2x what the ToolUniverse README advertises)
- **Compact Mode ON by default** — Claude sees 4-5 meta-tools (`list_tools`, `grep_tools`, `find_tools`, `get_tool_info`, `execute_tool`), not the full 2216
- **MCP entrypoint**: `uvx tooluniverse` (compact mode is default for this binary)
- **Windows users:** every `tu` CLI call needs `PYTHONIOENCODING=utf-8` — the `bionlp.sh` script handles this for you

## Usage

### From Claude Code (MCP — primary)

ToolUniverse is registered globally in `~/.claude.json`. Every Claude Code session sees the Compact Mode meta-tools. Claude discovers the right tool for your query via `find_tools` / `grep_tools`, then calls `execute_tool`.

**Copy-pasteable example prompts:**

*Literature:*
- "Find 5 recent papers on CRISPR off-target effects in liver cells and summarize the consensus."
- "Survey 2024-2026 BioRxiv preprints on senescence biomarkers — give me titles, authors, and 1-sentence summaries."
- "What does PubMed say about GLP-1 receptor agonist cardiovascular outcomes since 2023?"

*Drug discovery & pharmacology:*
- "For target EGFR, list approved inhibitors via ChEMBL, their mechanisms, and known resistance mutations."
- "Check FAERS for adverse events associated with osimertinib in the last 2 years."
- "What ADMET red flags does ChEMBL show for this SMILES: <paste SMILES>?"

*Genes, proteins, omics:*
- "Look up the OpenTargets association score between TP53 and pancreatic cancer, top 5 evidence sources."
- "Get the UniProt entry for BRCA1 — domain structure, known pathogenic variants, and interaction partners."
- "What's the STRING interaction network for MYC, first neighbors only?"

*Clinical / regulatory:*
- "Find active Phase 2/3 trials for idiopathic pulmonary fibrosis enrolling in the US right now."
- "Show the FDA approval history and label changes for pembrolizumab."

*Therapeutic reasoning (multi-step — Claude chains tools automatically):*
- "I'm researching drug repurposing for liver fibrosis. Find existing approved drugs with mechanisms relevant to hepatic stellate cell deactivation, cross-reference with known safety profiles and clinical trial history."
- "Given a patient with hepatomegaly, cataracts, and developmental delay — what rare diseases fit? Cross-reference OMIM, Orphanet, and gene associations."

### From the `tu` CLI

```bash
# Activate the venv (git-bash)
source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8   # Windows: avoid Unicode crash

tu --help
tu list --limit 0                                    # count all 2216 tools
tu grep pubmed --limit 5                             # find pubmed tools
tu info PubMed_search_articles                       # inspect a tool
tu run EuropePMC_search_articles query="CRISPR" limit=3   # run keyless
tu run PubMed_search_articles query="CRISPR" limit=3      # run keyed (needs NCBI_API_KEY in env)
```

### From the Python SDK

```python
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from tooluniverse import ToolUniverse
tu = ToolUniverse()
res = tu.run_tool("EuropePMC_search_articles", {"query": "CRISPR off-target", "limit": 5})
print(res)
```

## Re-validate / Health Check

```bash
bash scripts/bionlp.sh health
```

## Uninstall

```bash
bash scripts/bionlp.sh uninstall
```

Removes: venv, restores `~/.claude.json` from the pre-install backup (`.bak-pre-tooluniverse`), prunes uv cache, deletes memory files. Leaves the `bionlp/` workspace folder itself (delete manually if desired).

## API Keys

See `.env.example` for the 4 free keys. All free. UMLS takes 1-2 days for NIH approval; the other 3 are instant/fast.

## Troubleshooting

- **MCP tools missing in Claude** → restart Claude Code; verify `~/.claude.json` has the `tooluniverse` block; run `bash scripts/bionlp.sh health`
- **"Key missing" errors** → check `.env` has the key with NO quotes and NO trailing whitespace
- **`UnicodeEncodeError` on Windows CLI** → set `PYTHONIOENCODING=utf-8` before running `tu` (bionlp.sh does this for you)
- **`tu list` returns 0 tools** → `uv pip install --python .venv/Scripts/python.exe --force-reinstall tooluniverse` inside venv
- **Rate limit hit** → TU has built-in backoff + cache; repeat hits cache
