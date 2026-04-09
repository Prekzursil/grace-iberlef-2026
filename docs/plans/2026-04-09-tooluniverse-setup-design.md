# Design: ToolUniverse Install & Configuration

**Date:** 2026-04-09
**Target project:** `C:\Users\Prekzursil\Downloads\bionlp\` (new workspace)
**Upstream:** https://github.com/mims-harvard/ToolUniverse
**Paper:** arXiv:2509.23426
**Docs:** https://zitniklab.hms.harvard.edu/ToolUniverse/

## 1. Goals

- Install ToolUniverse in **hybrid mode**: global MCP access from any Claude Code session on this machine, plus a dedicated workspace at `C:\Users\Prekzursil\Downloads\bionlp\` for Python SDK, CLI, notebooks, keys, and analysis artifacts.
- All 1000+ scientific tools reachable via **Compact Mode** (4-5 meta-tools, ~99% context savings).
- All 68 pre-built **Agent Skills** installed into Claude Code for direct invocation and auto-trigger.
- Zero API keys in any git-tracked repo. Keys live only in `C:\Users\Prekzursil\Downloads\bionlp\.env`.
- Reproducible: pinned venv for SDK/CLI work; `uvx` for MCP (auto-refresh latest).

## 2. User Choices (captured from brainstorming session)

| Question | Choice |
|---|---|
| 1. Interface | D — MCP + CLI + SDK (all three) |
| 2. Domains | J — all (drug discovery, genes/proteins, literature, clinical trials, oncology, rare disease, therapeutic reasoning, multi-omics, BioNLP, structural biology) |
| 3. API keys | B — Standard academic set: NCBI, Semantic Scholar, FDA OpenFDA, UMLS |
| 4. Install location | A+C hybrid — global MCP in `~/.claude.json`, workspace at `C:\Users\Prekzursil\Downloads\bionlp\` |
| 5. Tool exposure | A — Compact Mode only (4-5 meta-tools) |
| 6a. MCP transport | A — stdio via `uvx --refresh tooluniverse` |
| 6b. Agent Skills | I — install all 68 via `npx skills add mims-harvard/ToolUniverse` |

## 3. Architecture

```
Claude Code (any project, anywhere on this machine)
 ├── MCP client reads ~/.claude.json → spawns `uvx --refresh tooluniverse`
 │     sees: tool_search, tool_describe, tool_run, skill_list, skill_run  (Compact Mode)
 └── Skill registry: 68 ToolUniverse agent skills (installed globally via npx skills)
        │
        ▼
 uvx cached venv (auto-managed by uv)
    tooluniverse package + deps
    env vars loaded from C:\Users\Prekzursil\Downloads\bionlp\.env
        │
        ▼
 1000+ external tools: PubMed, Semantic Scholar, OpenTargets, ChEMBL, UniProt,
                       FAERS, ClinicalTrials.gov, AlphaFold, PDB, Europe PMC,
                       ArXiv, BioRxiv, ...

C:\Users\Prekzursil\Downloads\bionlp\
  .venv\           pinned venv for CLI + SDK parity
  .env             all API keys (gitignored)
  .env.example     template
  docs\plans\      this design doc + future plans
  notebooks\       Jupyter analysis notebooks
  scripts\         standalone Python scripts
  data\            downloads, results, local caches (gitignored)
  README.md        workspace usage guide
  .gitignore       .env, .venv, data/, __pycache__/
```

## 4. Components & Responsibilities

| Component | Responsibility | Location |
|---|---|---|
| Prereq check script | Verify python 3.10+, uv, node, git; fail fast with clear messages | one-shot bash |
| `uv` install | Install uv if missing (`winget install astral-sh.uv` or `pip install uv`) | one-time |
| Workspace scaffold | Create folder tree, README, `.gitignore`, `.env.example` | one-shot |
| Pinned venv | `uv venv` + `uv pip install tooluniverse` inside workspace | reproducibility |
| `.env` file | NCBI_API_KEY, SEMANTIC_SCHOLAR_API_KEY, FDA_OPENFDA_KEY, UMLS_API_KEY | gitignored |
| Global MCP config | Add `tooluniverse` server block to `~/.claude.json` with `uvx` command + env refs | user-level |
| Agent Skills install | `npx skills add mims-harvard/ToolUniverse` | skill registry |
| Validation suite | 8-step smoke test ladder (prereqs → CLI → key-gated tool → MCP ping → skill invocation) | `bionlp/scripts/validate.ps1` |
| Workspace README | Usage patterns, example prompts, common gotchas, re-validation commands | docs |
| Memory pointers | Auto memory entries so Claude recalls setup in future sessions | `~/.claude/projects/.../memory/` |
| Desktop cheatsheet | New `tooluniverse-quickstart.md` with common prompt patterns | `Desktop\claude-cheatsheets\` |

## 5. Data Flow — Typical Query

1. User asks Claude: *"find recent papers on CRISPR off-target effects in liver cells, then check OpenTargets for relevant genes"*
2. Claude sees 5 Compact-Mode meta-tools → calls `tool_search("CRISPR off-target literature")`
3. ToolUniverse returns matching tool IDs: `pubmed_search`, `semantic_scholar_search`, `europe_pmc_search`
4. Claude calls `tool_run("pubmed_search", {...})` → TU proxies to NCBI with `NCBI_API_KEY` from `.env`
5. Results return; Claude synthesizes, then calls `tool_search("opentargets gene")` → `tool_run("opentargets_target_search", {...})`
6. Final answer composed from multi-tool results.

**Agent Skill flow:** User invokes `/drug_repurposing liver fibrosis` → skill auto-fires → preset tool-chain runs PubMed + OpenTargets + ChEMBL + FAERS → structured report returned.

## 6. API Key Signup Plan

| Key | URL | Time | Cost | Destination |
|---|---|---|---|---|
| `NCBI_API_KEY` | ncbi.nlm.nih.gov/account → Settings → API Key Management | 2 min | free | `.env` |
| `SEMANTIC_SCHOLAR_API_KEY` | semanticscholar.org/product/api#api-key-form | 5 min | free | `.env` |
| `FDA_OPENFDA_KEY` | open.fda.gov/apis/authentication | 1 min | free | `.env` |
| `UMLS_API_KEY` | uts.nlm.nih.gov/uts/signup-login | 1-2 days NIH approval | free | `.env` (lazy — install proceeds without) |

`.env.example` documents the structure. Real `.env` never committed.

## 7. Error Handling

- **uv missing** → fail fast with winget/pip install command
- **python < 3.10** → abort with upgrade instruction
- **.env missing** → MCP still starts; key-gated tools return graceful "key missing" errors
- **Rate limits** → ToolUniverse has built-in backoff + two-tier cache (LRU + SQLite)
- **Offline** → cached results served; new queries fail cleanly
- **Skill install failure** (`npx skills`) → document manual clone + local install fallback
- **MCP start failure** → Claude surfaces MCP error; `bionlp/scripts/health-check.ps1` diagnoses

## 8. Validation Ladder

1. `uv --version`, `python --version`, `node --version`, `git --version` all green
2. `uvx tooluniverse --help` exits 0
3. `tu list` returns > 1000 entries
4. `tu run europe_pmc_search --query "CRISPR"` returns non-empty (keyless tool)
5. `tu run pubmed_search --query "CRISPR" --limit 3` returns results (confirms NCBI key)
6. Fresh Claude Code session → "list MCP tools" shows tooluniverse meta-tools
7. Invoke a lightweight ToolUniverse skill from Claude → non-error response
8. Commit workspace README with these commands for re-runnable validation

## 9. Persistence / Memory

- Project memory: `bionlp_workspace.md` → workspace path + hybrid layout
- Reference memory: `tooluniverse_usage.md` → Compact Mode, skill names, prompt patterns
- Update `MEMORY.md` index
- Desktop cheatsheet: `tooluniverse-quickstart.md` with common queries

## 10. Non-Goals

- Not training any model (that was the previous misread of "ToolBrain")
- Not bundling ToolUniverse into the SWFOC editor repo (separate concerns)
- Not running a persistent HTTP server (stdio transport is enough)
- Not wiring commercial model inference keys (OpenAI/Anthropic/Together) in this phase — can add later if specific TU tools demand them

## 11. Open Items (for writing-plans phase)

- Exact `~/.claude.json` JSON block for the MCP server
- Whether to `git init` the bionlp workspace (probably yes, private)
- Whether Python 3.12 (default in TU readme conda example) or latest (3.13) — likely 3.12 for maximum compatibility
- Exact skill-install command verification (npx skills CLI may require specific flags on Windows)
- Whether to add `uv-tool-install tooluniverse` as an alternative-path fallback to `uvx`
