# ToolUniverse Install & Configuration — Implementation Plan (v3, round-2 gate fixes)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) for this plan — the steps are idempotent shell commands, subagent dispatch per step is overkill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install ToolUniverse in hybrid mode (global MCP + dedicated `bionlp` workspace) with Compact Mode, 4 free API keys, and 66 Agent Skills, validated end-to-end.

**Architecture:** ToolUniverse runs as a stdio MCP server via `tooluniverse-smcp-stdio --compact-mode` (dedicated binary from the `tooluniverse` package), registered in `~/.claude.json`. A pinned venv at `C:\Users\Prekzursil\Downloads\bionlp\.venv\` provides CLI + SDK parity. API keys live in `bionlp\.env` AND are referenced literally in the MCP config env block (no shell expansion supported). Compact Mode exposes 5 core tools: `list_tools`, `grep_tools`, `get_tool_info`, `execute_tool`, `find_tools`.

**Tech Stack:** Python 3.12, uv/uvx, ToolUniverse (PyPI, pinned), Node.js, Claude Code MCP stdio.

**Spec:** See `2026-04-09-tooluniverse-setup-design.md`.

**Revision notes (v3):** v1 made unverified assumptions about CLI shape. v2 used probed ground truth + addressed 11/12 gate concerns. v3 fixes the remaining issues found in round-2 gate review: pre-install backup tagging (uninstall restores correct state), proper exit-code handling in MCP pre-check, bionlp-scoped memory dir (not SWFOC), hard STOP gate for manual test, HALT on probe failure, Python-based JSON edit, corrected uv cache arg, dotfile sync caveat, probe for Claude skill target path, richer README prompt examples.

---

## Pre-flight Reference

**Paths:**
- Workspace root: `C:\Users\Prekzursil\Downloads\bionlp\`
- Venv: `C:\Users\Prekzursil\Downloads\bionlp\.venv\`
- Env file: `C:\Users\Prekzursil\Downloads\bionlp\.env`
- MCP config: `C:\Users\Prekzursil\.claude.json`
- Memory dir: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-SWFOC-editor\memory\`

**Shell:** git-bash. Unix paths where possible (`/c/Users/...`), Windows paths only in file contents.

**Compact Mode tools (verbatim from ToolUniverse docs):** `list_tools`, `grep_tools`, `get_tool_info`, `execute_tool`, `find_tools`.

---

## Task 0: Probe Task — Verify Third-Party CLI Shapes BEFORE Writing Scripts

**Goal:** Discover the real command surfaces so every later script uses verified flags, not guessed ones.

- [ ] **Step 0.1: Create workspace skeleton**

```bash
mkdir -p "/c/Users/Prekzursil/Downloads/bionlp/scripts"
mkdir -p "/c/Users/Prekzursil/Downloads/bionlp/docs/plans"
mkdir -p "/c/Users/Prekzursil/Downloads/bionlp/notebooks"
mkdir -p "/c/Users/Prekzursil/Downloads/bionlp/data"
```

The design and plan docs already exist in `docs/plans/` from the brainstorming phase — verify:

```bash
ls "/c/Users/Prekzursil/Downloads/bionlp/docs/plans/"
```

Expected: at least `2026-04-09-tooluniverse-setup-design.md` and `2026-04-09-tooluniverse-setup-plan.md` present.

- [ ] **Step 0.2: Warm uvx cache + probe top-level commands**

```bash
uvx --refresh tooluniverse --help 2>&1 | head -60
```

Expected: prints help for the ToolUniverse entrypoint. If it fails with `executable not found`, try:

```bash
uvx --from tooluniverse tooluniverse-smcp-stdio --help 2>&1 | head -30
uvx --from tooluniverse tu --help 2>&1 | head -30
```

One of these should work. **Record which command succeeded** — later tasks use the working invocation.

- [ ] **Step 0.3: Probe `tu` CLI surface**

Temporarily install into an ephemeral venv via `uvx` to exercise the CLI without committing to the pinned venv yet:

```bash
uvx --from tooluniverse tu --help
uvx --from tooluniverse tu list --help
uvx --from tooluniverse tu run --help
uvx --from tooluniverse tu grep --help
uvx --from tooluniverse tu find --help
```

**Record the actual flag names** for `tu list` (look for `--mode`, `--json`, `--categories`) and `tu run` (look for the arg format — key=value vs JSON).

- [ ] **Step 0.4: Probe real tool IDs we'll validate against later**

```bash
uvx --from tooluniverse tu grep europe_pmc
uvx --from tooluniverse tu grep pubmed
uvx --from tooluniverse tu grep opentargets
```

**Record the real tool IDs** that appear (e.g. `EuropePMC_search_articles`, `PubMed_search`, `OpenTargets_get_target_by_symbol` — exact IDs unknown until probed). These get used in validate script in Task 5.

- [ ] **Step 0.5: Probe the skill install path — verify or fall back to manual**

The `npx skills add` command is contradicted across ToolUniverse docs. Verify reality:

```bash
npx skills --help 2>&1 || echo "npx skills unavailable"
```

If `npx skills --help` prints usage → we can use `npx skills add mims-harvard/ToolUniverse` in Task 7.
If it fails with "command not found" or prompts to install an unknown npm package → **do not proceed with npx**. Task 7 uses the manual clone+copy path documented at aiscientist.tools/setup.md.

**Record which path to use.**

- [ ] **Step 0.6: Probe `tooluniverse-smcp-stdio` (MCP transport binary)**

```bash
uvx --from tooluniverse tooluniverse-smcp-stdio --help 2>&1 | head -40
```

Expected: prints help showing `--compact-mode` flag (and other options). Confirm `--compact-mode` exists. If the binary name is different (e.g. `tooluniverse-smcp` without `-stdio`), record the real name.

- [ ] **Step 0.7: Probe Claude Code skill target dir (for Path B fallback)**

If Task 6 will need Path B (manual clone+copy), we need to know where Claude Code discovers user skills on Windows. Check:

```bash
ls -la "/c/Users/Prekzursil/.claude/skills/" 2>/dev/null && echo "user-level skills dir exists"
ls -la "/c/Users/Prekzursil/.claude/plugins/" 2>/dev/null | head
```

Expected: either `~/.claude/skills/` exists (user-level, preferred target for Path B) or you find a plugin-local skills dir. **Record the real target path.**

- [ ] **Step 0.8: Summarize probe findings + HALT on incomplete**

At the end of Task 0, write a structured note with **all of these fields populated** — any missing field means Task 0 is incomplete and downstream tasks MUST HALT until reprobed:

```markdown
# PROBE-RESULTS.md — source of truth for Tasks 1-9

- TU_ENTRYPOINT: <working uvx command from Step 0.2>
- TU_LIST_FLAGS: <flag names verified from Step 0.3, e.g. "--mode names --json">
- TU_RUN_SYNTAX: <"key=value" or "json" — verified from Step 0.3>
- EUROPE_PMC_TOOL_ID: <real ID from Step 0.4 grep>
- PUBMED_TOOL_ID: <real ID from Step 0.4 grep>
- OPENTARGETS_TOOL_ID: <real ID from Step 0.4 grep>
- SKILL_INSTALL_PATH: <"A: npx skills add" or "B: manual clone+copy">
- SKILL_TARGET_DIR: <path from Step 0.7 for Path B, or "n/a" for Path A>
- MCP_BINARY: <"tooluniverse-smcp-stdio" confirmed, or the actual name>
- COMPACT_MODE_FLAG: <"--compact-mode" confirmed, or actual flag>
- PROBE_DATE: <YYYY-MM-DD>
- STATUS: <"COMPLETE" or "INCOMPLETE — reprobe required">
```

Save to `C:\Users\Prekzursil\Downloads\bionlp\scripts\PROBE-RESULTS.md`. **This file is the source of truth for all later tasks — if any later step conflicts with probe results, probe results win.**

**HALT rule:** If `STATUS != COMPLETE`, HALT. Do NOT proceed to Task 1. Reprobe the missing items.

---

## Task 1: Prerequisite Check + Workspace Scaffold (merged)

Per Scope reviewer, collapse prereq + validate + health-check into one script. This task creates the workspace skeleton and the single unified script.

**Files:**
- Create: `C:\Users\Prekzursil\Downloads\bionlp\scripts\bionlp.sh`  (unified: prereqs + sdk check + health check)
- Create: `C:\Users\Prekzursil\Downloads\bionlp\.gitignore`
- Create: `C:\Users\Prekzursil\Downloads\bionlp\.env.example`
- Create: `C:\Users\Prekzursil\Downloads\bionlp\README.md`

- [ ] **Step 1.1: Write `.gitignore`**

File: `C:\Users\Prekzursil\Downloads\bionlp\.gitignore`

```gitignore
# Secrets
.env
.env.*
!.env.example

# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/

# Data (large downloads, local caches, analysis outputs)
data/
!data/.gitkeep

# ToolUniverse local caches
.tooluniverse_cache/

# OS
.DS_Store
Thumbs.db

# MCP backups
.claude.json.bak-*
```

- [ ] **Step 1.2: Write `.env.example`**

File: `C:\Users\Prekzursil\Downloads\bionlp\.env.example`

```bash
# ToolUniverse API keys. Copy to .env and fill in real values.
# Real .env is gitignored. NEVER commit real keys.
# Values MUST be bare tokens — no quotes, no spaces, no trailing whitespace.
# If your key contains special chars, contact support (shouldn't happen for these 4).

# NCBI E-utilities (PubMed, Gene, etc.) — free, instant
# https://www.ncbi.nlm.nih.gov/account/ → Account Settings → API Key Management
NCBI_API_KEY=

# Semantic Scholar — free, ~5 min email delivery
# https://www.semanticscholar.org/product/api#api-key-form
SEMANTIC_SCHOLAR_API_KEY=

# FDA OpenFDA — free, instant
# https://open.fda.gov/apis/authentication/
FDA_OPENFDA_KEY=

# UMLS (MeSH, ontology) — free, 1-2 day NIH approval
# https://uts.nlm.nih.gov/uts/signup-login
# Lazy: install can proceed without this. Add when approved.
UMLS_API_KEY=
```

- [ ] **Step 1.3: Write `README.md`**

File: `C:\Users\Prekzursil\Downloads\bionlp\README.md`

```markdown
# bionlp — ToolUniverse Workspace

Dedicated workspace for biomedical/scientific research using [ToolUniverse](https://github.com/mims-harvard/ToolUniverse) from Harvard Mims/Zitnik Lab.

## Layout

- `.venv/` — pinned Python 3.12 venv (gitignored)
- `.env` — API keys (gitignored, never committed)
- `.env.example` — template showing required keys + signup URLs
- `notebooks/` — Jupyter analysis notebooks
- `scripts/` — install/validate/health scripts + probe results
- `data/` — downloads, results, caches (gitignored)
- `docs/plans/` — design docs and implementation plans
- `requirements.lock` — frozen dependency list (reproducibility)

## Install Status

See `docs/plans/2026-04-09-tooluniverse-setup-plan.md` (v2) for the install plan. Probe findings that the plan depends on are in `scripts/PROBE-RESULTS.md`.

## Usage

### From Claude Code (MCP — primary)

ToolUniverse is registered globally in `~/.claude.json`. Every Claude Code session sees 5 Compact Mode meta-tools: `list_tools`, `grep_tools`, `get_tool_info`, `execute_tool`, `find_tools`. Claude discovers the right tool for your query via `find_tools` / `grep_tools`, then calls `execute_tool`.

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

*Therapeutic reasoning (multi-step — Claude will chain tools automatically):*
- "I'm researching drug repurposing for liver fibrosis. Find existing approved drugs with mechanisms relevant to hepatic stellate cell deactivation, cross-reference with known safety profiles and clinical trial history."
- "Given a patient with hepatomegaly, cataracts, and developmental delay — what rare diseases fit? Cross-reference OMIM, Orphanet, and gene associations."

### From the `tu` CLI

```bash
source .venv/Scripts/activate   # git-bash
# or: .venv\Scripts\Activate.ps1  (PowerShell)

tu --help
tu list --mode names              # list all tool names
tu grep pubmed                    # find pubmed-related tools
tu info <TOOL_ID> --detail brief  # inspect a tool
tu run <TOOL_ID> key=value        # run a tool
```

### From the Python SDK

```python
from tooluniverse import ToolUniverse
tu = ToolUniverse()
tools = tu.list_tools()
print(f"{len(tools)} tools available")
res = tu.run_tool("EuropePMC_search_articles", {"query": "CRISPR off-target"})
```

*(Exact method names may differ — see `tu --help` and the ToolUniverse docs.)*

## Re-validate / Health Check

```bash
bash scripts/bionlp.sh health
```

## Uninstall

```bash
bash scripts/bionlp.sh uninstall
```

Removes: venv, skills install, MCP server block in `~/.claude.json` (from backup), workspace memory entries. Leaves: the `bionlp/` workspace folder itself (delete manually if desired).

## API Keys

See `.env.example` for the 4 free keys. All free. UMLS takes 1-2 days for NIH approval; others are instant.

## Troubleshooting

- **MCP tools missing in Claude** → restart Claude Code; verify `~/.claude.json` has the `tooluniverse` block; run `bash scripts/bionlp.sh health`
- **"Key missing" errors** → check `.env` has the key with NO quotes and NO trailing whitespace
- **`tu list` returns 0 tools** → `uv pip install --python .venv/Scripts/python.exe --force-reinstall tooluniverse` inside venv
- **Rate limit hit** → TU has built-in backoff + cache; repeat hits cache
```

- [ ] **Step 1.4: Write the unified `bionlp.sh` script (skeleton — commands stubbed, filled in by later tasks)**

File: `C:\Users\Prekzursil\Downloads\bionlp\scripts\bionlp.sh`

```bash
#!/usr/bin/env bash
# bionlp.sh — unified prereq / install-validate / health-check / uninstall script.
# Usage:
#   bash bionlp.sh prereqs    # check prerequisites only
#   bash bionlp.sh health     # full health check (prereqs + venv + sdk + keys + uvx)
#   bash bionlp.sh uninstall  # clean uninstall
set -euo pipefail

WORKSPACE="/c/Users/Prekzursil/Downloads/bionlp"
VENV="$WORKSPACE/.venv"
TU="$VENV/Scripts/tu.exe"
CLAUDE_CONFIG="/c/Users/Prekzursil/.claude.json"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

check_prereqs() {
  echo "=== 1. Prerequisites ==="
  local fail=0
  local tools=(python uv node npx git)
  for t in "${tools[@]}"; do
    if command -v "$t" >/dev/null 2>&1; then
      green "OK: $t → $("$t" --version 2>&1 | head -1)"
    else
      red "MISSING: $t"; fail=1
    fi
  done

  # Python 3.10+ actual version check
  if command -v python >/dev/null 2>&1; then
    local pyver
    pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local pymajor="${pyver%%.*}"
    local pyminor="${pyver##*.}"
    if [[ "$pymajor" -lt 3 ]] || { [[ "$pymajor" -eq 3 ]] && [[ "$pyminor" -lt 10 ]]; }; then
      red "FAIL: python $pyver < 3.10"; fail=1
    else
      green "OK: python $pyver >= 3.10"
    fi
  fi

  if [[ "$fail" -ne 0 ]]; then
    red "Prereqs failed. Install missing tools:"
    echo "  uv:    winget install astral-sh.uv  OR  pip install uv"
    echo "  node:  winget install OpenJS.NodeJS.LTS"
    echo "  git:   winget install Git.Git"
    return 1
  fi
  return 0
}

check_venv_and_sdk() {
  echo ""
  echo "=== 2. Venv + ToolUniverse SDK ==="

  if [[ ! -d "$VENV" ]]; then
    red "FAIL: venv not found at $VENV"; return 1
  fi

  # Find tu binary — may be tu.exe, tu, or python -m tooluniverse
  local tu_cmd=""
  if [[ -x "$VENV/Scripts/tu.exe" ]]; then
    tu_cmd="$VENV/Scripts/tu.exe"
  elif [[ -x "$VENV/Scripts/tu" ]]; then
    tu_cmd="$VENV/Scripts/tu"
  elif "$VENV/Scripts/python.exe" -m tooluniverse --help >/dev/null 2>&1; then
    tu_cmd="$VENV/Scripts/python.exe -m tooluniverse"
  else
    red "FAIL: no usable tu entrypoint in venv"; return 1
  fi
  green "OK: tu entrypoint → $tu_cmd"

  # Tool catalog size — use --mode names --json for a parseable count
  local tool_count
  if tool_count=$($tu_cmd list --mode names --json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else sum(len(v) for v in d.values()))' 2>/dev/null); then
    if [[ "$tool_count" -lt 100 ]]; then
      yellow "WARN: only $tool_count tools visible (expected hundreds)"
    else
      green "OK: $tool_count tools in catalog"
    fi
  else
    yellow "WARN: could not parse tool count — tu list output format unknown"
    $tu_cmd list --mode names 2>/dev/null | head -3 || true
  fi
}

check_env_and_uvx() {
  echo ""
  echo "=== 3. .env + uvx MCP binary ==="

  if [[ ! -f "$WORKSPACE/.env" ]]; then
    yellow "WARN: $WORKSPACE/.env missing — keyed tools will return 'key missing'"
  else
    green "OK: .env present"
    # Validate no quotes in values
    if grep -E '^[A-Z_]+=".*"$|^[A-Z_]+='"'"'.*'"'"'$' "$WORKSPACE/.env" >/dev/null 2>&1; then
      yellow "WARN: .env contains quoted values — remove quotes (bare tokens only)"
    fi
  fi

  if uvx --from tooluniverse tooluniverse-smcp-stdio --help >/dev/null 2>&1; then
    green "OK: uvx tooluniverse-smcp-stdio --help responds"
  else
    red "FAIL: uvx tooluniverse-smcp-stdio not working"
    return 1
  fi
}

check_claude_config() {
  echo ""
  echo "=== 4. Claude MCP config ==="

  if [[ ! -f "$CLAUDE_CONFIG" ]]; then
    red "FAIL: $CLAUDE_CONFIG not found"; return 1
  fi

  if python -c "import json; json.load(open(r'$CLAUDE_CONFIG', encoding='utf-8'))" 2>/dev/null; then
    green "OK: $CLAUDE_CONFIG is valid JSON"
  else
    red "FAIL: $CLAUDE_CONFIG is invalid JSON"; return 1
  fi

  if python -c "import json; d=json.load(open(r'$CLAUDE_CONFIG', encoding='utf-8')); import sys; sys.exit(0 if 'tooluniverse' in str(d) else 1)" 2>/dev/null; then
    green "OK: tooluniverse MCP entry present"
  else
    yellow "WARN: tooluniverse not found in $CLAUDE_CONFIG (MCP not registered yet)"
  fi
}

do_uninstall() {
  echo "=== Uninstall ==="
  yellow "This will remove venv, restore ~/.claude.json from the PRE-INSTALL backup, and delete memory files."
  yellow "Press Ctrl+C to abort, Enter to continue."
  read -r _

  # 1. Remove venv
  if [[ -d "$VENV" ]]; then
    rm -rf "$VENV" && green "Removed venv"
  fi

  # 2. Restore ~/.claude.json from the dedicated pre-install backup (never overwritten)
  local pre_bak="$CLAUDE_CONFIG.bak-pre-tooluniverse"
  if [[ -f "$pre_bak" ]]; then
    cp "$pre_bak" "$CLAUDE_CONFIG" && green "Restored $CLAUDE_CONFIG from $pre_bak"
  else
    # Fallback: try to json-parse and delete the tooluniverse key
    yellow "No .bak-pre-tooluniverse found — attempting in-place JSON surgery to remove tooluniverse key"
    python - "$CLAUDE_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, encoding='utf-8') as f:
    d = json.load(f)
removed = False
def strip(obj):
    global removed
    if isinstance(obj, dict):
        if 'mcpServers' in obj and isinstance(obj['mcpServers'], dict):
            if 'tooluniverse' in obj['mcpServers']:
                del obj['mcpServers']['tooluniverse']
                removed = True
        for v in obj.values():
            strip(v)
    elif isinstance(obj, list):
        for v in obj:
            strip(v)
strip(d)
with open(p, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)
print("REMOVED" if removed else "NOT_FOUND")
PY
  fi

  # 3. Clear uv cache (prune entire cache — uv cache clean does not take pkg args in current uv)
  uv cache prune 2>/dev/null && green "Pruned uv cache" || yellow "uv cache prune unavailable — skipped"

  # 4. Delete memory entries — check both possible memory dirs (bionlp-scoped + SWFOC-scoped legacy)
  local bionlp_mem="/c/Users/Prekzursil/.claude/projects/C--Users-Prekzursil-Downloads-bionlp/memory"
  local swfoc_mem="/c/Users/Prekzursil/.claude/projects/C--Users-Prekzursil-Downloads-SWFOC-editor/memory"
  for dir in "$bionlp_mem" "$swfoc_mem"; do
    rm -f "$dir/bionlp_workspace.md" "$dir/tooluniverse_usage.md" 2>/dev/null && green "Removed memory in $dir" || true
  done

  yellow "Workspace folder $WORKSPACE left intact — delete manually if desired."
  green "Uninstall complete."
}

case "${1:-health}" in
  prereqs)   check_prereqs ;;
  health)    check_prereqs && check_venv_and_sdk && check_env_and_uvx && check_claude_config && green "All checks passed." ;;
  uninstall) do_uninstall ;;
  *) echo "Usage: bash bionlp.sh {prereqs|health|uninstall}"; exit 1 ;;
esac
```

- [ ] **Step 1.5: Add `.gitkeep` files**

```bash
touch "/c/Users/Prekzursil/Downloads/bionlp/notebooks/.gitkeep"
touch "/c/Users/Prekzursil/Downloads/bionlp/data/.gitkeep"
```

- [ ] **Step 1.6: Run prereq-only check**

```bash
chmod +x "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh"
bash "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh" prereqs
```

Expected: all tools print `OK:` lines, exit 0.

- [ ] **Step 1.7: Initialize git repo + first commit**

Verify `docs/plans/` has files before adding (gate finding):

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
ls docs/plans/   # must show the design and plan docs
git init
git add .gitignore .env.example README.md scripts/bionlp.sh scripts/PROBE-RESULTS.md notebooks/.gitkeep data/.gitkeep docs/
git status
git commit -m "chore: scaffold bionlp workspace + unified script + probe results"
```

Expected: commit succeeds, `.env` absent from diff.

---

## Task 2: Pinned Venv + ToolUniverse SDK Install + Version Freeze

**Files:**
- Creates: `C:\Users\Prekzursil\Downloads\bionlp\.venv\` (gitignored)
- Create: `C:\Users\Prekzursil\Downloads\bionlp\requirements.lock`

- [ ] **Step 2.1: Create venv (Python 3.12)**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
uv venv --python 3.12 .venv
```

Expected: `Creating virtual environment at: .venv` with Python 3.12.x.

- [ ] **Step 2.2: Install tooluniverse with full activation**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
source .venv/Scripts/activate
uv pip install tooluniverse
```

Expected: `Installed N packages` with tooluniverse + deps.

- [ ] **Step 2.3: Freeze versions for reproducibility**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
source .venv/Scripts/activate
uv pip freeze > requirements.lock
head -5 requirements.lock
```

Expected: `requirements.lock` contains pinned versions like `tooluniverse==X.Y.Z` and its deps.

- [ ] **Step 2.4: Run the health check — venv+sdk section should now pass**

```bash
bash "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh" health
```

Expected: sections 1 and 2 print `OK:` lines. Sections 3 and 4 may still print warnings (no `.env` yet, MCP not registered yet) — that's fine at this stage.

If the tool-count check warns about unknown output format: consult `PROBE-RESULTS.md` from Task 0 Step 0.7 and update `check_venv_and_sdk()` in `bionlp.sh` with the real flag discovered by the probe.

- [ ] **Step 2.5: Commit venv milestone**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
git add requirements.lock
git commit -m "chore: pin tooluniverse venv via requirements.lock"
```

---

## Task 3: API Keys — Signup + `.env` Creation

This task is user-interactive. If running via `executing-plans`, pause and surface these instructions to the user.

- [ ] **Step 3.1: Copy template**

```bash
cp "/c/Users/Prekzursil/Downloads/bionlp/.env.example" "/c/Users/Prekzursil/Downloads/bionlp/.env"
```

- [ ] **Step 3.2: Sign up for NCBI API key**

1. https://www.ncbi.nlm.nih.gov/account/ → sign in or register (free)
2. Account Settings → API Key Management → Create API Key
3. Copy key
4. Paste into `.env`: `NCBI_API_KEY=<your_key>` (bare token, no quotes, no spaces)

- [ ] **Step 3.3: Sign up for Semantic Scholar key**

1. https://www.semanticscholar.org/product/api#api-key-form
2. Fill form (name, email, "personal biomedical research")
3. Wait for email (usually <5 min)
4. Paste into `.env`: `SEMANTIC_SCHOLAR_API_KEY=<your_key>`

- [ ] **Step 3.4: Sign up for FDA OpenFDA key**

1. https://open.fda.gov/apis/authentication/
2. Enter email → instant delivery
3. Paste into `.env`: `FDA_OPENFDA_KEY=<your_key>`

- [ ] **Step 3.5: (Optional, slow) Sign up for UMLS**

1. https://uts.nlm.nih.gov/uts/signup-login → register NIH account
2. Wait 1-2 days for approval email
3. After approval: Profile → API Key → paste into `.env` → `UMLS_API_KEY=<your_key>`

**Install proceeds without this key.** Tools needing UMLS return graceful "key missing" errors.

- [ ] **Step 3.6: Verify `.env` is gitignored**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
git status --ignored | grep -F ".env" || echo ".env not seen — may already be properly ignored"
git status | grep -F ".env" && echo "PROBLEM: .env is tracked!" || echo "OK: .env not in git status"
```

If `.env` appears as tracked/untracked → STOP, fix `.gitignore` before any commit.

- [ ] **Step 3.7: Run health check again — env section should now pass**

```bash
bash "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh" health
```

Expected: section 3 now shows `OK: .env present`. MCP section 4 still warns (not yet registered).

---

## Task 4: SDK + Key Plumbing Smoke Test (uses verified tool IDs)

This task exercises ONE keyless tool + ONE keyed tool using tool IDs discovered in Task 0 probe.

**Prerequisite:** `scripts/PROBE-RESULTS.md` from Task 0 must list real tool IDs for europe_pmc and pubmed searches. If not probed, re-run Task 0 Step 0.4.

- [ ] **Step 4.1: Read probe results for tool IDs**

```bash
cat "/c/Users/Prekzursil/Downloads/bionlp/scripts/PROBE-RESULTS.md"
```

Extract the two tool IDs to use: `$EUROPE_PMC_TOOL` and `$PUBMED_TOOL`. If uncertain, use `tu grep`:

```bash
source /c/Users/Prekzursil/Downloads/bionlp/.venv/Scripts/activate
tu grep europe_pmc | head
tu grep pubmed | head
```

- [ ] **Step 4.2: Keyless smoke test (Europe PMC)**

Activate venv, then run the tool with the ACTUAL arg format discovered in Task 0 Step 0.3. Two candidate formats:

```bash
# Format A: key=value
tu run <EUROPE_PMC_TOOL_ID> query="CRISPR off-target" limit=2

# Format B: JSON
tu run <EUROPE_PMC_TOOL_ID> '{"query": "CRISPR off-target", "limit": 2}'
```

Use whichever matches probe results. Expected: non-empty result. If the tool expects different parameter names (e.g. `search_query` instead of `query`), use `tu info <TOOL_ID>` to see the schema first.

```bash
tu info <EUROPE_PMC_TOOL_ID> --detail brief
```

- [ ] **Step 4.3: Keyed smoke test (PubMed with NCBI key)**

First, load `.env` into current shell:

```bash
set -a
source /c/Users/Prekzursil/Downloads/bionlp/.env
set +a
```

Then run pubmed with correct syntax:

```bash
tu run <PUBMED_TOOL_ID> query="CRISPR off-target" limit=2
# or JSON form — whichever was confirmed
```

Expected: non-empty result. If "key missing" error → `.env` has quoted value or trailing whitespace; fix format.

- [ ] **Step 4.4: Commit success marker**

File: `C:\Users\Prekzursil\Downloads\bionlp\.install-state`

```
phase=sdk-validated
date=2026-04-09
tooluniverse_frozen_version=see requirements.lock
keyless_probe=<EUROPE_PMC_TOOL_ID> OK
keyed_probe=<PUBMED_TOOL_ID> OK
```

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
git add .install-state
git commit -m "test: validate SDK + NCBI key plumbing end-to-end via real tool IDs"
```

---

## Task 5: Global MCP Registration — Concrete Launcher Script + `~/.claude.json` Edit

Gate finding: `${VAR}` expansion in MCP env is not supported. Two options — we use **literal values** in `~/.claude.json` because that file is already user-scoped (not in any repo) and is the simplest correct path. For users who want keys out of that file, an optional launcher script path is documented but not the default.

**Files:**
- Modify: `C:\Users\Prekzursil\.claude.json`
- (Optional) Create: `C:\Users\Prekzursil\Downloads\bionlp\scripts\mcp-launcher.sh`

- [ ] **Step 5.1: Back up `~/.claude.json` with a pre-install tag**

We use a dedicated filename (not a timestamped generic one) so uninstall can find the correct pre-install state even if the user runs install multiple times:

```bash
# Only create pre-install backup if one doesn't already exist — don't overwrite!
if [[ ! -f "/c/Users/Prekzursil/.claude.json.bak-pre-tooluniverse" ]]; then
  cp "/c/Users/Prekzursil/.claude.json" "/c/Users/Prekzursil/.claude.json.bak-pre-tooluniverse"
  echo "Created pre-install backup"
else
  echo "Pre-install backup already exists — not overwriting (preserves first-install state)"
fi
# Always make a timestamped safety backup too
cp "/c/Users/Prekzursil/.claude.json" "/c/Users/Prekzursil/.claude.json.bak-$(date +%Y%m%d-%H%M%S)"
ls -1 /c/Users/Prekzursil/.claude.json.bak-* 2>/dev/null
```

Expected: `.bak-pre-tooluniverse` exists (one-shot, never overwritten) + a fresh timestamped safety backup.

- [ ] **Step 5.2: Read current `~/.claude.json`**

Use the Read tool (agent instruction — this is for Claude/subagent, not a shell command) on `C:\Users\Prekzursil\.claude.json`. Locate the `mcpServers` object. It may be at top level or nested under a project key.

- [ ] **Step 5.3: Load real key values into shell so we can embed them**

```bash
set -a
source /c/Users/Prekzursil/Downloads/bionlp/.env
set +a
echo "NCBI_API_KEY=${NCBI_API_KEY:-MISSING}"
echo "SEMANTIC_SCHOLAR_API_KEY=${SEMANTIC_SCHOLAR_API_KEY:-MISSING}"
echo "FDA_OPENFDA_KEY=${FDA_OPENFDA_KEY:-MISSING}"
echo "UMLS_API_KEY=${UMLS_API_KEY:-empty_ok}"
```

Expected: each line prints the first few chars of the real key (don't echo full keys if recording this session). `UMLS_API_KEY=empty_ok` is acceptable.

- [ ] **Step 5.4: Add `tooluniverse` MCP server block via Python JSON surgery (safer than text Edit)**

Do NOT use the Edit tool — `mcpServers` may be nested under a project key, making a text-based Edit target ambiguous. Use this Python script that loads, mutates, and writes back the JSON:

```bash
set -a; source /c/Users/Prekzursil/Downloads/bionlp/.env; set +a

python - <<'PY'
import json, os, sys
p = r"C:\Users\Prekzursil\.claude.json"
with open(p, encoding='utf-8') as f:
    d = json.load(f)

# Find or create the top-level mcpServers block. If nested, the user must edit manually.
if 'mcpServers' not in d:
    d['mcpServers'] = {}

d['mcpServers']['tooluniverse'] = {
    "command": "uvx",
    "args": ["--from", "tooluniverse", "tooluniverse-smcp-stdio", "--compact-mode"],
    "env": {
        "PYTHONIOENCODING": "utf-8",
        "NCBI_API_KEY": os.environ.get("NCBI_API_KEY", ""),
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        "FDA_OPENFDA_KEY": os.environ.get("FDA_OPENFDA_KEY", ""),
        "UMLS_API_KEY": os.environ.get("UMLS_API_KEY", ""),
    }
}

with open(p, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2)
print("OK: tooluniverse MCP block written")
PY
```

Expected: prints `OK: tooluniverse MCP block written`. If the script errors, check backup was created in Step 5.1 and restore.

**Notes:**
- `uvx --from tooluniverse tooluniverse-smcp-stdio` is the verified command per Task 0 probe
- `--compact-mode` is the flag (not an env var)
- Keys are LITERAL strings — no `${VAR}` — because Claude Code MCP does not expand shell vars
- This file is user-scoped (`~/.claude.json`), not in any repo, so literal keys are acceptable

**⚠️ DOTFILE SYNC CAVEAT:** If `~/.claude.json` is synced via Dropbox, OneDrive, iCloud, or a dotfiles repo, your API keys will be synced too. Check before proceeding:

```bash
# Quick sync detection
ls -la /c/Users/Prekzursil/.claude.json
# If it's a symlink pointing into a sync folder, or if the directory itself is synced (OneDrive),
# prefer the launcher script pattern in Step 5.5 (keys stay in bionlp\.env only).
```

If in doubt, use Step 5.5 instead of 5.4.

**Alternative for nested mcpServers:** If your `~/.claude.json` has `mcpServers` nested under a project key (rare but possible), the script above will create a NEW top-level `mcpServers`, which may not be what Claude Code uses. Read `~/.claude.json` first to understand the structure, and adapt the script's path to the correct location.

- [ ] **Step 5.5: (OPTIONAL) Launcher script pattern — only if user wants keys out of `~/.claude.json`**

Skip this step if you completed Step 5.4. Otherwise:

File: `C:\Users\Prekzursil\Downloads\bionlp\scripts\mcp-launcher.sh`

```bash
#!/usr/bin/env bash
# MCP launcher: loads .env then execs tooluniverse stdio.
# Used when you don't want API keys in ~/.claude.json.
set -euo pipefail
WORKSPACE="/c/Users/Prekzursil/Downloads/bionlp"
if [[ -f "$WORKSPACE/.env" ]]; then
  set -a
  source "$WORKSPACE/.env"
  set +a
fi
exec uvx --from tooluniverse tooluniverse-smcp-stdio --compact-mode
```

Then in `~/.claude.json`, use this block instead:

```json
"tooluniverse": {
  "command": "C:\\Program Files\\Git\\bin\\bash.exe",
  "args": ["-lc", "/c/Users/Prekzursil/Downloads/bionlp/scripts/mcp-launcher.sh"],
  "env": {"PYTHONIOENCODING": "utf-8"}
}
```

**Caveat:** full path to `bash.exe` is required because Claude Code's MCP spawner may not inherit PATH. Adjust the path if git-bash is installed elsewhere.

- [ ] **Step 5.6: Validate JSON is still valid**

```bash
python -c "import json; json.load(open(r'C:/Users/Prekzursil/.claude.json', encoding='utf-8')); print('JSON OK')"
```

Expected: `JSON OK`. If JSONDecodeError → restore from backup:

```bash
cp "$(ls -1t /c/Users/Prekzursil/.claude.json.bak-* | head -1)" /c/Users/Prekzursil/.claude.json
```

Then retry Step 5.4 more carefully.

- [ ] **Step 5.7: Pre-flight verify the MCP server command works standalone**

Before depending on Claude Code to spawn it, confirm the same command runs manually with proper exit-code handling:

```bash
set -a; source /c/Users/Prekzursil/Downloads/bionlp/.env; set +a

# Run with timeout, capture exit code, distinguish 124 (timeout = healthy boot) from real errors
set +e
timeout 5 uvx --from tooluniverse tooluniverse-smcp-stdio --compact-mode < /dev/null > /tmp/tu-mcp-probe.log 2>&1
rc=$?
set -e

echo "exit code: $rc"
echo "--- server output (first 20 lines) ---"
head -20 /tmp/tu-mcp-probe.log

if [[ "$rc" -eq 124 ]]; then
  echo "OK: server ran until timeout (124 = healthy boot, killed by timeout)"
elif [[ "$rc" -eq 0 ]]; then
  echo "OK: server exited cleanly"
else
  echo "FAIL: server exited with $rc — check /tmp/tu-mcp-probe.log above"
  exit 1
fi

# Also check for "key missing" / "not set" errors in the startup log
if grep -iE 'key.*(missing|not set|required)|error|traceback' /tmp/tu-mcp-probe.log; then
  echo "WARN: startup log contains error/missing-key messages — review above"
fi
```

Expected: exit code 124 (timeout killed a running server — healthy) OR exit code 0. Any other code fails the step. No tracebacks or "key missing" lines in the output.

If the server errors on startup: check `uvx --from tooluniverse tooluniverse-smcp-stdio --help` for required args.

- [ ] **Step 5.8: Run full health check**

```bash
bash "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh" health
```

Expected: all 4 sections print `OK:` lines, final `All checks passed.`

---

## Task 6: Install 66 Agent Skills (probe-driven path)

Per Task 0 Step 0.5, one of two paths was chosen.

- [ ] **Step 6.1: Path A — if `npx skills` verified working**

```bash
npx skills add mims-harvard/ToolUniverse
```

Expected: lists installed skills, exit 0.

- [ ] **Step 6.2: Path B — manual clone + copy (fallback, always works)**

If Path A unavailable:

```bash
# Clone to a cache location
mkdir -p /tmp/tu-skills
git clone --depth 1 https://github.com/mims-harvard/ToolUniverse.git /tmp/tu-skills

# Find the skills directory in the repo
ls /tmp/tu-skills/skills/ 2>/dev/null || find /tmp/tu-skills -maxdepth 2 -name "skills" -type d

# Copy into Claude Code's skill directory
# (verify the correct target — may be ~/.claude/skills/ or plugin-local)
mkdir -p /c/Users/Prekzursil/.claude/skills/tooluniverse
cp -r /tmp/tu-skills/skills/* /c/Users/Prekzursil/.claude/skills/tooluniverse/ 2>/dev/null || echo "no skills/ dir — check repo layout"
```

If the ToolUniverse repo has no `skills/` directory in the expected place: consult https://aiscientist.tools/setup.md for the canonical path and update this step.

- [ ] **Step 6.3: Defer skill verification to Task 7**

Skills become visible only in a fresh Claude Code session.

---

## Task 7: End-to-End Integration Test (manual — requires fresh Claude Code session)

**🛑 HARD STOP: This task CANNOT be automated.** It requires a human to open a new Claude Code session and type queries. If running via `executing-plans`, the executor MUST pause before Step 7.1 and wait for explicit user confirmation before proceeding to Step 7.8. **Do not auto-commit past this gate.**

- [ ] **Step 7.1: 🛑 STOP — surface pause to user, wait for explicit confirmation**

Agent/executor: print this banner and halt:

```
===============================================================
🛑 MANUAL CHECKPOINT — Tasks 0-6 complete. You must now:
  1. Open a NEW Claude Code terminal in any directory
  2. Run Steps 7.2-7.7 interactively (ask Claude the prompts)
  3. Come back and type "Task 7 passed" before Step 7.8 runs
===============================================================
```

DO NOT continue to Step 7.2 automatically. Wait for user confirmation.

- [ ] **Step 7.2: Verify MCP server loaded**

Ask: *"list MCP servers currently connected and their tools"*

Expected: `tooluniverse` listed with 5 Compact Mode tools: `list_tools`, `grep_tools`, `get_tool_info`, `execute_tool`, `find_tools`.

If missing: check Claude Code's MCP logs (usually in `~/.claude/logs/` or similar), verify `~/.claude.json` is still valid JSON, run `bash bionlp.sh health`.

- [ ] **Step 7.3: Verify skills registered (if Path A was used in Task 6)**

Ask: *"list available skills related to drug discovery or rare disease"*

Expected: ToolUniverse skills appear. If not, re-run Task 6.

- [ ] **Step 7.4: Keyless end-to-end test**

Ask: *"Using ToolUniverse, find 3 recent Europe PMC papers on CRISPR off-target effects. List titles + authors."*

Expected: Claude calls `find_tools` or `grep_tools` to discover the right tool, then `execute_tool` to run it, returns real paper metadata.

- [ ] **Step 7.5: Keyed end-to-end test**

Ask: *"Using ToolUniverse, search PubMed for 3 recent papers on GLP-1 receptor agonists in cardiovascular outcomes."*

Expected: Claude's tool call succeeds — confirming NCBI key plumbing end-to-end.

If "key missing" error → MCP env block in `~/.claude.json` isn't passing the key; check Step 5.4 used literal values.

- [ ] **Step 7.6: Skill invocation test (only if Task 6 Path A worked)**

Invoke a ToolUniverse skill by name from Claude, e.g. *"use the drug_repurposing skill with target: liver fibrosis"*.

Expected: skill runs, returns structured output.

- [ ] **Step 7.7: Memory read-back test (gate-added)**

In the fresh session, ask: *"where is my bionlp workspace?"*

Expected: Claude answers with `C:\Users\Prekzursil\Downloads\bionlp\` without being told — confirms memory files from Task 8 loaded.

*(If Task 8 hasn't run yet, skip this step and revisit after Task 8.)*

- [ ] **Step 7.8: Update `.install-state`**

```
phase=install-complete
date=2026-04-09
mcp_global=true
skills_installed=<path-a-or-path-b>
validation=end-to-end-passed
```

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
git add .install-state
git commit -m "chore: ToolUniverse install validated end-to-end from fresh Claude session"
```

---

## Task 8: Memory Persistence (gate-trimmed — bionlp-scoped primary + SWFOC pointer)

Per Scope reviewer round 2: primary memory lives in the **bionlp-scoped** project memory dir (auto-loaded when Claude is started in bionlp). A thin pointer entry in the SWFOC project memory dir so Claude working on other projects still knows bionlp exists and can refer to it.

**Files:**
- Create dir (if missing): `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-bionlp\memory\`
- Create: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-bionlp\memory\bionlp_workspace.md` (full details)
- Create: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-bionlp\memory\MEMORY.md` (index, new file)
- Modify: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-SWFOC-editor\memory\MEMORY.md` (append 1-line pointer only)
- Create (pointer): `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-SWFOC-editor\memory\bionlp_pointer.md` (thin pointer so SWFOC sessions know bionlp exists)

- [ ] **Step 8.0: Create bionlp-scoped memory dir + index**

```bash
mkdir -p "/c/Users/Prekzursil/.claude/projects/C--Users-Prekzursil-Downloads-bionlp/memory"
```

Then create the index file `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-bionlp\memory\MEMORY.md`:

```markdown
- [bionlp workspace + ToolUniverse](bionlp_workspace.md) — workspace layout, MCP details, Compact Mode 5 tools, common prompts
```

- [ ] **Step 8.1: Write `bionlp_workspace.md` (bionlp-scoped, primary)**

File: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-bionlp\memory\bionlp_workspace.md`

Content:

```markdown
---
name: bionlp workspace + ToolUniverse usage
description: Dedicated bio-research workspace at C:\Users\Prekzursil\Downloads\bionlp, ToolUniverse MCP global — how to access and common patterns
type: project
---

**Workspace:** `C:\Users\Prekzursil\Downloads\bionlp\`
- `.venv\` — pinned Python 3.12 venv with ToolUniverse (pinned in `requirements.lock`)
- `.env` — NCBI/Semantic Scholar/FDA/UMLS keys (gitignored)
- `scripts/bionlp.sh` — unified health/uninstall script (`bash bionlp.sh health`)
- `notebooks/`, `scripts/`, `data/` — analysis artifacts

**MCP integration:** ToolUniverse registered globally in `~/.claude.json` as `tooluniverse`, command `uvx --from tooluniverse tooluniverse-smcp-stdio --compact-mode`. Keys are literal in the env block (Claude MCP doesn't expand `${VAR}`).

**Compact Mode tools (the only 5 visible to Claude):**
- `list_tools` — enumerate tools
- `grep_tools` — keyword search over tools
- `find_tools` — semantic search (default on)
- `get_tool_info` — tool schema + description
- `execute_tool` — actually run a tool

**How to use:** Ask Claude for bio/medical/drug/gene/paper research. Claude will call `find_tools` → `execute_tool` automatically. No need to know tool IDs in advance.

**Why:** User wants biomedical research from any Claude Code session without polluting the SWFOC editor repo (separate concerns).

**How to apply:**
- Any bio/medical query → use ToolUniverse tools via MCP (automatic)
- Workspace-specific work (notebooks, Python scripts) → route to `bionlp/` folder
- Never suggest moving API keys elsewhere — they live in `bionlp\.env` and literally in `~/.claude.json`
- Re-validate install: `bash /c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh health`
- Uninstall cleanly: `bash /c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh uninstall`
```

- [ ] **Step 8.2: Write thin pointer in SWFOC memory (so other-project sessions know bionlp exists)**

File: `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-SWFOC-editor\memory\bionlp_pointer.md`

Content:

```markdown
---
name: bionlp workspace exists
description: Pointer — a separate bio-research workspace exists at C:\Users\Prekzursil\Downloads\bionlp with ToolUniverse MCP global
type: reference
---

User has a dedicated bio-research workspace at `C:\Users\Prekzursil\Downloads\bionlp\`. ToolUniverse is globally registered in `~/.claude.json` — the 5 Compact Mode tools (`list_tools`, `grep_tools`, `find_tools`, `get_tool_info`, `execute_tool`) are available in every Claude Code session, including from this SWFOC editor project.

Full details (layout, scripts, common prompts) are in the bionlp project memory dir — if user asks for more, switch to that project's context or read the workspace README at `C:\Users\Prekzursil\Downloads\bionlp\README.md`.

**Do not:** bundle bio tools into the SWFOC editor repo, or suggest moving the workspace.
**Do:** use MCP bio tools from any session when user asks bio/medical/drug/gene/paper questions.
```

- [ ] **Step 8.3: Append pointer line to SWFOC `MEMORY.md` index**

Add this line to the existing `C:\Users\Prekzursil\.claude\projects\C--Users-Prekzursil-Downloads-SWFOC-editor\memory\MEMORY.md`:

```markdown
- [bionlp pointer](bionlp_pointer.md) — bio-research workspace exists at `C:\Users\Prekzursil\Downloads\bionlp\`, ToolUniverse MCP global (5 Compact Mode tools)
```

---

## Task 9: Final Health Check

- [ ] **Step 9.1: Run full health check one last time**

```bash
bash "/c/Users/Prekzursil/Downloads/bionlp/scripts/bionlp.sh" health
```

Expected: all 4 sections green. Final line `All checks passed.`

- [ ] **Step 9.2: Final commit**

```bash
cd "/c/Users/Prekzursil/Downloads/bionlp"
git log --oneline
```

Expected: clean commit history showing the install journey.

---

## Self-Review v2

1. **Gate blockers fixed:**
   - ✅ `tu list` output format: Task 0 Step 0.3 probes first; `check_venv_and_sdk` in bionlp.sh uses `--mode names --json`
   - ✅ `tu run` flag shape: Task 0 Step 0.3/0.4 probes; Task 4 uses verified syntax from probe results
   - ✅ `npx skills` supply-chain: Task 0 Step 0.5 probes; Task 6 has concrete Path A/Path B fallback
   - ✅ MCP `${VAR}` expansion: Task 5 Step 5.4 uses LITERAL values; launcher script pattern (Step 5.5) is a concrete optional alternative

2. **Other gate concerns addressed:**
   - ✅ Scripts collapsed: 1 unified `bionlp.sh` instead of 3
   - ✅ Doc surfaces reduced: 1 README + 1 memory file (dropped Desktop cheatsheet, dropped 2nd memory file)
   - ✅ Version pin: Task 2 Step 2.3 writes `requirements.lock`
   - ✅ Uninstall path: `bionlp.sh uninstall` subcommand
   - ✅ `docs/` precheck: Task 1 Step 1.7 lists before `git add`
   - ✅ MCP env pre-check: Task 5 Step 5.7 runs the server standalone before depending on Claude
   - ✅ Python 3.10+ actual version check: `check_prereqs` uses real major/minor comparison
   - ✅ Memory read-back test: Task 7 Step 7.7
   - ✅ Execution method: inline `executing-plans` (per Scope reviewer)

3. **Remaining acknowledged risks:**
   - `tu` binary path inside venv (`tu.exe` vs `tu` vs `python -m tooluniverse`) — `bionlp.sh` handles all 3 via detection
   - Tool IDs in Task 4 depend on Task 0 probe — if probe produces unexpected results, Task 4 must be adjusted
   - Manual skill install path (Task 6 Path B) depends on ToolUniverse repo layout having a `skills/` directory — if not, Task 6 Step 6.2 documents fallback to docs lookup

4. **Placeholder scan:** No TBD, no TODO, no "similar to Task N", no "fill in details". Task 0 explicitly records findings for later tasks. Task 4 and Task 6 have `<TOOL_ID>` and `<path>` placeholders that ARE meant to be substituted from probe results — these are not implementation gaps, they're correct parameterization.

---

## Execution Handoff

Plan v2 complete and saved. Gate issues addressed.

**Execution method: inline via `superpowers:executing-plans`** (recommended by Scope reviewer — idempotent shell commands, subagent dispatch is overkill).

Next action: re-run plan-review-gate on v2 to confirm fixes land, then proceed to execution.
