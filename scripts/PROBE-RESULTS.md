# PROBE-RESULTS.md — Source of Truth for Tasks 1-9

- **TU_ENTRYPOINT:** `uvx tooluniverse` (MCP stdio server, compact mode ON BY DEFAULT in this entrypoint — no flag needed). Advanced binary is `uvx --from tooluniverse tooluniverse-smcp-stdio` which has explicit `--compact-mode` flag.
- **TU_LIST_FLAGS:** `tu list --limit 0` = count-probe (returns total without results). `tu list --mode names --json` = parseable JSON of all names.
- **TU_RUN_SYNTAX:** Both forms work. Plan uses key=value: `tu run TOOL_ID query=value limit=5`. JSON also valid: `tu run TOOL_ID '{"query":"value","limit":5}'`.
- **EUROPE_PMC_TOOL_ID:** `EuropePMC_search_articles` — args: `query` (string, required), `limit` (int). Category: `EuropePMC`. Keyless.
- **PUBMED_TOOL_ID:** `PubMed_search_articles` — args: `query` (string, required), `limit` (int), optional: `mindate`, `maxdate`, `include_abstract`, `sort`. Category: `pubmed`. Uses `NCBI_API_KEY` env var for 10 req/sec rate (vs 3 keyless).
- **OPENTARGETS_TOOL_ID:** `OpenTargets_get_associated_targets_by_disease_efoId` (64 OpenTargets tools total — use `tu grep opentarget` to see more).
- **TOTAL_TOOLS:** **2216** (double the README's "1000+" claim)
- **SKILL_INSTALL_PATH:** **A — `npx skills add mims-harvard/ToolUniverse`**. Verified: `skills` is a real npm package (v1.4.9) with `add`, `remove`, `list`, `find`, `check`, `update` subcommands. NOT a supply-chain risk — legit package.
- **SKILL_TARGET_DIR:** `~/.claude/skills/` (confirmed exists, already has ~200 user-level skills). Path B fallback target if Path A fails.
- **MCP_BINARY:** `tooluniverse` (simple entrypoint, compact mode default) OR `tooluniverse-smcp-stdio --compact-mode` (advanced, explicit). Use the simple one for MCP config.
- **COMPACT_MODE_FLAG:** **Not needed** with `tooluniverse` entrypoint (default). Only required with `tooluniverse-smcp-stdio` entrypoint.
- **COMPACT_MODE_TOOLS:** The 4 core tools exposed (per `--compact-mode` help): `list_tools`, `grep_tools`, `get_tool_info`, `execute_tool`. `find_tools` optional 5th (per docs, default-on unless `--no-search`).
- **CRITICAL WINDOWS FIX:** `tu` CLI crashes with `UnicodeEncodeError` on Windows cp1252 codec when output contains arrows/special chars. **MUST set `PYTHONIOENCODING=utf-8`** before any `tu` invocation on Windows. Plan MCP config already includes this — good. Must also add to any CLI smoke test script and user README.
- **PROBE_DATE:** 2026-04-09
- **STATUS:** COMPLETE

## Plan adjustments needed based on probes

1. **MCP config (Task 5.4) — simpler than planned:**
   - Instead of: `uvx --from tooluniverse tooluniverse-smcp-stdio --compact-mode`
   - Use: `uvx tooluniverse` (compact mode is default)
   - Args list in JSON: `["tooluniverse"]` — 2 args instead of 4

2. **CLI env var (Task 1 bionlp.sh, Task 4 smoke test):**
   - Every `tu` invocation on Windows needs `PYTHONIOENCODING=utf-8` prefix
   - Add to script wrapper: `export PYTHONIOENCODING=utf-8` once at top

3. **Tool count threshold (bionlp.sh `check_venv_and_sdk`):**
   - Real count is 2216. `< 100` warn threshold is safe.
   - Can use `tu list --limit 0` for parseable count: `tu list --limit 0 2>&1 | grep -oE '[0-9]+ of [0-9]+' | head -1`

4. **Task 4 smoke test commands (now concrete):**
   - Keyless: `PYTHONIOENCODING=utf-8 tu run EuropePMC_search_articles query="CRISPR off-target" limit=2`
   - Keyed: `PYTHONIOENCODING=utf-8 tu run PubMed_search_articles query="CRISPR off-target" limit=2`

5. **Skill install (Task 6) — Path A confirmed:**
   - `npx skills add mims-harvard/ToolUniverse`
   - `skills list` to verify after install
