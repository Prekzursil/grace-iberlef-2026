#!/usr/bin/env bash
# bionlp.sh — unified prereq / health-check / uninstall script for ToolUniverse workspace.
# Usage:
#   bash bionlp.sh prereqs    # check prerequisites only
#   bash bionlp.sh health     # full health check (prereqs + venv + sdk + env + uvx + claude config)
#   bash bionlp.sh uninstall  # clean uninstall
set -euo pipefail

# Windows: ensure UTF-8 for any tu CLI calls (avoids cp1252 UnicodeEncodeError)
export PYTHONIOENCODING=utf-8

WORKSPACE="/c/Users/Prekzursil/Downloads/bionlp"
VENV="$WORKSPACE/.venv"
CLAUDE_CONFIG="/c/Users/Prekzursil/.claude.json"
# Windows-native path for Python calls (Python.exe doesn't understand git-bash /c/... paths)
CLAUDE_CONFIG_WIN='C:\Users\Prekzursil\.claude.json'

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

find_tu_cmd() {
  # Detect usable tu entrypoint inside the venv. Returns the command as a string.
  if [[ -x "$VENV/Scripts/tu.exe" ]]; then
    echo "$VENV/Scripts/tu.exe"
  elif [[ -x "$VENV/Scripts/tu" ]]; then
    echo "$VENV/Scripts/tu"
  elif "$VENV/Scripts/python.exe" -m tooluniverse.cli --help >/dev/null 2>&1; then
    echo "$VENV/Scripts/python.exe -m tooluniverse.cli"
  else
    echo ""
  fi
}

check_prereqs() {
  echo "=== 1. Prerequisites ==="
  local fail=0
  local tools=(python uv uvx node npx git)
  for t in "${tools[@]}"; do
    if command -v "$t" >/dev/null 2>&1; then
      green "OK: $t → $("$t" --version 2>&1 | head -1)"
    else
      red "MISSING: $t"; fail=1
    fi
  done

  # Python 3.10+ real version check
  if command -v python >/dev/null 2>&1; then
    local pyver pymajor pyminor
    pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    pymajor="${pyver%%.*}"
    pyminor="${pyver##*.}"
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
    yellow "WARN: venv not found at $VENV (run Task 2 of the install plan)"
    return 0
  fi

  local tu_cmd
  tu_cmd="$(find_tu_cmd)"
  if [[ -z "$tu_cmd" ]]; then
    red "FAIL: no usable tu entrypoint in venv"; return 1
  fi
  green "OK: tu entrypoint → $tu_cmd"

  # Tool catalog size via count-probe "--limit 0". Parse "N of M tools" pattern.
  local count_line tool_count
  count_line=$($tu_cmd list --limit 0 2>&1 | grep -oE '[0-9]+ of [0-9]+ tools' | head -1 || true)
  if [[ -n "$count_line" ]]; then
    tool_count=$(echo "$count_line" | awk '{print $3}')
    if [[ "$tool_count" -lt 100 ]]; then
      yellow "WARN: only $tool_count tools visible (expected ~2000+)"
    else
      green "OK: $tool_count tools in catalog"
    fi
  else
    yellow "WARN: could not parse tool count from 'tu list --limit 0' output"
    $tu_cmd list --limit 0 2>&1 | head -3 || true
  fi
}

check_env_and_uvx() {
  echo ""
  echo "=== 3. .env + uvx MCP binary ==="

  if [[ ! -f "$WORKSPACE/.env" ]]; then
    yellow "WARN: $WORKSPACE/.env missing — keyed tools will return 'key missing' errors"
  else
    green "OK: .env present"
    # Detect quoted values (common mistake)
    if grep -E '^[A-Z_]+=".*"$|^[A-Z_]+='"'"'.*'"'"'$' "$WORKSPACE/.env" >/dev/null 2>&1; then
      yellow "WARN: .env contains quoted values — remove quotes (bare tokens only)"
    fi
  fi

  if uvx tooluniverse --help >/dev/null 2>&1; then
    green "OK: uvx tooluniverse --help responds"
  else
    red "FAIL: uvx tooluniverse not working"
    return 1
  fi
}

check_claude_config() {
  echo ""
  echo "=== 4. Claude MCP config ==="

  if [[ ! -f "$CLAUDE_CONFIG" ]]; then
    red "FAIL: $CLAUDE_CONFIG not found"; return 1
  fi

  if python -c "import json; json.load(open(r'$CLAUDE_CONFIG_WIN', encoding='utf-8'))" 2>/dev/null; then
    green "OK: $CLAUDE_CONFIG is valid JSON"
  else
    red "FAIL: $CLAUDE_CONFIG is invalid JSON"; return 1
  fi

  if python -c "
import json, sys
d = json.load(open(r'$CLAUDE_CONFIG_WIN', encoding='utf-8'))
def has_tu(obj):
    if isinstance(obj, dict):
        if 'tooluniverse' in obj: return True
        return any(has_tu(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_tu(v) for v in obj)
    return False
sys.exit(0 if has_tu(d) else 1)
" 2>/dev/null; then
    green "OK: tooluniverse MCP entry present"
  else
    yellow "WARN: tooluniverse not found in $CLAUDE_CONFIG (MCP not registered yet — run Task 5)"
  fi
}

do_uninstall() {
  echo "=== Uninstall ==="
  yellow "This will remove venv, restore ~/.claude.json from PRE-INSTALL backup, prune uv cache, delete memory files."
  yellow "Press Ctrl+C to abort, Enter to continue."
  read -r _

  # 1. Remove venv
  if [[ -d "$VENV" ]]; then
    rm -rf "$VENV" && green "Removed venv"
  fi

  # 2. Restore ~/.claude.json from the dedicated pre-install backup
  local pre_bak="$CLAUDE_CONFIG.bak-pre-tooluniverse"
  if [[ -f "$pre_bak" ]]; then
    cp "$pre_bak" "$CLAUDE_CONFIG" && green "Restored $CLAUDE_CONFIG from $pre_bak"
  else
    yellow "No .bak-pre-tooluniverse found — attempting JSON surgery to remove tooluniverse key"
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

  # 3. Prune uv cache
  uv cache prune 2>/dev/null && green "Pruned uv cache" || yellow "uv cache prune unavailable — skipped"

  # 4. Remove memory entries from both possible dirs
  local bionlp_mem="/c/Users/Prekzursil/.claude/projects/C--Users-Prekzursil-Downloads-bionlp/memory"
  local swfoc_mem="/c/Users/Prekzursil/.claude/projects/C--Users-Prekzursil-Downloads-SWFOC-editor/memory"
  for dir in "$bionlp_mem" "$swfoc_mem"; do
    rm -f "$dir/bionlp_workspace.md" "$dir/bionlp_pointer.md" 2>/dev/null && green "Removed memory in $dir" || true
  done

  # 5. (Optional) uninstall the npx skills package-set
  yellow "Optionally remove installed Agent Skills: npx skills remove mims-harvard/ToolUniverse"

  yellow "Workspace folder $WORKSPACE left intact — delete manually if desired."
  green "Uninstall complete."
}

case "${1:-health}" in
  prereqs)   check_prereqs ;;
  health)
    check_prereqs || exit 1
    check_venv_and_sdk || true
    check_env_and_uvx || true
    check_claude_config || true
    echo ""
    green "Health check complete."
    ;;
  uninstall) do_uninstall ;;
  *) echo "Usage: bash bionlp.sh {prereqs|health|uninstall}"; exit 1 ;;
esac
