#!/bin/bash
# Poll a wave of Kaggle kernels until each truly finishes.
# Usage: poll_wave.sh <slug1> [slug2 ...]
# Status endpoint 500s server-side, so we re-pull `kernels output` each cycle
# (returns a fresh snapshot of /kaggle/working) and detect TERMINAL state via:
#   success  -> tee log contains "DONE: model="
#   failure  -> tee log contains "FAILED:" or "Traceback (most recent call last)"
#               OR the Kaggle exec .log is non-empty without a DONE marker
# Auth: relies on KAGGLE_API_TOKEN already in the environment (do NOT hardcode it).
if [ -z "$KAGGLE_API_TOKEN" ]; then
  echo "ERROR: KAGGLE_API_TOKEN not set in environment." >&2
  exit 1
fi
USER=prekzursil
ROOT="/c/Users/Prekzursil/Downloads/bionlp/FINAL/kaggle_runs/outputs"
mkdir -p "$ROOT"
SLUGS="$*"
echo "[poller] watching: $SLUGS"
for i in $(seq 1 160); do          # up to 160 * 120s ~= 5.3h
  all_done=1
  for slug in $SLUGS; do
    d="$ROOT/$slug"
    [ -f "$d/_TERMINAL" ] && continue
    rm -rf "$d/_pull"; mkdir -p "$d/_pull"
    kaggle kernels output "$USER/$slug" -p "$d/_pull" >/dev/null 2>&1
    log=$(ls "$d/_pull"/*_log.txt 2>/dev/null | head -1)
    klog=$(ls "$d/_pull/$slug.log" 2>/dev/null | head -1)
    state=""
    if [ -n "$log" ] && grep -q "^DONE: model=" "$log"; then state="SUCCESS"; fi
    if [ -z "$state" ] && [ -n "$log" ] && grep -qE "^FAILED:|Traceback \(most recent call last\)" "$log"; then state="FAIL"; fi
    if [ -z "$state" ] && [ -n "$klog" ] && [ -s "$klog" ]; then
        # Kaggle wrote a non-empty exec log => run ended; classify by DONE marker
        if [ -n "$log" ] && grep -q "^DONE: model=" "$log"; then state="SUCCESS"; else state="FAIL"; fi
    fi
    if [ -n "$state" ]; then
      # persist the final pull as the artifacts dir
      cp -f "$d/_pull"/* "$d"/ 2>/dev/null
      echo "$state" > "$d/_TERMINAL"
      echo "[poller] $state: $slug"
    else
      all_done=0
    fi
  done
  [ "$all_done" = "1" ] && { echo "[poller] wave finished"; break; }
  sleep 120
done
echo ""
echo "================ WAVE SUMMARY ================"
for slug in $SLUGS; do
  d="$ROOT/$slug"; log=$(ls "$d"/*_log.txt 2>/dev/null | head -1)
  state=$(cat "$d/_TERMINAL" 2>/dev/null || echo "TIMEOUT/RUNNING")
  echo "----- $slug [$state] -----"
  if [ -n "$log" ]; then
    echo "[artifacts]"; ls "$d" 2>/dev/null | grep -E "\.(npy|csv|json)$" | head -40
    echo "[best-model dev macro]"; grep -E "Best-model dev macro|BEST EPOCH=" "$log" | tail -2
    echo "[last 12 log lines]"; tail -12 "$log"
  else
    echo "NO OUTPUT (still running or failed before first commit)"
  fi
  echo ""
done
