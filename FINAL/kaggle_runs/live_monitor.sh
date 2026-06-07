#!/bin/bash
# Live monitor: polls `kernels output` and streams NEW log lines as they appear.
# Distinguishes the current run by the v2 marker "GPU compute OK on Tesla T4".
# Usage: MAXCYCLES=14 INTERVAL=35 live_monitor.sh <slug1> [slug2 ...]
if [ -z "$KAGGLE_API_TOKEN" ]; then echo "ERROR: KAGGLE_API_TOKEN not set" >&2; exit 1; fi
USER=prekzursil
ROOT="/c/Users/Prekzursil/Downloads/bionlp/FINAL/kaggle_runs/outputs_live"
mkdir -p "$ROOT"
SLUGS="$*"
MAXCYCLES="${MAXCYCLES:-14}"
INTERVAL="${INTERVAL:-35}"
declare -A seen
echo "[live] monitoring: $SLUGS (every ${INTERVAL}s, up to ${MAXCYCLES} cycles)"
for i in $(seq 1 "$MAXCYCLES"); do
  echo ""; echo "######## cycle $i/$MAXCYCLES @ $(date -u +%H:%M:%S) UTC ########"
  all_term=1
  for slug in $SLUGS; do
    d="$ROOT/$slug"; rm -rf "$d"; mkdir -p "$d"
    kaggle kernels output "$USER/$slug" -p "$d" >/dev/null 2>&1
    log=$(ls "$d"/*_log.txt 2>/dev/null | head -1)
    if [ -z "$log" ]; then echo "[$slug] (no committed output yet)"; all_term=0; continue; fi
    n=$(wc -l < "$log"); prev=${seen[$slug]:-0}
    live="STALE-v1"; grep -q "GPU compute OK" "$log" && live="LIVE-v2"
    if [ "$n" -gt "$prev" ]; then
      echo "----- [$slug] $live  (+$((n-prev)) new / $n total) -----"
      tail -n +$((prev+1)) "$log" | sed 's/^/  | /'
      seen[$slug]=$n
    else
      echo "[$slug] $live  no new lines ($n total)"
    fi
    grep -qE "^DONE: model=|^FAILED:" "$log" || all_term=0
  done
  if [ "$all_term" = "1" ]; then echo ""; echo "[live] all kernels terminal"; break; fi
  sleep "$INTERVAL"
done
echo ""; echo "[live] monitor window ended @ $(date -u +%H:%M:%S) UTC"