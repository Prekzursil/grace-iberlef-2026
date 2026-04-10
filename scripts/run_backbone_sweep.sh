#!/bin/bash
# GRACE @ IberLEF 2026 — Backbone sweep on RunPod
#
# Run this on the pod after setup. Tests all 7 configs (BETO baseline + 6 new)
# with the full pipeline (tagger + relation classifier).
#
# Usage:
#   bash scripts/run_backbone_sweep.sh [--out experiments/runs]

set -e

OUT="${1:-experiments/runs}"

echo "=== GRACE Backbone Sweep ==="
echo "Output directory: $OUT"
echo ""

for cfg in beto_base rigoberta2 rigoberta2_lowlr rigoberta2_midlr mdeberta_ctebmsp mdeberta_ctebmsp_lowlr beto_clinical; do
    echo ""
    echo "################################################################"
    echo "# Config: $cfg"
    echo "################################################################"
    python scripts/train_track1.py \
        --config "configs/track1/${cfg}.yaml" \
        --seed 42 \
        --out "$OUT"
    echo ""
done

echo ""
echo "=== Sweep complete ==="
echo "Results in: $OUT"
echo ""
echo "Quick comparison:"
for d in "$OUT"/*/; do
    if [ -f "$d/metrics.json" ]; then
        tag=$(basename "$d")
        metrics=$(python -c "import json; m=json.load(open('$d/metrics.json')); print(f'S1={m[\"summary\"][\"subtask1_official\"]:.4f}  S2={m[\"summary\"][\"subtask2_official\"]:.4f}  overall={m[\"summary\"][\"overall\"]:.4f}')")
        echo "  $tag: $metrics"
    fi
done
