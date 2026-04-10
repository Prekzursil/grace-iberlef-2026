#!/bin/bash
# GRACE @ IberLEF 2026 — Full experiment sweep on RunPod
#
# Phase 2 (backbone discovery) + Phase 3 (loss/architecture) in one go.
# Total: 12 experiments × ~15 min each = ~3 hours on A100 80GB.
#
# Usage:
#   bash scripts/run_full_sweep.sh [--out experiments/runs]

set -e

OUT="${1:-experiments/runs}"

echo "============================================================"
echo "  GRACE Full Experiment Sweep"
echo "  Phase 2: 7 backbone configs"
echo "  Phase 3: 5 architecture configs"
echo "  Output: $OUT"
echo "============================================================"

# Phase 2: Backbone Discovery
echo ""
echo "############################################################"
echo "# PHASE 2: Backbone Discovery"
echo "############################################################"

for cfg in beto_base rigoberta2 rigoberta2_lowlr rigoberta2_midlr mdeberta_ctebmsp mdeberta_ctebmsp_lowlr beto_clinical; do
    echo ""
    echo "--- Config: $cfg ---"
    python scripts/train_track1.py \
        --config "configs/track1/${cfg}.yaml" \
        --seed 42 \
        --out "$OUT" || echo "FAILED: $cfg (continuing...)"
done

# Phase 3: Architecture Upgrades (using BETO — swap backbone if Phase 2 winner is different)
echo ""
echo "############################################################"
echo "# PHASE 3: Loss Function & Architecture Upgrades"
echo "############################################################"

for cfg in beto_focal beto_focal_g2 beto_crf beto_focal_crf beto_focal_crf_bilstm; do
    echo ""
    echo "--- Config: $cfg ---"
    python scripts/train_track1.py \
        --config "configs/track1/${cfg}.yaml" \
        --seed 42 \
        --out "$OUT" || echo "FAILED: $cfg (continuing...)"
done

# Phase 4: NLI Relation Classifiers
echo ""
echo "############################################################"
echo "# PHASE 4: NLI Relation Classifiers"
echo "############################################################"

for cfg in beto_nli_small beto_nli_base beto_nli_xlmr; do
    echo ""
    echo "--- Config: $cfg ---"
    python scripts/train_track1.py \
        --config "configs/track1/${cfg}.yaml" \
        --seed 42 \
        --out "$OUT" || echo "FAILED: $cfg (continuing...)"
done

echo ""
echo "============================================================"
echo "  Sweep complete!"
echo "============================================================"
echo ""
echo "Results summary:"
echo ""
printf "%-50s  %8s  %8s  %8s\n" "Config" "S1-F1" "S2-F1" "Overall"
printf "%-50s  %8s  %8s  %8s\n" "------" "-----" "-----" "-------"
for d in "$OUT"/*/; do
    if [ -f "$d/metrics.json" ]; then
        tag=$(basename "$d" | sed 's/^[0-9T:.-]*-//')
        metrics=$(python -c "
import json, sys
m=json.load(open('${d}metrics.json'))['summary']
print(f'{m[\"subtask1_official\"]:.4f}  {m[\"subtask2_official\"]:.4f}  {m[\"overall\"]:.4f}')
" 2>/dev/null || echo "ERROR")
        printf "%-50s  %s\n" "$tag" "$metrics"
    fi
done
