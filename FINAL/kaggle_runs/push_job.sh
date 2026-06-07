#!/bin/bash
# Reliable kernel launch: CPU-create (init.py, no machine_shape/dataset -> always
# succeeds on a fresh slug) THEN full T4 update (real script + dataset + T4).
# The [run] step needs a free GPU slot (Kaggle cap = 2 concurrent).
# Usage: push_job.sh <kernel_dir>
if [ -z "$KAGGLE_API_TOKEN" ]; then echo "ERROR: KAGGLE_API_TOKEN not set" >&2; exit 1; fi
d="${1%/}"; meta="$d/kernel-metadata.json"
[ -f "$meta" ] || { echo "no metadata in $d" >&2; exit 1; }
find "$d" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
cp "$meta" "$meta.full"
python - "$d" <<'PY'
import json, sys, os
d = sys.argv[1]; p = os.path.join(d, "kernel-metadata.json"); m = json.load(open(p))
mini = {"id": m["id"], "title": m["title"], "code_file": "init.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False}
json.dump(mini, open(p, "w"), indent=2)
PY
echo -n "[create] "; kaggle kernels push -p "$d" 2>&1 | head -1
mv "$meta.full" "$meta"
echo -n "[run]    "; kaggle kernels push -p "$d" 2>&1 | head -1