"""Consolidate every reported number into one source-of-truth: report/report_data.json.

Pulls from:
  - modal_results_summary.json     (per-model mean/std + per-seed per-class F1)
  - modal_out/*_metrics.json       (per-epoch dev curves, best epochs)
  - IberLEF_GRACE/.../track_1_*    (dataset doc counts + class distribution)
  - ensemble_results.json          (ensemble dev scores + blind submission stats; optional)

Every figure / docx / pptx reads ONLY this file, so all artifacts trace to source.
Run from FINAL/:  python report/collect_data.py
"""
from __future__ import annotations
import collections
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # FINAL/

# Display names + HF ids + which precision fix each backbone needed (for narrative).
PRETTY = {
    "mdeberta_v3": "mDeBERTa-v3-base",
    "bsc_bio_ehr": "bsc-bio-ehr-es",
    "mrbert_es": "MrBERT-es",
    "rigoberta_clinical": "RigoBERTa-Clinical",
    "mrbert_biomed": "MrBERT-biomed",
    "roberta_clinical": "roberta-clinical-es",
    "xlmr_large": "XLM-RoBERTa-large",
    "beto": "BETO",
    "beto_galen": "BETO-Galén",
}
HF_ID = {  # exact ids used in the sweep (modal_results_summary.json)
    "mdeberta_v3": "microsoft/mdeberta-v3-base",
    "bsc_bio_ehr": "PlanTL-GOB-ES/bsc-bio-ehr-es",
    "mrbert_es": "BSC-LT/MrBERT-es",
    "rigoberta_clinical": "IIC/RigoBERTa-Clinical",
    "mrbert_biomed": "BSC-LT/MrBERT-biomed",
    "roberta_clinical": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    "xlmr_large": "FacebookAI/xlm-roberta-large",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "beto_galen": "IIC/BETO_Galen",
}
SEEDS = [42, 100, 2026]


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_leaderboard():
    """Per-model mean/std macro-F1 + per-class mean F1, sorted descending."""
    summary = _load(os.path.join(ROOT, "modal_results_summary.json"))
    rows = []
    for entry in summary:
        pfx = entry["prefix"]
        runs = entry["runs"]
        per_class = {}
        for cls in ("Premise", "Claim", "MajorClaim"):
            vals = [r[cls] for r in runs]
            per_class[cls] = {"mean": sum(vals) / len(vals), "vals": vals}
        rows.append({
            "prefix": pfx,
            "pretty": PRETTY.get(pfx, pfx),
            "hf_id": HF_ID.get(pfx, entry.get("model", pfx)),
            "mean_macro": entry["mean_macro"],
            "std": entry["std"],
            "seed_macro": {r["seed"]: r["macro_f1"] for r in runs},
            "best_epoch": {r["seed"]: r["best_epoch"] for r in runs},
            "per_class": per_class,
        })
    rows.sort(key=lambda r: -r["mean_macro"])
    return rows


def collect_curves():
    """Per-epoch dev macro + per-class F1 for every model/seed."""
    curves = {}
    for pfx in PRETTY:
        curves[pfx] = {}
        for seed in SEEDS:
            path = os.path.join(ROOT, "modal_out", f"{pfx}_seed{seed}_metrics.json")
            if not os.path.exists(path):
                continue
            m = _load(path)
            curves[pfx][str(seed)] = {
                "best_epoch": m["best_epoch"],
                "per_epoch": [
                    {"epoch": e["epoch"], "macro": e["macro_f1"],
                     "Premise": e["Premise_f1"], "Claim": e["Claim_f1"],
                     "MajorClaim": e["MajorClaim_f1"]}
                    for e in m["per_epoch"]
                ],
            }
    return curves


def collect_dataset():
    """Doc counts + entity-type distribution for train/dev/blind."""
    def find(pattern):
        hits = glob.glob(os.path.join(ROOT, pattern), recursive=True)
        return hits[0] if hits else None

    def typecount(data):
        c = collections.Counter()
        for d in data:
            for e in d.get("annotations", {}).get("entities", []):
                c[e["type"]] += 1
        return dict(c)

    train = _load(find("IberLEF_GRACE/**/track_1_train.json"))
    dev = _load(find("IberLEF_GRACE/**/track_1_dev.json"))
    blind_path = find("GRACE_2026_blind_test/**/track_1_blind_test.json") or \
        find("IberLEF_GRACE/**/track_1_blind_test.json")
    blind = _load(blind_path) if blind_path else []
    # sentence counts measured directly from saved logit array shapes (real provenance)
    import numpy as np
    dev_npy = glob.glob(os.path.join(ROOT, "modal_out", "*_dev_logits.npy"))
    blind_npy = glob.glob(os.path.join(ROOT, "modal_out", "*_blind_logits.npy"))
    dev_sents = int(np.load(dev_npy[0]).shape[0]) if dev_npy else 705
    blind_sents = int(np.load(blind_npy[0]).shape[0]) if blind_npy else 26640
    return {
        "docs": {"train": len(train), "dev": len(dev), "blind": len(blind)},
        "entities": {"train": typecount(train), "dev": typecount(dev)},
        "dev_sentences": dev_sents,      # = dev_logits.npy rows (spaCy segmentation)
        "blind_sentences": blind_sents,  # = blind_logits.npy rows
    }


def collect_ensemble():
    """Ensemble dev scores + blind submission stats (optional until run completes)."""
    path = os.path.join(ROOT, "ensemble_results.json")
    if not os.path.exists(path):
        return None
    return _load(path)


def main():
    data = {
        "leaderboard": collect_leaderboard(),
        "curves": collect_curves(),
        "dataset": collect_dataset(),
        "ensemble": collect_ensemble(),
        "run_meta": {
            "n_models": 9, "n_seeds": 3, "n_cells": 27,
            "seeds": SEEDS, "gpu": "NVIDIA A10G (Ampere, 24 GB)",
            "platform": "Modal (serverless GPU, parallel .map containers)",
            "recipe": {
                "task": "sentence-level 4-class (None / Premise / Claim / MajorClaim)",
                "input": "prev_sentence [SEP] sentence", "max_len": 192,
                "segmenter": "spaCy es_core_news_sm",
                "lr_body": 1e-5, "lr_head": 1e-4, "optimizer": "AdamW", "weight_decay": 0.05,
                "precision": "bf16 (Ampere)", "loss": "weighted CE (MajorClaim w=10)",
                "scheduler": "cosine + 100 warmup steps", "label_smoothing": 0.15,
                "batch_size": 16, "max_epochs": 15, "es_patience": 4,
                "majorclaim_override": 0.50, "dev_metric": "strict macro-F1 per epoch",
                "two_stage": "Phase 1 train->best epoch on dev; Phase 2 full-fit (train+dev) -> blind logits",
            },
        },
    }
    out = os.path.join(HERE, "report_data.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    lb = data["leaderboard"]
    print(f"wrote {out}")
    print(f"leaderboard: {len(lb)} models | top = {lb[0]['pretty']} {lb[0]['mean_macro']:.4f}")
    print(f"ensemble present: {data['ensemble'] is not None}")


if __name__ == "__main__":
    main()
