"""GRACE Track-1 sweep on Modal — 9 models x 3 seeds, massively parallel.

Why Modal: real GPUs we choose (A10G/Ampere -> no Kaggle P100/torch-2.10 hang),
pinned reproducible Image, bf16 for DeBERTa (fp32 range -> no NaN), live-streamed
logs, and .map() fan-out (no 2-concurrent cap).

Setup (one-time, you run it):  python -m modal setup
Run:                           modal run modal_sweep.py
Collect artifacts after:       modal volume get grace-out / ./modal_out
"""
from __future__ import annotations

import modal

app = modal.App("grace-sweep")

# Image: torch 2.4.1 (stable on Ampere + DeBERTa) + transformers 4.48 (ModernBERT).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.48.0",
        "sentencepiece",
        "accelerate>=0.34",
        "datasets>=2.20",
        "pandas",
        "numpy<2",
        "spacy",
        "click>=8.1",          # spaCy 3.8 imports its CLI (needs click) on `import spacy`
        "scikit-learn",
    )
    .run_commands("python -m spacy download es_core_news_sm")
    # bake the GRACE data into the image (run `modal run` from FINAL/)
    .add_local_dir("IberLEF_GRACE", "/data/IberLEF_GRACE", copy=True)
    .add_local_dir("GRACE_2026_blind_test", "/data/GRACE_2026_blind_test", copy=True)
)

OUT = modal.Volume.from_name("grace-out", create_if_missing=True)

# (model_id, prefix, fp16, bf16, attn_eager)
MODELS = [
    ("BSC-LT/MrBERT-biomed", "mrbert_biomed", True, False, True),
    ("PlanTL-GOB-ES/bsc-bio-ehr-es", "bsc_bio_ehr", True, False, False),
    ("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es", "roberta_clinical", True, False, False),
    ("microsoft/mdeberta-v3-base", "mdeberta_v3", False, True, False),  # bf16 -> no NaN
    ("IIC/RigoBERTa-Clinical", "rigoberta_clinical", True, False, False),
    ("BSC-LT/MrBERT-es", "mrbert_es", True, False, True),
    ("FacebookAI/xlm-roberta-large", "xlmr_large", True, False, False),
    ("dccuchile/bert-base-spanish-wwm-cased", "beto", True, False, False),
    ("IIC/BETO_Galen", "beto_galen", True, False, False),
]
SEEDS = [42, 100, 2026]
JOBS = [
    {"model": m, "prefix": p, "fp16": f16, "bf16": bf, "eager": eg, "seed": s}
    for (m, p, f16, bf, eg) in MODELS for s in SEEDS
]


@app.function(image=image, gpu="A10G", volumes={"/out": OUT}, timeout=60 * 60,
              max_containers=12, retries=1)
def train_one(job: dict) -> dict:
    import os, json, re, random, glob, gc
    import numpy as np, pandas as pd, torch
    import torch.nn.functional as F
    import spacy
    from collections import defaultdict
    from typing import NamedTuple
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback)
    from datasets import Dataset

    MODEL_NAME, FILE_PREFIX, SEED = job["model"], job["prefix"], job["seed"]
    USE_FP16, USE_BF16, EAGER = job["fp16"], job["bf16"], job["eager"]
    MAX_LEN, OUTD = 192, "/out"
    tag = f"{FILE_PREFIX}_seed{SEED}"
    print(f"=== {MODEL_NAME} | seed={SEED} | fp16={USE_FP16} bf16={USE_BF16} eager={EAGER} ===", flush=True)
    print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)

    def seed_everything(s):
        random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
        np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    seed_everything(SEED)

    def find(name):
        return sorted(glob.glob(f"/data/**/{name}", recursive=True), key=len)[0]
    TRAIN, DEV, BLIND = find("track_1_train.json"), find("track_1_dev.json"), find("track_1_blind_test.json")
    LABEL_MAP = {"None": 0, "Premise": 1, "Claim": 2, "MajorClaim": 3}
    INV = {v: k for k, v in LABEL_MAP.items()}
    nlp = spacy.load("es_core_news_sm")

    def parse(path):
        data = json.load(open(path, encoding="utf-8")); recs = []
        for doc in data:
            rt = doc["raw_text"]; ann = doc.get("annotations", {}).get("entities", [])
            spans = [(a["start"], a["end"]) for a in ann]
            sr = [{"text": a["text"].strip(), "label": a["type"], "start": a["start"]} for a in ann]
            for s in nlp(rt).sents:
                if not any(max(s.start_char, a) < min(s.end_char, b) for a, b in spans) and len(s.text.strip()) > 10:
                    sr.append({"text": s.text.strip(), "label": "None", "start": s.start_char})
            sr.sort(key=lambda x: x["start"])
            for i, r in enumerate(sr):
                recs.append({"text": r["text"], "prev_text": sr[i-1]["text"] if i > 0 else "", "label": r["label"]})
        return pd.DataFrame(recs)

    def parse_blind(path):
        data = json.load(open(path, encoding="utf-8")); recs = []
        for doc in data:
            rt = doc["raw_text"]
            sr = [{"text": s.text.strip(), "start": s.start_char, "end": s.end_char}
                  for s in nlp(rt).sents if len(s.text.strip()) > 10]
            sr.sort(key=lambda x: x["start"])
            for i, r in enumerate(sr):
                recs.append({"text": r["text"], "prev_text": sr[i-1]["text"] if i > 0 else "", "label": 0})
        return pd.DataFrame(recs)

    p1_tr, p1_dev = parse(TRAIN), parse(DEV)
    p2_tr = pd.concat([p1_tr, p1_dev], ignore_index=True)
    p2_te = parse_blind(BLIND)
    for df in (p1_tr, p1_dev, p2_tr):
        df["label"] = df["label"].map(LABEL_MAP)
    print(f"sentences: train={len(p1_tr)} dev={len(p1_dev)} merged={len(p2_tr)} blind={len(p2_te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tok_fn(ex):
        return tok(ex["prev_text"], ex["text"], padding="max_length", truncation=True,
                   max_length=MAX_LEN, return_token_type_ids=False)
    p1_tr_ds = Dataset.from_pandas(p1_tr).map(tok_fn, batched=True)
    p1_dev_ds = Dataset.from_pandas(p1_dev).map(tok_fn, batched=True)
    p2_tr_ds = Dataset.from_pandas(p2_tr).map(tok_fn, batched=True)
    p2_te_ds = Dataset.from_pandas(p2_te).map(tok_fn, batched=True)

    dev_raw = json.load(open(DEV, encoding="utf-8")); templ = []
    for doc in dev_raw:
        rt = doc["raw_text"]; ann = doc.get("annotations", {}).get("entities", [])
        spans = [(a["start"], a["end"]) for a in ann]
        sr = [{"text": a["text"].strip(), "start": a["start"], "end": a["end"]} for a in ann]
        for s in nlp(rt).sents:
            if not any(max(s.start_char, a) < min(s.end_char, b) for a, b in spans) and len(s.text.strip()) > 10:
                sr.append({"text": s.text.strip(), "start": s.start_char, "end": s.end_char})
        sr.sort(key=lambda x: x["start"])
        templ.append({"id": doc["id"], "raw_text": rt, "annotations": {"entities": ann}, "sent_records": sr})

    TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    def toks(t): return [(m.start(), m.end()) for m in TOK_RE.finditer(t)]
    def tset(tp, s, e): return frozenset(i for i, (a, b) in enumerate(tp) if a >= s and b <= e)
    class CT(NamedTuple):
        start: int; end: int; type: str; tokens: frozenset
    def exact(a, b): return a.type == b.type and a.start == b.start and a.end == b.end
    def match(A, B):
        used, m, ua = set(), [], []
        for a in A:
            j = next((j for j, b in enumerate(B) if j not in used and exact(a, b)), None)
            if j is not None: m.append((a, B[j])); used.add(j)
            else: ua.append(a)
        return m, ua, [B[j] for j in range(len(B)) if j not in used]
    def strict_prf(cases):
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
        for c in cases:
            tps = toks(c["raw_text"])
            g = [CT(e["start"], e["end"], e["type"], tset(tps, e["start"], e["end"])) for e in c["annotations"]["entities"]]
            pr = [CT(e["start"], e["end"], e["type"], tset(tps, e["start"], e["end"])) for e in c["predictions"]["entities"]]
            mm, ug, up = match(g, pr)
            for x, _ in mm: tp[x.type] += 1
            for x in up: fp[x.type] += 1
            for x in ug: fn[x.type] += 1
        out, f1s = {}, []
        for t in ["Premise", "Claim", "MajorClaim"]:
            p = tp[t]/(tp[t]+fp[t]) if (tp[t]+fp[t]) else 0.0
            r = tp[t]/(tp[t]+fn[t]) if (tp[t]+fn[t]) else 0.0
            f = 2*p*r/(p+r) if (p+r) else 0.0
            out[t] = {"p": p, "r": r, "f1": f}; f1s.append(f)
        out["macro_f1"] = sum(f1s)/len(f1s) if f1s else 0.0
        return out
    def build_cases(preds):
        t2p = {t: int(p) for t, p in zip(p1_dev["text"], preds)}
        cs = []
        for tp in templ:
            ents, eid = [], 1
            for sr in tp["sent_records"]:
                lab = INV.get(t2p.get(sr["text"], 0), "None")
                if lab != "None":
                    ents.append({"id": f"T{eid}", "text": sr["text"], "start": sr["start"], "end": sr["end"], "type": lab}); eid += 1
            cs.append({"id": tp["id"], "raw_text": tp["raw_text"], "annotations": tp["annotations"], "predictions": {"entities": ents, "relations": []}})
        return cs

    METRICS, EPOCH = [], [0]
    def compute_metrics(ep):
        logits, _ = ep
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(probs, axis=-1); preds[probs[:, 3] > 0.50] = 3
        EPOCH[0] += 1
        r = strict_prf(build_cases(preds))
        METRICS.append({"model": MODEL_NAME, "seed": SEED, "epoch": EPOCH[0], "macro_f1": r["macro_f1"],
                        "Premise_f1": r["Premise"]["f1"], "Claim_f1": r["Claim"]["f1"], "MajorClaim_f1": r["MajorClaim"]["f1"]})
        print(f"--- EPOCH {EPOCH[0]} DEV --- macro={r['macro_f1']:.4f} P={r['Premise']['f1']:.4f} "
              f"C={r['Claim']['f1']:.4f} MC={r['MajorClaim']['f1']:.4f}", flush=True)
        return {"strict_macro_f1": r["macro_f1"]}

    def cweights(df):
        c = np.bincount(df["label"].values, minlength=4).astype(float)
        base = c.sum()/(len(c)*c); w = (base/base.mean()).tolist(); w[3] = 10.0
        return w

    class WTrainer(Trainer):
        def __init__(self, *a, cw=None, **k): super().__init__(*a, **k); self.cw = cw
        def compute_loss(self, model, inputs, return_outputs=False, **k):
            labels = inputs.pop("labels"); out = model(**inputs)
            w = torch.tensor(self.cw, dtype=out.logits.dtype, device=out.logits.device)
            loss = F.cross_entropy(out.logits, labels, weight=w)
            return (loss, out) if return_outputs else loss

    mk_kwargs = {"num_labels": 4, "ignore_mismatched_sizes": True}
    if EAGER: mk_kwargs["attn_implementation"] = "eager"
    def make_model(): return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **mk_kwargs)
    def make_optim(m):
        head = [p for n, p in m.named_parameters() if "classifier" in n]
        base = [p for n, p in m.named_parameters() if "classifier" not in n]
        return torch.optim.AdamW([{"params": base, "lr": 1e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.05)

    class Best(TrainerCallback):
        def __init__(self): self.best, self.epoch = 0.0, 1
        def on_evaluate(self, args, state, control, metrics, **k):
            f = metrics.get("eval_strict_macro_f1", 0.0)
            if f > self.best: self.best, self.epoch = f, round(state.epoch)

    common = dict(warmup_steps=100, lr_scheduler_type="cosine", per_device_train_batch_size=16,
                  per_device_eval_batch_size=32, fp16=USE_FP16, bf16=USE_BF16, report_to="none",
                  disable_tqdm=True, label_smoothing_factor=0.15, seed=SEED, data_seed=SEED)

    # ---- Phase 1 ----
    print("PHASE 1: search best epoch", flush=True)
    m1 = make_model()
    a1 = TrainingArguments(output_dir="/tmp/p1", eval_strategy="epoch", save_strategy="epoch",
                           num_train_epochs=15, load_best_model_at_end=True,
                           metric_for_best_model="strict_macro_f1", greater_is_better=True,
                           save_total_limit=1, save_only_model=True, **common)
    best = Best()
    t1 = WTrainer(model=m1, args=a1, train_dataset=p1_tr_ds, eval_dataset=p1_dev_ds,
                  compute_metrics=compute_metrics,
                  callbacks=[best, EarlyStoppingCallback(early_stopping_patience=4)],
                  optimizers=(make_optim(m1), None), cw=cweights(p1_tr))
    t1.train()
    BEST, TARGET = best.epoch, best.epoch + 1
    dev_logits = t1.predict(p1_dev_ds).predictions
    np.save(f"{OUTD}/{tag}_dev_logits.npy", dev_logits)
    dp = F.softmax(torch.tensor(dev_logits), dim=-1).numpy(); dpred = np.argmax(dp, axis=-1); dpred[dp[:, 3] > 0.50] = 3
    cases = build_cases(dpred); final = strict_prf(cases)
    json.dump([{"id": c["id"], "raw_text": c["raw_text"], "predictions": c["predictions"]} for c in cases],
              open(f"{OUTD}/{tag}_dev_predictions.json", "w", encoding="utf-8"), ensure_ascii=False)
    pd.DataFrame(METRICS).to_csv(f"{OUTD}/{tag}_metrics.csv", index=False)
    json.dump({"model": MODEL_NAME, "seed": SEED, "best_epoch": BEST, "best_model_dev": final, "per_epoch": METRICS},
              open(f"{OUTD}/{tag}_metrics.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"BEST EPOCH={BEST} dev macro={final['macro_f1']:.4f}", flush=True)
    del m1, t1; gc.collect(); torch.cuda.empty_cache()

    # ---- Phase 2: full-fit + blind logits (final epoch) ----
    print(f"PHASE 2: full-fit {TARGET} epochs + blind logits", flush=True)
    m2 = make_model()
    class Blind(TrainerCallback):
        def __init__(self): self.ref = None
        def on_epoch_end(self, args, state, control, **k):
            ep = round(state.epoch)
            if self.ref is not None and ep >= TARGET:
                np.save(f"{OUTD}/{tag}_epoch{ep}_blind_logits.npy", self.ref.predict(p2_te_ds).predictions)
                print(f"saved blind logits epoch {ep}", flush=True)
    a2 = TrainingArguments(output_dir="/tmp/p2", eval_strategy="no", save_strategy="no",
                           num_train_epochs=TARGET, **common)
    bl = Blind()
    t2 = WTrainer(model=m2, args=a2, train_dataset=p2_tr_ds, callbacks=[bl], optimizers=(make_optim(m2), None), cw=cweights(p2_tr))
    bl.ref = t2
    t2.train()
    OUT.commit()
    print(f"DONE {tag}", flush=True)
    return {"model": MODEL_NAME, "prefix": FILE_PREFIX, "seed": SEED,
            "best_epoch": BEST, "macro_f1": final["macro_f1"],
            "Premise": final["Premise"]["f1"], "Claim": final["Claim"]["f1"], "MajorClaim": final["MajorClaim"]["f1"]}


@app.local_entrypoint()
def main():
    import json
    from collections import defaultdict
    results = list(train_one.map(JOBS, return_exceptions=True))
    rows = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if not isinstance(r, dict)]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["prefix"]].append(r)
    print("\n================ GRACE Track-1 head-to-head (dev macro F1) ================")
    print(f"{'model':24s} {'mean':>7s} {'std':>6s}  per-seed")
    import statistics
    summary = []
    for prefix, rs in sorted(by_model.items(), key=lambda kv: -statistics.mean([x['macro_f1'] for x in kv[1]])):
        ms = [x["macro_f1"] for x in rs]
        mean = statistics.mean(ms); std = statistics.pstdev(ms) if len(ms) > 1 else 0.0
        seeds = ", ".join(f"s{x['seed']}={x['macro_f1']:.4f}" for x in sorted(rs, key=lambda x: x["seed"]))
        print(f"{prefix:24s} {mean:7.4f} {std:6.4f}  {seeds}")
        summary.append({"model": rs[0]["model"], "prefix": prefix, "mean_macro": mean, "std": std, "runs": rs})
    if errs:
        print(f"\n{len(errs)} job(s) errored:")
        for e in errs[:10]:
            print("  ", str(e)[:160])
    json.dump(summary, open("modal_results_summary.json", "w"), indent=2, default=str)
    print("\nsummary -> modal_results_summary.json ; artifacts in volume 'grace-out' "
          "(modal volume get grace-out / ./modal_out)")
