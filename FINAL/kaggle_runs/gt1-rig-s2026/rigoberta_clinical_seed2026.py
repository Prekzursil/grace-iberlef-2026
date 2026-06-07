# =====================================================================
# GRACE Track 1 - Two-Stage Auto-Runner  (model=IIC/RigoBERTa-Clinical, seed=2026)
# Faithful to notebook94c050a41f.ipynb recipe + per-model overrides.
# =====================================================================
import os, sys, subprocess, json, re, random, glob, gc, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=False)

# Per-model transformers spec (ModernBERT needs >=4.48; mDeBERTa pinned for stability).
_pip("transformers>=4.48.0", "sentencepiece", "accelerate")
subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"], check=False)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import spacy
from collections import defaultdict
from typing import NamedTuple
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback,
)
from datasets import Dataset

# ===================== CONFIG =====================
SEED = 2026
MODEL_NAME = "IIC/RigoBERTa-Clinical"
FILE_PREFIX = "rigoberta_clinical"
USE_FP16 = True
EXTRA_EPOCHS_FOR_FULL_FIT = 1
MAX_LEN = 192
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

# ---- tee stdout+stderr to a log file ----
class _Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stdout.flush(); self.f.flush()
    def isatty(self):
        return False
    def __getattr__(self, name):
        # delegate fileno/encoding/buffer/etc. to the real stdout
        std = self.__dict__.get("stdout")
        if std is None:
            raise AttributeError(name)
        return getattr(std, name)

sys.stdout = _Tee(os.path.join(OUT, f"{FILE_PREFIX}_seed{SEED}_log.txt"))
sys.stderr = sys.stdout  # capture tracebacks/stderr into the same artifact log

def _excepthook(et, ev, tb):
    import traceback as _tb
    print(f"\nFAILED: {et.__name__}: {ev}")
    _tb.print_exception(et, ev, tb)
    sys.stdout.flush()
sys.excepthook = _excepthook

print(f"=== GRACE T1 | model={MODEL_NAME} | seed={SEED} | fp16={USE_FP16} ===")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# ---- seed lock ----
def seed_everything(seed):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

# ---- GPU compatibility guard: fail FAST instead of hanging ----
# Kaggle's torch 2.10 dropped Pascal sm_60; a P100 makes CUDA kernels hang.
# A real compute op surfaces the mismatch immediately so we never hang for hours.
if not torch.cuda.is_available():
    print("FAILED: CUDA not available"); raise SystemExit(1)
try:
    _t = torch.randn(8, 8, device="cuda")
    _ = (_t @ _t).sum().item()
    print(f"GPU compute OK on {torch.cuda.get_device_name(0)}")
except Exception as _e:
    print(f"FAILED: GPU '{torch.cuda.get_device_name(0)}' cannot run torch "
          f"{torch.__version__} ({type(_e).__name__}: {_e})")
    raise SystemExit(1)

# ---- robust data discovery (mount path varies) ----
def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return sorted(hits, key=len)[0]

TRAIN_PATH = find_file("track_1_train.json")
DEV_PATH   = find_file("track_1_dev.json")
BLIND_PATH = find_file("track_1_blind_test.json")
print("TRAIN:", TRAIN_PATH); print("DEV:", DEV_PATH); print("BLIND:", BLIND_PATH)

LABEL_MAP = {"None": 0, "Premise": 1, "Claim": 2, "MajorClaim": 3}
INV = {v: k for k, v in LABEL_MAP.items()}
_NLP = spacy.load("es_core_news_sm")

# ---- parsers (prev-sentence context) ----
def parse_grace_json(path):
    data = json.load(open(path, encoding="utf-8"))
    recs = []
    for doc in data:
        rt = doc["raw_text"]
        ann = doc.get("annotations", {}).get("entities", [])
        spans = [(a["start"], a["end"]) for a in ann]
        sr = [{"text": a["text"].strip(), "label": a["type"], "start": a["start"]} for a in ann]
        for sent in _NLP(rt).sents:
            s, e = sent.start_char, sent.end_char
            if not any(max(s, a) < min(e, b) for a, b in spans) and len(sent.text.strip()) > 10:
                sr.append({"text": sent.text.strip(), "label": "None", "start": s})
        sr.sort(key=lambda x: x["start"])
        for i, r in enumerate(sr):
            prev = sr[i - 1]["text"] if i > 0 else ""
            recs.append({"text": r["text"], "prev_text": prev, "label": r["label"]})
    return pd.DataFrame(recs)

def parse_blind(path):
    data = json.load(open(path, encoding="utf-8"))
    recs = []
    for doc in data:
        rt = doc["raw_text"]
        sr = [{"text": s.text.strip(), "start": s.start_char, "end": s.end_char}
              for s in _NLP(rt).sents if len(s.text.strip()) > 10]
        sr.sort(key=lambda x: x["start"])
        for i, r in enumerate(sr):
            prev = sr[i - 1]["text"] if i > 0 else ""
            recs.append({"text": r["text"], "prev_text": prev, "label": 0})
    return pd.DataFrame(recs)

p1_train_df = parse_grace_json(TRAIN_PATH)
p1_dev_df   = parse_grace_json(DEV_PATH)
p2_train_df = pd.concat([p1_train_df, p1_dev_df], ignore_index=True)  # strings, mapped below
p2_test_df  = parse_blind(BLIND_PATH)
p1_train_df["label"] = p1_train_df["label"].map(LABEL_MAP)
p1_dev_df["label"]   = p1_dev_df["label"].map(LABEL_MAP)
p2_train_df["label"] = p2_train_df["label"].map(LABEL_MAP)
print(f"Sentences: p1_train={len(p1_train_df)} p1_dev={len(p1_dev_df)} "
      f"p2_train={len(p2_train_df)} p2_test={len(p2_test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tok_fn(ex):
    return tokenizer(ex["prev_text"], ex["text"], padding="max_length",
                     truncation=True, max_length=MAX_LEN, return_token_type_ids=False)

p1_train_ds = Dataset.from_pandas(p1_train_df).map(tok_fn, batched=True)
p1_dev_ds   = Dataset.from_pandas(p1_dev_df).map(tok_fn, batched=True)
p2_train_ds = Dataset.from_pandas(p2_train_df).map(tok_fn, batched=True)
p2_test_ds  = Dataset.from_pandas(p2_test_df).map(tok_fn, batched=True)

# ---- dev eval template + strict scorer (official rules) ----
dev_raw = json.load(open(DEV_PATH, encoding="utf-8"))
dev_eval_template = []
for doc in dev_raw:
    rt = doc["raw_text"]; ann = doc.get("annotations", {}).get("entities", [])
    spans = [(a["start"], a["end"]) for a in ann]
    sr = [{"text": a["text"].strip(), "start": a["start"], "end": a["end"]} for a in ann]
    for sent in _NLP(rt).sents:
        s, e = sent.start_char, sent.end_char
        if not any(max(s, a) < min(e, b) for a, b in spans) and len(sent.text.strip()) > 10:
            sr.append({"text": sent.text.strip(), "start": s, "end": e})
    sr.sort(key=lambda x: x["start"])
    dev_eval_template.append({"id": doc["id"], "raw_text": rt,
                              "annotations": {"entities": ann}, "sent_records": sr})

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def _toks(t): return [(m.start(), m.end()) for m in TOKEN_RE.finditer(t)]
def _tset(tp, s, e): return frozenset(i for i, (ts, te) in enumerate(tp) if ts >= s and te <= e)
class CT(NamedTuple):
    start: int; end: int; type: str; tokens: frozenset
def _exact(a, b): return a.type == b.type and a.start == b.start and a.end == b.end
def _match(A, B, fn):
    used, m, ua = set(), [], []
    for a in A:
        j = next((j for j, b in enumerate(B) if j not in used and fn(a, b)), None)
        if j is not None: m.append((a, B[j])); used.add(j)
        else: ua.append(a)
    ub = [B[j] for j in range(len(B)) if j not in used]
    return m, ua, ub
def strict_prf(cases):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for c in cases:
        tps = _toks(c["raw_text"])
        gold = [CT(e["start"], e["end"], e["type"], _tset(tps, e["start"], e["end"]))
                for e in c["annotations"]["entities"]]
        pred = [CT(e["start"], e["end"], e["type"], _tset(tps, e["start"], e["end"]))
                for e in c["predictions"]["entities"]]
        mm, ug, up = _match(gold, pred, _exact)
        for g, _ in mm: tp[g.type] += 1
        for r in up: fp[r.type] += 1
        for r in ug: fn[r.type] += 1
    out, f1s = {}, []
    for t in ["Premise", "Claim", "MajorClaim"]:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) else 0.0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[t] = {"p": p, "r": r, "f1": f, "tp": tp[t], "fp": fp[t], "fn": fn[t]}
        f1s.append(f)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return out

def build_dev_cases(preds):
    t2p = {text: int(p) for text, p in zip(p1_dev_df["text"], preds)}
    cases = []
    for tpl in dev_eval_template:
        ents, eid = [], 1
        for sr in tpl["sent_records"]:
            lab = INV.get(t2p.get(sr["text"], 0), "None")
            if lab != "None":
                ents.append({"id": f"T{eid}", "text": sr["text"], "start": sr["start"],
                             "end": sr["end"], "type": lab}); eid += 1
        cases.append({"id": tpl["id"], "raw_text": tpl["raw_text"],
                      "annotations": tpl["annotations"],
                      "predictions": {"entities": ents, "relations": []}})
    return cases

METRICS = []
EPOCH = [0]
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    preds[probs[:, 3] > 0.50] = 3  # MajorClaim override (notebook recipe)
    EPOCH[0] += 1
    r = strict_prf(build_dev_cases(preds))
    METRICS.append({
        "model": MODEL_NAME, "seed": SEED, "epoch": EPOCH[0], "macro_f1": r["macro_f1"],
        "Premise_f1": r["Premise"]["f1"], "Claim_f1": r["Claim"]["f1"],
        "MajorClaim_f1": r["MajorClaim"]["f1"],
        "Premise_p": r["Premise"]["p"], "Premise_r": r["Premise"]["r"],
        "Claim_p": r["Claim"]["p"], "Claim_r": r["Claim"]["r"],
        "MajorClaim_p": r["MajorClaim"]["p"], "MajorClaim_r": r["MajorClaim"]["r"],
    })
    print(f"\n--- EPOCH {EPOCH[0]} DEV --- macro={r['macro_f1']:.4f} "
          f"P={r['Premise']['f1']:.4f} C={r['Claim']['f1']:.4f} MC={r['MajorClaim']['f1']:.4f}")
    return {"strict_macro_f1": r["macro_f1"]}

def class_weights(df):
    c = np.bincount(df["label"].values, minlength=4).astype(float)
    base = c.sum() / (len(c) * c)
    w = (base / base.mean()).tolist()
    w[3] = 10.0  # extra weight for rare MajorClaim
    return w

class WTrainer(Trainer):
    def __init__(self, *a, cw=None, **k):
        super().__init__(*a, **k); self.cw = cw
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = torch.tensor(self.cw, dtype=outputs.logits.dtype, device=outputs.logits.device)
        loss = F.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss

def make_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, ignore_mismatched_sizes=True)

def make_optim(model):
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    base = [p for n, p in model.named_parameters() if "classifier" not in n]
    return torch.optim.AdamW(
        [{"params": base, "lr": 1e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.05)

class BestEpoch(TrainerCallback):
    def __init__(self): self.best = 0.0; self.epoch = 1
    def on_evaluate(self, args, state, control, metrics, **kw):
        f = metrics.get("eval_strict_macro_f1", 0.0)
        if f > self.best: self.best = f; self.epoch = round(state.epoch)

class StepHeartbeat(TrainerCallback):
    """Print a heartbeat every N steps so progress is observable (tqdm is off)."""
    def __init__(self, every=25): self.every = every; self.t0 = time.time()
    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time(); print("  [hb] training started", flush=True)
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every == 0:
            print(f"  [hb] step={state.global_step} epoch={state.epoch:.2f} "
                  f"elapsed={time.time()-self.t0:.0f}s", flush=True)

# ===================== PHASE 1 =====================
print("\n" + "=" * 70 + f"\nPHASE 1: SEARCH BEST EPOCH (seed {SEED})\n" + "=" * 70)
m1 = make_model()
W1 = class_weights(p1_train_df)
args1 = TrainingArguments(
    output_dir=f"{OUT}/p1", eval_strategy="epoch", save_strategy="epoch",
    warmup_steps=100, lr_scheduler_type="cosine",
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    num_train_epochs=15, fp16=USE_FP16, load_best_model_at_end=True,
    metric_for_best_model="strict_macro_f1", greater_is_better=True,
    save_total_limit=1, save_only_model=True, report_to="none", disable_tqdm=True,
    label_smoothing_factor=0.15, seed=SEED, data_seed=SEED,
)
tracker = BestEpoch()
tr1 = WTrainer(model=m1, args=args1, train_dataset=p1_train_ds, eval_dataset=p1_dev_ds,
               compute_metrics=compute_metrics,
               callbacks=[tracker, StepHeartbeat(),
                          EarlyStoppingCallback(early_stopping_patience=4)],
               optimizers=(make_optim(m1), None), cw=W1)
tr1.train()
BEST = tracker.epoch
TARGET = BEST + EXTRA_EPOCHS_FOR_FULL_FIT
print(f"\nBEST EPOCH={BEST} (macro={tracker.best:.4f}) -> Phase 2 target={TARGET} epochs")

# dev logits + dev predictions JSON + metrics (best model loaded via load_best_model_at_end)
dev_logits = tr1.predict(p1_dev_ds).predictions
np.save(f"{OUT}/{FILE_PREFIX}_seed{SEED}_dev_logits.npy", dev_logits)
dprobs = F.softmax(torch.tensor(dev_logits), dim=-1).numpy()
dpreds = np.argmax(dprobs, axis=-1); dpreds[dprobs[:, 3] > 0.50] = 3
dev_cases = build_dev_cases(dpreds)
json.dump([{"id": c["id"], "raw_text": c["raw_text"], "predictions": c["predictions"]}
           for c in dev_cases],
          open(f"{OUT}/{FILE_PREFIX}_seed{SEED}_dev_predictions.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
pd.DataFrame(METRICS).to_csv(f"{OUT}/{FILE_PREFIX}_seed{SEED}_metrics.csv", index=False)
final_dev = strict_prf(dev_cases)
json.dump({"model": MODEL_NAME, "seed": SEED, "best_epoch": BEST,
           "best_macro_f1_during_search": tracker.best,
           "best_model_dev": final_dev, "per_epoch": METRICS},
          open(f"{OUT}/{FILE_PREFIX}_seed{SEED}_metrics.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"Saved dev logits + predictions + metrics. Best-model dev macro={final_dev['macro_f1']:.4f}")

del m1, tr1; gc.collect(); torch.cuda.empty_cache()

# ===================== PHASE 2 =====================
print("\n" + "=" * 70 + f"\nPHASE 2: FULL-FIT (train+dev, {TARGET} epochs) + BLIND LOGITS\n" + "=" * 70)
m2 = make_model()
W2 = class_weights(p2_train_df)

class BlindSaver(TrainerCallback):
    """Save blind-test logits only at the FINAL epoch (speed; per-epoch was the killer)."""
    def __init__(self): self.ref = None
    def on_epoch_end(self, args, state, control, **kw):
        ep = round(state.epoch)
        if self.ref is not None and ep >= TARGET:
            bl = self.ref.predict(p2_test_ds).predictions
            np.save(f"{OUT}/{FILE_PREFIX}_seed{SEED}_epoch{ep}_blind_logits.npy", bl)
            print(f"Saved blind logits epoch {ep}")

args2 = TrainingArguments(
    output_dir=f"{OUT}/p2", eval_strategy="no", save_strategy="no",
    warmup_steps=100, lr_scheduler_type="cosine",
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    num_train_epochs=TARGET, fp16=USE_FP16, report_to="none", disable_tqdm=True,
    label_smoothing_factor=0.15, seed=SEED, data_seed=SEED,
)
bs = BlindSaver()
tr2 = WTrainer(model=m2, args=args2, train_dataset=p2_train_ds,
               callbacks=[bs, StepHeartbeat()], optimizers=(make_optim(m2), None), cw=W2)
bs.ref = tr2
tr2.train()
print(f"\nDONE: model={MODEL_NAME} seed={SEED}")
