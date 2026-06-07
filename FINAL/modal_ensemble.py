"""GRACE Track-1 ensemble on Modal — combines the 27 saved logit sets.

Reuses the EXACT parse/template/scorer logic from modal_sweep.py so the
row->sentence->entity alignment is identical to training. Gates:
  - assert every dev_logits is (N_dev, 4)
  - reproduce a single model's known per-seed F1 from its own logits
    through the ensemble code path (alignment proof)
Reports BOTH greedy-selected (dev-optimistic) AND a-priori ensembles
(all-9, top-3-by-mean), scores the chosen one with the OFFICIAL GRACE
scorer, builds + validates the blind-test submission JSON.

Run: modal run modal_ensemble.py
"""
from __future__ import annotations
import modal

app = modal.App("grace-ensemble")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "spacy", "click>=8.1", "scikit-learn", "pandas")
    .run_commands("python -m spacy download es_core_news_sm")
    .add_local_dir("IberLEF_GRACE", "/data/IberLEF_GRACE", copy=True)
    .add_local_dir("GRACE_2026_blind_test", "/data/GRACE_2026_blind_test", copy=True)
)
VOL = modal.Volume.from_name("grace-out")
REPORT = modal.Volume.from_name("grace-report", create_if_missing=True)

PREFIXES = ["mrbert_biomed", "bsc_bio_ehr", "roberta_clinical", "mdeberta_v3",
            "rigoberta_clinical", "mrbert_es", "xlmr_large", "beto", "beto_galen"]
PRETTY = {"mrbert_biomed": "MrBERT-biomed", "bsc_bio_ehr": "bsc-bio-ehr-es",
          "roberta_clinical": "roberta-clinical-es", "mdeberta_v3": "mdeberta-v3-base",
          "rigoberta_clinical": "RigoBERTa-Clinical", "mrbert_es": "MrBERT-es",
          "xlmr_large": "xlm-roberta-large", "beto": "BETO", "beto_galen": "BETO_Galen"}
SEEDS = [42, 100, 2026]


@app.function(image=image, volumes={"/vol": VOL, "/rep": REPORT}, timeout=1800)
def ensemble() -> dict:
    import json, re, glob, importlib.util
    import numpy as np, spacy
    from collections import defaultdict
    from typing import NamedTuple

    LABEL_MAP = {"None": 0, "Premise": 1, "Claim": 2, "MajorClaim": 3}
    INV = {v: k for k, v in LABEL_MAP.items()}
    nlp = spacy.load("es_core_news_sm")

    def find(name):
        return sorted(glob.glob(f"/data/**/{name}", recursive=True), key=len)[0]
    DEV, BLIND = find("track_1_dev.json"), find("track_1_blind_test.json")

    # ---- dev sentence order (must match training parse) ----
    def parse_dev_texts(path):
        data = json.load(open(path, encoding="utf-8")); texts = []
        for doc in data:
            rt = doc["raw_text"]; ann = doc.get("annotations", {}).get("entities", [])
            spans = [(a["start"], a["end"]) for a in ann]
            sr = [{"text": a["text"].strip(), "start": a["start"]} for a in ann]
            for s in nlp(rt).sents:
                if not any(max(s.start_char, a) < min(s.end_char, b) for a, b in spans) and len(s.text.strip()) > 10:
                    sr.append({"text": s.text.strip(), "start": s.start_char})
            sr.sort(key=lambda x: x["start"])
            texts.extend([r["text"] for r in sr])
        return texts
    dev_texts = parse_dev_texts(DEV)
    N_DEV = len(dev_texts)

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

    def probs_to_cases(prob):
        pred = np.argmax(prob, axis=-1); pred[prob[:, 3] > 0.50] = 3
        t2p = {t: int(p) for t, p in zip(dev_texts, pred)}
        cs = []
        for tp in templ:
            ents, eid = [], 1
            for sr in tp["sent_records"]:
                lab = INV.get(t2p.get(sr["text"], 0), "None")
                if lab != "None":
                    ents.append({"id": f"T{eid}", "text": sr["text"], "start": sr["start"], "end": sr["end"], "type": lab}); eid += 1
            cs.append({"id": tp["id"], "raw_text": tp["raw_text"], "annotations": tp["annotations"], "predictions": {"entities": ents, "relations": []}})
        return cs

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

    # ---- load dev logits, assert shape ----
    dev_prob = {}  # prefix -> mean-over-seeds prob (N_DEV,4)
    per_seed_f1 = {}
    for pfx in PREFIXES:
        seed_probs = []
        for s in SEEDS:
            arr = np.load(f"/vol/{pfx}_seed{s}_dev_logits.npy")
            assert arr.shape == (N_DEV, 4), f"{pfx}_seed{s} shape {arr.shape} != ({N_DEV},4)"
            pr = softmax(arr)
            seed_probs.append(pr)
            per_seed_f1[f"{pfx}_s{s}"] = strict_prf(probs_to_cases(pr))["macro_f1"]
        dev_prob[pfx] = np.mean(seed_probs, axis=0)

    # ---- GATE: reproduce a known single-model/seed F1 from its logits ----
    gate = per_seed_f1["mdeberta_v3_s42"]
    print(f"[GATE] mdeberta_v3 s42 reproduced macro={gate:.4f} (expected ~0.7368)", flush=True)
    assert abs(gate - 0.7368) < 0.02, f"alignment gate FAILED: {gate}"

    per_model_mean = {pfx: float(np.mean([per_seed_f1[f"{pfx}_s{s}"] for s in SEEDS])) for pfx in PREFIXES}

    def ens_f1(members):
        prob = np.mean([dev_prob[m] for m in members], axis=0)
        return strict_prf(probs_to_cases(prob))["macro_f1"], prob

    # ---- greedy forward selection (dev-optimistic) ----
    remaining = sorted(PREFIXES, key=lambda m: -per_model_mean[m])
    chosen, best_f1 = [], 0.0
    while remaining:
        cand = [(ens_f1(chosen + [m])[0], m) for m in remaining]
        cand.sort(reverse=True)
        if cand[0][0] <= best_f1 + 1e-9:
            break
        best_f1, pick = cand[0]
        chosen.append(pick); remaining.remove(pick)
    greedy_f1, _ = ens_f1(chosen)

    # ---- candidate ensembles ----
    all9 = list(PREFIXES)
    top3 = sorted(PREFIXES, key=lambda m: -per_model_mean[m])[:3]
    candidates = {"greedy": chosen, "top3": top3, "all9": all9}

    # ---- OFFICIAL scorer (loaded once); score every candidate honestly ----
    spec = importlib.util.spec_from_file_location("scorer", find("track1_scoring_program.py"))
    scr = importlib.util.module_from_spec(spec); spec.loader.exec_module(scr)

    def score_dev(members):
        _, prob = ens_f1(members)
        cases = probs_to_cases(prob)
        off = scr.evaluate_subtask1(cases)
        return {"members": members,
                "dev_macro_f1": float(strict_prf(cases)["macro_f1"]),
                "official_score": off["official_score"],
                "strict_macro": off["strict"]["macro_avg"],
                "relaxed_macro": off["relaxed"]["macro_avg"],
                "strict_per_type": {k: off["strict"]["per_type"][k] for k in off["strict"]["per_type"]}}

    dev_scores = {name: score_dev(m) for name, m in candidates.items()}
    for name, sc in dev_scores.items():
        print(f"[OFFICIAL] {name:6s} ({len(sc['members'])}) strict-macroF1={sc['official_score']:.4f} "
              f"relaxed={sc['relaxed_macro']['f1']:.4f}", flush=True)
    # save the all-9 (zero-peek) dev prediction file for the report
    _, ref_prob = ens_f1(all9)
    json.dump([{"id": c["id"], "raw_text": c["raw_text"], "predictions": c["predictions"]}
               for c in probs_to_cases(ref_prob)],
              open("/rep/ensemble_dev_predictions_all9.json", "w", encoding="utf-8"), ensure_ascii=False)

    # ---- blind: parse once, load each model's blind prob once (mean over seeds) ----
    def parse_blind_docs(path):
        data = json.load(open(path, encoding="utf-8")); flat = []; docs = []
        for doc in data:
            rt = doc["raw_text"]; sr = []
            for s in nlp(rt).sents:
                txt = s.text
                if len(txt.strip()) <= 10:
                    continue
                # tight offsets so raw_text[start:end] == stripped text (spaCy spans
                # include trailing/leading whitespace); logit row order is unchanged.
                start = s.start_char + (len(txt) - len(txt.lstrip()))
                end = start + len(txt.strip())
                sr.append({"text": txt.strip(), "start": start, "end": end})
            sr.sort(key=lambda x: x["start"])
            docs.append({"id": doc["id"], "raw_text": rt, "sent": sr})
            flat.extend([(doc["id"], r["text"], r["start"], r["end"]) for r in sr])
        return docs, flat
    bdocs, bflat = parse_blind_docs(BLIND)
    blind_prob_by_model = {}
    for m in PREFIXES:
        seedp = []
        for s in SEEDS:
            matches = sorted(glob.glob(f"/vol/{m}_seed{s}_epoch*_blind_logits.npy"))
            assert len(matches) == 1, f"expected 1 blind file for {m}_seed{s}, found {matches}"
            seedp.append(softmax(np.load(matches[0])))
        bp = np.mean(seedp, axis=0)
        assert bp.shape[0] == len(bflat), f"blind rows {bp.shape[0]} != sentences {len(bflat)} for {m}"
        blind_prob_by_model[m] = bp

    def build_blind(members, tag):
        bprob = np.mean([blind_prob_by_model[m] for m in members], axis=0)
        bpred = np.argmax(bprob, axis=-1); bpred[bprob[:, 3] > 0.50] = 3
        by_doc = defaultdict(list)
        for (did, text, st, en), p in zip(bflat, bpred):
            lab = INV.get(int(p), "None")
            if lab != "None":
                by_doc[did].append({"text": text, "start": st, "end": en, "type": lab})
        sub = []
        for d in bdocs:
            ents = [{"id": f"T{i+1}", **e} for i, e in enumerate(by_doc.get(d["id"], []))]
            sub.append({"id": d["id"], "raw_text": d["raw_text"], "predictions": {"entities": ents, "relations": []}})
        bad = sum(1 for c in sub for e in c["predictions"]["entities"]
                  if c["raw_text"][e["start"]:e["end"]] != e["text"])
        json.dump(sub, open(f"/rep/blind_submission_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False)
        by_type = defaultdict(int)
        for c in sub:
            for e in c["predictions"]["entities"]:
                by_type[e["type"]] += 1
        n_ents = sum(len(c["predictions"]["entities"]) for c in sub)
        print(f"[SUBMISSION:{tag}] {len(sub)} docs, {n_ents} ents {dict(by_type)}, {bad} offset-mismatches", flush=True)
        return {"docs": len(sub), "entities": n_ents, "by_type": dict(by_type),
                "offset_mismatches": bad, "file": f"blind_submission_{tag}.json"}

    blind_stats = {name: build_blind(m, name) for name, m in candidates.items()}

    results = {
        "per_seed_f1": per_seed_f1, "per_model_mean": per_model_mean,
        "dev_scores": dev_scores, "blind": blind_stats,
        "n_blind_sentences": len(bflat),
        "pretty": PRETTY, "n_dev_sentences": N_DEV,
    }
    json.dump(results, open("/rep/ensemble_results.json", "w", encoding="utf-8"), indent=2)
    REPORT.commit()
    return results


@app.local_entrypoint()
def main():
    import json
    r = ensemble.remote()
    print("\n================ ENSEMBLE RESULTS ================")
    bs = max(r["per_model_mean"].items(), key=lambda kv: kv[1])
    print(f"best single : {bs[0]} = {bs[1]:.4f}")
    for name, sc in r["dev_scores"].items():
        print(f"{name:6s} ({len(sc['members'])}) dev strict-macroF1={sc['official_score']:.4f} "
              f"relaxed={sc['relaxed_macro']['f1']:.4f}  members={sc['members']}")
    print("blind:", json.dumps(r["blind"], indent=2))
    json.dump(r, open("ensemble_results.json", "w"), indent=2)
    print("saved -> ensemble_results.json ; submissions in volume grace-report")
