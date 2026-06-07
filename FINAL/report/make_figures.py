"""Generate all GRACE report figures from report/report_data.json into report/figures/.

Dashboard-style dark theme (see style.py). Every number traces to report_data.json,
which itself traces to the sweep artifacts + official scorer. Run from FINAL/:
    python report/collect_data.py && python report/make_figures.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import style as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
S.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "report_data.json"), encoding="utf-8"))
LB = DATA["leaderboard"]
ENS = DATA["ensemble"]
CURVES = DATA["curves"]
DS = DATA["dataset"]
META = DATA["run_meta"]
ALL9 = ENS["dev_scores"]["all9"]["official_score"]
TOP3 = ENS["dev_scores"]["top3"]["official_score"]
CLASSES = ("Premise", "Claim", "MajorClaim")
MC_TRAIN = DS["entities"]["train"]["MajorClaim"]
MC_DEV = DS["entities"]["dev"]["MajorClaim"]
MC_PCT = MC_TRAIN / sum(DS["entities"]["train"].values()) * 100
NDEV_S, NBLIND_S = DS["dev_sentences"], DS["blind_sentences"]
NBLIND_DOCS = DS["docs"]["blind"]


def fig_leaderboard():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.26, right=0.965, top=0.80, bottom=0.13)
    rows = LB[::-1]  # ascending so best is on top
    names = [r["pretty"] for r in rows]
    means = [r["mean_macro"] for r in rows]
    stds = [r["std"] for r in rows]
    y = np.arange(len(rows))
    colors = [S.GOLD if r is LB[0] else S.MODEL_COLORS[i % len(S.MODEL_COLORS)]
              for i, r in enumerate(rows)]
    ax.barh(y, means, xerr=stds, color=colors, height=0.66,
            error_kw=dict(ecolor=S.MUTED, elinewidth=1.3, capsize=4))
    for yi, m, sd in zip(y, means, stds):
        ax.text(m + sd + 0.006, yi, f"{m:.3f}", va="center", ha="left",
                color=S.FG, fontsize=11, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Dev macro-F1 (strict)")
    ax.set_xlim(0.50, 0.80)
    ax.axvline(ALL9, color=S.ACCENT, ls="--", lw=1.8, zorder=5)
    ax.text(ALL9, len(rows) - 0.35, f"  all-9 ensemble {ALL9:.3f}",
            color=S.ACCENT, fontsize=11, fontweight="bold", va="center")
    S.titled(fig, "Model Leaderboard — Dev Macro-F1",
             "9 Spanish/clinical transformer backbones · mean ± std over seeds 42/100/2026")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "01_leaderboard.png"))


def fig_per_class():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=0.205, right=1.0, top=0.80, bottom=0.12)
    rows = LB  # best first
    mat = np.array([[r["per_class"][c]["mean"] for c in CLASSES] for r in rows])
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.25, vmax=0.95)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, fontsize=12)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["pretty"] for r in rows], fontsize=11)
    for i in range(len(rows)):
        for j in range(len(CLASSES)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if v < 0.62 else "black", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("mean F1", color=S.FG); cb.ax.yaxis.set_tick_params(color=S.MUTED)
    plt.setp(cb.ax.get_yticklabels(), color=S.MUTED)
    ax.set_xticks(np.arange(-.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=S.BG, linewidth=2); ax.tick_params(which="minor", length=0)
    S.titled(fig, "Per-Class F1 by Model",
             f"MajorClaim (only {MC_DEV} dev / {MC_TRAIN} train spans) is the hard, high-variance column")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "02_per_class_f1.png"))


def fig_seed_variance():
    fig, ax = plt.subplots(figsize=(11, 6.0))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.20)
    rows = LB
    x = np.arange(len(rows))
    seed_colors = {42: S.ACCENT2, 100: S.ACCENT, 2026: S.ACCENT3}
    for seed, col in seed_colors.items():
        ys = [r["seed_macro"][str(seed)] for r in rows]
        ax.scatter(x, ys, s=70, color=col, label=f"seed {seed}", zorder=4,
                   edgecolors=S.BG, linewidths=1.0)
    means = [r["mean_macro"] for r in rows]
    ax.scatter(x, means, marker="_", s=900, color=S.FG, linewidths=2.2, label="mean", zorder=3)
    for xi, r in zip(x, rows):
        lo = min(r["seed_macro"].values()); hi = max(r["seed_macro"].values())
        ax.plot([xi, xi], [lo, hi], color=S.MUTED, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([r["pretty"] for r in rows], rotation=32, ha="right", fontsize=10)
    ax.set_ylabel("Dev macro-F1 (strict)"); ax.set_ylim(0.54, 0.78)
    ax.legend(ncol=4, loc="upper right", fontsize=10)
    S.titled(fig, "Seed Stability",
             "Per-seed dev macro-F1 (42 / 100 / 2026); vertical span = sensitivity to initialization")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "03_seed_variance.png"))


def fig_ensemble_gain():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    fig.subplots_adjust(left=0.10, right=0.965, top=0.80, bottom=0.14)
    best_mean = LB[0]["mean_macro"]
    best_seed = max(max(r["seed_macro"].values()) for r in LB)
    labels = ["Best single\n(mean of 3 seeds)", "Best single\n(best seed)",
              "Top-3 ensemble", "All-9 ensemble\n(submitted)"]
    vals = [best_mean, best_seed, TOP3, ALL9]
    colors = [S.MUTED, S.ACCENT2, S.ACCENT3, S.GOLD]
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", color=S.FG, fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Dev macro-F1 (strict)"); ax.set_ylim(0.66, 0.76)
    ax.annotate(f"+{(ALL9-best_mean)*100:.1f} pts vs best-single-mean",
                xy=(3, ALL9), xytext=(1.35, 0.752), color=S.ACCENT, fontsize=11.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=1.6))
    S.titled(fig, "Ensemble vs Single Models",
             f"Averaging {META['n_cells']} softmax distributions ({META['n_models']} models × {META['n_seeds']} seeds) "
             f"lifts dev macro-F1 to {ALL9:.3f}")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "04_ensemble_gain.png"))


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(11, 6.0))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.12)
    top = LB[:5]
    for i, r in enumerate(top):
        pfx = r["prefix"]
        c = CURVES.get(pfx, {}).get("42")
        if not c:
            continue
        ep = [e["epoch"] for e in c["per_epoch"]]
        mac = [e["macro"] for e in c["per_epoch"]]
        col = S.GOLD if i == 0 else S.MODEL_COLORS[i % len(S.MODEL_COLORS)]
        ax.plot(ep, mac, color=col, lw=2.0, marker="o", ms=4, label=r["pretty"])
        be = c["best_epoch"]
        if be in ep:
            ax.scatter([be], [mac[ep.index(be)]], s=150, facecolors="none",
                       edgecolors=col, linewidths=2.2, zorder=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dev macro-F1 (strict)")
    ax.legend(loc="lower right", fontsize=10)
    S.titled(fig, "Training Dynamics (seed 42)",
             "Dev macro-F1 per epoch · circles = best epoch kept (load-best-at-end, ES patience 4)")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "05_training_curves.png"))


def fig_data_overview():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.6))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.74, bottom=0.13, wspace=0.32)
    # left: doc counts
    docs = DS["docs"]
    keys = ["train", "dev", "blind"]
    vals = [docs[k] for k in keys]
    bars = axL.bar(keys, vals, color=[S.ACCENT2, S.ACCENT, S.WARN], width=0.6)
    for b, v in zip(bars, vals):
        axL.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:,}",
                 ha="center", color=S.FG, fontsize=12, fontweight="bold")
    axL.set_ylabel("documents (RCT abstracts)"); axL.set_title("Corpus size", color=S.FG, fontsize=13)
    axL.set_ylim(0, max(vals) * 1.15)
    # right: train class distribution
    ent = DS["entities"]["train"]
    cvals = [ent[c] for c in CLASSES]
    cbars = axR.bar(CLASSES, cvals, color=[S.CLASS_COLORS[c] for c in CLASSES], width=0.6)
    tot = sum(cvals)
    for b, v in zip(cbars, cvals):
        axR.text(b.get_x() + b.get_width() / 2, v + tot * 0.015, f"{v}\n({v/tot*100:.1f}%)",
                 ha="center", color=S.FG, fontsize=11, fontweight="bold")
    axR.set_ylabel("annotated components (train)")
    axR.set_title("Class imbalance", color=S.FG, fontsize=13)
    axR.set_ylim(0, max(cvals) * 1.2)
    S.titled(fig, "Dataset & Class Imbalance",
             f"MajorClaim = {MC_PCT:.1f}% of train components ({MC_DEV} dev spans) → weighted CE (w=10) + 0.50 decision override")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "06_data_overview.png"))


def _tile(ax, x, y, w, h, value, label, accent):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.2, edgecolor=S.GRID, facecolor=S.PANEL, zorder=2)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x, y), 0.012, h, boxstyle="round,pad=0,rounding_size=0.0",
                                linewidth=0, facecolor=accent, zorder=3))
    ax.text(x + w / 2, y + h * 0.60, value, ha="center", va="center",
            color=S.FG, fontsize=25, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.20, label, ha="center", va="center",
            color=S.MUTED, fontsize=11, zorder=4)


def fig_run_panel():
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.06)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    blind = ENS["blind"]["all9"]
    nc = META["n_cells"]
    tiles = [
        (f"{nc}", "training runs", S.ACCENT),
        (f"{META['n_models']} × {META['n_seeds']}", "models × seeds", S.ACCENT2),
        ("A10G", "GPU (Ampere, bf16)", S.ACCENT3),
        (f"{ALL9:.3f}", "dev macro-F1 (all-9)", S.GOLD),
        (f"{blind['entities']:,}", "blind components", S.ACCENT),
        (f"{NBLIND_DOCS:,}", "blind abstracts", S.ACCENT2),
        (f"{NDEV_S} / {NBLIND_S/1000:.1f}k", "dev / blind sentences", S.ACCENT3),
        ("100%", f"runs converged ({nc}/{nc})", S.GOLD),
    ]
    cols, rows = 4, 2
    pad = 0.018; w = (1 - pad * (cols + 1)) / cols; h = (0.86 - pad * (rows + 1)) / rows
    for i, (val, lab, acc) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = 0.86 - (pad + (r + 1) * h + r * pad)
        _tile(ax, x, y, w, h, val, lab, acc)
    S.titled(fig, "GPU Sweep at a Glance",
             "Full 9×3 grid trained in parallel on Modal serverless GPU · single launch, two-stage recipe")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "07_run_panel.png"))


def fig_blind_distribution():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    fig.subplots_adjust(left=0.10, right=0.965, top=0.80, bottom=0.12)
    blind = ENS["blind"]["all9"]["by_type"]
    train = DS["entities"]["train"]
    bt = sum(blind.values()); tt = sum(train.values())
    x = np.arange(len(CLASSES)); w = 0.38
    bvals = [blind[c] / bt * 100 for c in CLASSES]
    tvals = [train[c] / tt * 100 for c in CLASSES]
    from matplotlib.patches import Patch
    ax.bar(x - w / 2, tvals, w, color=S.MUTED)
    ax.bar(x + w / 2, bvals, w, color=[S.CLASS_COLORS[c] for c in CLASSES])
    for xi, c in zip(x, CLASSES):
        ax.text(xi + w / 2, blind[c] / bt * 100 + 1.2, f"{blind[c]:,}", ha="center",
                color=S.FG, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=12)
    ax.set_ylabel("share of components (%)"); ax.set_ylim(0, 80)
    ax.legend(handles=[Patch(facecolor=S.MUTED, label="train (gold) %"),
                       Patch(facecolor=S.ACCENT, label="blind, predicted % (bars coloured by class)")],
              loc="upper right", fontsize=10)
    S.titled(fig, "Blind-Test Predictions (all-9 ensemble)",
             f"{NBLIND_DOCS:,} abstracts → {ENS['blind']['all9']['entities']:,} components · majority classes track the "
             f"train prior, MajorClaim predicted conservatively")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "08_blind_distribution.png"))


def _rbox(ax, x, y, w, h, fill=S.PANEL, edge=S.GRID, lw=1.2, rounding=0.025):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={rounding}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def _arrow(ax, x1, y1, x2, y2, color=S.MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, shrinkA=2, shrinkB=2), zorder=4)


def fig_annotated_example():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.07)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        ("MajorClaim", "El impacto del hipertiroidismo subclínico prolongado en la calidad de vida no está claro.",
         "the abstract's overall conclusion", S.CLASS_COLORS["MajorClaim"]),
        ("Claim", "El resultado de estos pacientes sigue siendo insatisfactorio.",
         "interpretive statement about the findings", S.CLASS_COLORS["Claim"]),
        ("Premise", "La calidad de vida fue similar en los dos grupos.",
         "reported evidence / measured outcome", S.CLASS_COLORS["Premise"]),
        ("None", "Los pacientes fueron asignados aleatoriamente a dos grupos de tratamiento.",
         "non-argumentative text (background / methods)", S.MUTED),
    ]
    h = 0.185; gap = 0.038; y = 0.95
    for name, text, desc, col in rows:
        _rbox(ax, 0.03, y - h, 0.94, h)
        ax.add_patch(FancyBboxPatch((0.03, y - h), 0.012, h, boxstyle="round,pad=0,rounding_size=0",
                                    linewidth=0, facecolor=col, zorder=3))
        ax.text(0.055, y - 0.040, name, color=col, fontsize=15, fontweight="bold", va="top")
        ax.text(0.055, y - 0.103, f"“{text}”", color=S.FG, fontsize=12.5, va="top", style="italic")
        ax.text(0.055, y - 0.158, desc, color=S.MUTED, fontsize=10.5, va="top")
        y -= (h + gap)
    S.titled(fig, "Sentence-Level Labelling",
             "Each segment of a Spanish RCT abstract is assigned one of four roles (real dev examples)")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "09_annotated_example.png"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.04)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # input column
    _rbox(ax, 0.02, 0.46, 0.16, 0.20, fill=S.PANEL, edge=S.ACCENT2)
    ax.text(0.10, 0.56, "RCT abstract", color=S.FG, fontsize=12, fontweight="bold", ha="center")
    ax.text(0.10, 0.50, "spaCy → sentences\nprev [SEP] sent", color=S.MUTED, fontsize=9.5, ha="center", va="center")
    # stage 1
    _rbox(ax, 0.26, 0.58, 0.26, 0.26, fill=S.PANEL, edge=S.ACCENT)
    ax.text(0.39, 0.79, "STAGE 1 — search", color=S.ACCENT, fontsize=12.5, fontweight="bold", ha="center", va="top")
    ax.text(0.39, 0.725, "train on train\neval strict-F1 on dev / epoch\nkeep best epoch", color=S.FG,
            fontsize=10, ha="center", va="top")
    _rbox(ax, 0.585, 0.60, 0.16, 0.12, fill="#10261c", edge=S.ACCENT)
    ax.text(0.665, 0.66, f"dev logits\n({NDEV_S} × 4)", color=S.ACCENT, fontsize=10, ha="center", va="center", fontweight="bold")
    # stage 2
    _rbox(ax, 0.26, 0.16, 0.26, 0.26, fill=S.PANEL, edge=S.ACCENT3)
    ax.text(0.39, 0.37, "STAGE 2 — full-fit", color=S.ACCENT3, fontsize=12.5, fontweight="bold", ha="center", va="top")
    ax.text(0.39, 0.305, "refit on train + dev\n(best_epoch + 1)\npredict blind test", color=S.FG,
            fontsize=10, ha="center", va="top")
    _rbox(ax, 0.585, 0.22, 0.16, 0.12, fill="#241a30", edge=S.ACCENT3)
    ax.text(0.665, 0.28, f"blind logits\n({NBLIND_S/1000:.1f}k × 4)", color=S.ACCENT3, fontsize=10, ha="center", va="center", fontweight="bold")
    # ensemble + outputs
    _rbox(ax, 0.79, 0.41, 0.19, 0.18, fill=S.PANEL, edge=S.GOLD)
    ax.text(0.885, 0.535, "ENSEMBLE", color=S.GOLD, fontsize=12.5, fontweight="bold", ha="center", va="top")
    ax.text(0.885, 0.475, "average 27 dists\n+ 0.50 MC override", color=S.FG, fontsize=9.5, ha="center", va="top")
    # arrows
    _arrow(ax, 0.18, 0.62, 0.26, 0.68, S.ACCENT2)
    _arrow(ax, 0.18, 0.52, 0.26, 0.30, S.ACCENT2)
    _arrow(ax, 0.52, 0.69, 0.585, 0.66, S.ACCENT)
    _arrow(ax, 0.52, 0.28, 0.585, 0.28, S.ACCENT3)
    _arrow(ax, 0.745, 0.64, 0.83, 0.57, S.ACCENT)      # dev logits -> ensemble (dev score)
    _arrow(ax, 0.745, 0.28, 0.83, 0.45, S.ACCENT3)     # blind logits -> ensemble (submission)
    ax.text(0.80, 0.66, "dev macro-F1", color=S.ACCENT, fontsize=9, ha="center", style="italic")
    ax.text(0.80, 0.34, "submission", color=S.ACCENT3, fontsize=9, ha="center", style="italic")
    ax.text(0.39, 0.50, "×9 models × 3 seeds = 27 runs, in parallel", color=S.MUTED, fontsize=10,
            ha="center", style="italic")
    S.titled(fig, "System Architecture — Two-Stage Auto-Runner",
             "Held-out model selection (Stage 1) is cleanly separated from the full-data fit used for submission (Stage 2)")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "10_pipeline.png"))


def fig_eval_methodology():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.06)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    toks = ["La", "calidad", "de", "vida", "fue", "similar", "en", "ambos", "grupos"]
    n = len(toks); x0 = 0.06; tw = 0.86 / n
    # strict panel
    ax.text(0.06, 0.86, "Strict match  —  exact start & end offsets", color=S.ACCENT, fontsize=13, fontweight="bold", va="top")
    for i, t in enumerate(toks):
        x = x0 + i * tw
        gold = 1 <= i <= 8
        pred = 1 <= i <= 8
        _rbox(ax, x, 0.66, tw * 0.92, 0.08, fill="#10261c" if gold else S.PANEL, edge=S.GRID, lw=0.8, rounding=0.06)
        ax.text(x + tw * 0.46, 0.70, t, color=S.FG if gold else S.MUTED, fontsize=9, ha="center", va="center")
    ax.text(0.02, 0.70, "", fontsize=8)
    ax.annotate("gold = pred → TP", xy=(0.94, 0.70), fontsize=10.5, color=S.ACCENT, ha="left", va="center")
    # relaxed panel
    ax.text(0.06, 0.50, "Relaxed match  —  token Jaccard (IoU) ≥ τ", color=S.ACCENT2, fontsize=13, fontweight="bold", va="top")
    for i, t in enumerate(toks):
        x = x0 + i * tw
        gold = 1 <= i <= 8
        pred = 1 <= i <= 6
        fill = S.PANEL
        edge = S.GRID
        if gold and pred:
            fill = "#0e2030"; edge = S.ACCENT2
        elif gold:
            edge = S.MUTED
        _rbox(ax, x, 0.30, tw * 0.92, 0.08, fill=fill, edge=edge, lw=1.0 if gold else 0.8, rounding=0.06)
        ax.text(x + tw * 0.46, 0.34, t, color=S.FG if (gold or pred) else S.MUTED, fontsize=9, ha="center", va="center")
    ax.annotate("overlap / union ≥ τ\n→ still a match", xy=(0.94, 0.34), fontsize=10.5, color=S.ACCENT2, ha="left", va="center")
    # footer note
    _rbox(ax, 0.06, 0.07, 0.88, 0.13, fill=S.PANEL)
    ax.text(0.5, 0.135, "Official ranking metric:  macro-F1 (strict)  =  mean of per-class F1 over {Premise, Claim, MajorClaim}",
            color=S.FG, fontsize=12, ha="center", va="center", fontweight="bold")
    ax.text(0.5, 0.092, "Macro (not micro) averaging gives the rare MajorClaim class equal weight to Premise.",
            color=S.MUTED, fontsize=10, ha="center", va="center")
    S.titled(fig, "Evaluation Methodology",
             "How predicted spans are matched to gold annotations, and how the score is aggregated")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "11_eval_methodology.png"))


def fig_engineering_journey():
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.05)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0.08, 0.92], [0.62, 0.62], color=S.GRID, lw=3, zorder=1)
    stops = [
        (0.20, "Kaggle", S.WARN, "1", "WORKS, BUT CAPPED",
         "T4 free tier; torch-2.10 P100 hang (3 h) fixed by\nforcing T4 + GPU guard. Serial & quota-limited.", False),
        (0.50, "Azure", "#f85149", "2", "BLOCKED",
         "Student subscription GPU quota = 0\n(NCASv3_T4). Cannot be raised → abandoned.", False),
        (0.80, "Modal", S.ACCENT, "3", "WINNER",
         "Serverless A10G + bf16. All 27 runs in ONE\nparallel launch. Canonical platform.", True),
    ]
    for x, name, col, num, status, desc, win in stops:
        ax.scatter([x], [0.62], s=520, color=col, edgecolors=S.BG, linewidths=2.5, zorder=3)
        ax.text(x, 0.62, num, color=S.BG, fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
        _rbox(ax, x - 0.15, 0.20, 0.30, 0.26, fill=S.PANEL, edge=col, lw=1.8 if win else 1.2)
        ax.text(x, 0.42, name, color=col, fontsize=15, fontweight="bold", ha="center", va="top")
        ax.text(x, 0.355, desc, color=S.FG, fontsize=9.3, ha="center", va="top")
        ax.text(x, 0.71, status, color=col, fontsize=12, fontweight="bold", ha="center")
    S.titled(fig, "Infrastructure — The Path to a Working Sweep",
             "Three platforms, two dead-ends, one win: honest engineering on a tight GPU budget")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "12_engineering_journey.png"))


def fig_ensemble_method():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.05)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # grid of 27 mini distributions
    ax.text(0.16, 0.86, "27 softmax distributions", color=S.FG, fontsize=12, fontweight="bold", ha="center", va="top")
    cols, rows = 9, 3
    gx0, gy0, cw, ch = 0.03, 0.30, 0.028, 0.10
    rng = [0.6, 0.25, 0.1, 0.05]
    for r in range(rows):
        for c in range(cols):
            x = gx0 + c * (cw + 0.005); y = gy0 + r * (ch + 0.06)
            base = [0.55 + 0.03 * ((c + r) % 4), 0.25, 0.12, 0.08]
            for k, v in enumerate(base):
                _rbox(ax, x, y + k * (ch / 4) * 0.0, 0, 0)  # noop keep spacing
            # draw 4 tiny bars
            for k, v in enumerate(base):
                bw = cw / 4
                S_h = v * ch
                ax.add_patch(plt.Rectangle((x + k * bw, y), bw * 0.8, S_h,
                             color=[S.MUTED, S.ACCENT2, S.ACCENT, S.WARN][k]))
    ax.text(0.16, 0.20, "9 models × 3 seeds", color=S.MUTED, fontsize=10, ha="center", style="italic")
    _arrow(ax, 0.32, 0.45, 0.40, 0.45, S.GOLD)
    ax.text(0.36, 0.50, "mean", color=S.GOLD, fontsize=11, ha="center", fontweight="bold")
    # averaged distribution
    avg = [0.56, 0.27, 0.11, 0.06]
    bx = 0.44
    for k, (v, lab, col) in enumerate(zip(avg, ["None", "Prem", "Claim", "Major"],
                                          [S.MUTED, S.ACCENT2, S.ACCENT, S.WARN])):
        ax.add_patch(plt.Rectangle((bx + k * 0.035, 0.34), 0.028, v * 0.34, color=col))
        ax.text(bx + k * 0.035 + 0.014, 0.32, lab, color=S.MUTED, fontsize=8, ha="center", va="top")
    ax.text(bx + 0.07, 0.74, "averaged\ndistribution", color=S.FG, fontsize=11, ha="center", fontweight="bold")
    _arrow(ax, 0.60, 0.45, 0.68, 0.45, S.GOLD)
    ax.text(0.64, 0.50, "argmax\n+0.50 MC", color=S.GOLD, fontsize=9.5, ha="center", fontweight="bold")
    _rbox(ax, 0.71, 0.36, 0.26, 0.20, fill=S.PANEL, edge=S.GOLD)
    ax.text(0.84, 0.50, "all-9 ensemble", color=S.GOLD, fontsize=13, fontweight="bold", ha="center", va="top")
    ax.text(0.84, 0.44, f"dev strict macro-F1\n{ALL9:.4f}", color=S.FG, fontsize=12, ha="center", va="top")
    S.titled(fig, "Ensemble Methodology — Probability Averaging",
             "Mean over all 27 softmax outputs, then argmax with a 0.50 MajorClaim override — no training, no dev peeking")
    S.footer(fig)
    return S.save(fig, os.path.join(FIG, "13_ensemble_method.png"))


def main():
    figs = [fig_leaderboard(), fig_per_class(), fig_seed_variance(), fig_ensemble_gain(),
            fig_training_curves(), fig_data_overview(), fig_run_panel(), fig_blind_distribution(),
            fig_annotated_example(), fig_pipeline(), fig_eval_methodology(),
            fig_engineering_journey(), fig_ensemble_method()]
    for f in figs:
        print("wrote", os.path.relpath(f, HERE))
    print(f"\n{len(figs)} figures -> {os.path.relpath(FIG, HERE)}/")


if __name__ == "__main__":
    main()
