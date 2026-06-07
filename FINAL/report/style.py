"""Shared visual theme for all GRACE report figures.

One dark, dashboard-style theme (near-black panels, green/teal accent, light text)
so the rebuilt charts read like Modal dashboard tiles and stay consistent across
the DOCX and PPTX. High-DPI raster output for crisp embedding.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---- palette -------------------------------------------------------------
BG = "#0d1117"        # page background (GitHub/Modal dark)
PANEL = "#161b22"     # axes panel
GRID = "#21262d"      # gridlines
FG = "#e6edf3"        # primary text
MUTED = "#8b949e"     # secondary text
ACCENT = "#7ee787"    # Modal-green primary accent
ACCENT2 = "#58a6ff"   # blue secondary
ACCENT3 = "#d2a8ff"   # purple tertiary
WARN = "#f0883e"      # orange (highlight / caveat)
GOLD = "#e3b341"      # gold (winner)

# 9-model categorical palette (greens -> blues -> warm), winner gets GOLD at draw time
MODEL_COLORS = [
    "#7ee787", "#56d364", "#3fb950", "#58a6ff", "#388bfd",
    "#1f6feb", "#d2a8ff", "#bc8cff", "#f0883e",
]
# 3-class palette
CLASS_COLORS = {"Premise": "#58a6ff", "Claim": "#7ee787", "MajorClaim": "#f0883e"}

FOOTER = "GRACE @ IberLEF 2026  ·  Track 1: Evidence Component Detection (ES)"


def _register_fonts():
    """Prefer a clean sans (DejaVu Sans ships with matplotlib); try nicer ones."""
    for name in ("Segoe UI", "Inter", "Helvetica Neue", "Arial"):
        try:
            if any(name == f.name for f in fm.fontManager.ttflist):
                return name
        except Exception:
            pass
    return "DejaVu Sans"


FONT = _register_fonts()


def apply():
    """Install the theme into matplotlib rcParams."""
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.edgecolor": GRID,
        "axes.labelcolor": FG, "axes.titlecolor": FG,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": FG, "font.family": FONT, "font.size": 12,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": FG,
    })


def titled(fig, title, subtitle=None, *, x=0.012, y=0.975):
    """Add a left-aligned title + optional subtitle with visual hierarchy."""
    fig.text(x, y, title, fontsize=19, fontweight="bold", color=FG, ha="left", va="top")
    if subtitle:
        fig.text(x, y - 0.062, subtitle, fontsize=12, color=MUTED, ha="left", va="top")


def footer(fig, x=0.012, y=0.018):
    fig.text(x, y, FOOTER, fontsize=9, color=MUTED, ha="left", va="bottom")
    fig.text(0.988, y, "n=27 runs · 9 models × 3 seeds", fontsize=9, color=MUTED,
             ha="right", va="bottom")


def save(fig, path):
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path
