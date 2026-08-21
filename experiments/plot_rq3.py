"""
plot_rq3.py — Figure RQ3 : closure strategies vs baseline
Reads experiments/results_rq3_final.json, produces rq3_closures.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_rq3_final.json"
OUT  = ROOT / "figures/rq3_closures.pdf"

# ── Palette Klimt (cohérente avec la thèse) ──────────────────────────────────
COLORS = {
    "none":         "#7A7A7A",   # gris neutre
    "motivational": "#8B1A1A",   # PadovaRed
    "alphabet":     "#C8970A",   # GoldDark
    "mot+socratic": "#3B5998",   # bleu profond
    "contrast":     "#4A7A4A",   # vert
}
LABELS = {
    "none":         "No closure",
    "motivational": "Motivational",
    "alphabet":     "Alphabet",
    "mot+socratic": "Mot.+Socratic",
    "contrast":     "Contrast",
}

QUERY_LABELS = {
    "T1q5":  "T1q5\nTuring Award\n(GT=77)",
    "T2q2":  "T2q2\nFilms by directors\n(GT=58)",
    "T2q5":  "T2q5\nPop. > 10M\n(GT=90)",
    "T5q2":  "T5q2\nPapal predecessors\n(GT=264)",
    "T5q10": "T5q10\nBaseException\n(GT=64)",
}

CLOSURE_ORDER = ["none", "motivational", "alphabet", "mot+socratic", "contrast"]


def load():
    return json.loads(DATA.read_text())


def make_figure(data: dict):
    queries = [q for q in ["T1q5","T2q2","T2q5","T5q2","T5q10"] if q in data]
    # exclude T5q10 from main figure (too noisy) but keep for annotation
    queries_main = [q for q in queries if q != "T5q10"]

    n_q = len(queries_main)
    n_c = len(CLOSURE_ORDER)
    x   = np.arange(n_q)
    width = 0.13
    offsets = np.linspace(-(n_c-1)/2, (n_c-1)/2, n_c) * width

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "RQ3 — Effect of Closure Strategies on Plateau Queries\n"
        r"$\Delta$ relative to no-closure baseline",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── Panel 1 : F1 absolute ─────────────────────────────────────────────────
    ax1 = axes[0]
    for ci, closure in enumerate(CLOSURE_ORDER):
        f1s  = []
        errs = []
        for q in queries_main:
            s = data[q].get(closure, {})
            f1s.append(s.get("F1_mean", 0))
            errs.append(s.get("F1_std",  0))
        bars = ax1.bar(x + offsets[ci], f1s, width,
                       yerr=errs, capsize=3,
                       color=COLORS[closure], alpha=0.88,
                       label=LABELS[closure],
                       error_kw={"elinewidth": 1.2, "ecolor": "#333"})

    # baseline dashed line per query
    for qi, q in enumerate(queries_main):
        base = data[q]["none"]["F1_mean"]
        ax1.hlines(base, x[qi]-0.38, x[qi]+0.38,
                   colors=COLORS["none"], linewidths=1.2,
                   linestyles="--", alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([QUERY_LABELS[q] for q in queries_main],
                        fontsize=8.5)
    ax1.set_ylabel("F1 (mean ± std, N=3)", fontsize=10)
    ax1.set_title("(a) F1 score", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")

    # ── Panel 2 : Recall absolu toutes closures ───────────────────────────────
    ax2 = axes[1]
    for ci, closure in enumerate(CLOSURE_ORDER):
        rs   = []
        errs = []
        for q in queries_main:
            s = data[q].get(closure, {})
            rs.append(s.get("R_mean", 0))
            errs.append(s.get("R_std",  0))
        ax2.bar(x + offsets[ci], rs, width,
                yerr=errs, capsize=3,
                color=COLORS[closure], alpha=0.88,
                label=LABELS[closure],
                error_kw={"elinewidth": 1.2, "ecolor": "#333"})

    # baseline dashed line per query
    for qi, q in enumerate(queries_main):
        base = data[q]["none"]["R_mean"]
        ax2.hlines(base, x[qi]-0.38, x[qi]+0.38,
                   colors=COLORS["none"], linewidths=1.2,
                   linestyles="--", alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels([QUERY_LABELS[q] for q in queries_main],
                        fontsize=8.5)
    ax2.set_ylabel("Recall (mean ± std, N=3)", fontsize=10)
    ax2.set_title("(b) Recall score", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, ncol=2, loc="upper right")

    # ── Annotation T5q10 note ────────────────────────────────────────────────
    fig.text(0.5, -0.04,
             "T5q10 (BaseException, GT=64) excluded: high variance (std>0.4) due to "
             "LLMRecScan instability on DAG patterns. "
             "Dashed lines in (a) indicate the no-closure F1 baseline per query.",
             ha="center", fontsize=7.5, color="#555",
             style="italic", wrap=True)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    print(f"  Saved → {OUT}")

    # ── Also save PNG for quick preview ──────────────────────────────────────
    png = OUT.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"  Saved → {png}")
    plt.close(fig)


if __name__ == "__main__":
    if not DATA.exists():
        print(f"[error] {DATA} not found — run rq3_final.py + rq3_extra.py first")
        sys.exit(1)
    data = load()
    make_figure(data)
    print("Done.")
