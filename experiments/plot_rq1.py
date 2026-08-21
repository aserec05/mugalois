"""
experiments/plot_rq1.py
=======================
Figure RQ1 — F1 scatter plot across representative queries,
ordered by increasing complexity (left = simple, right = complex).

Reads from experiments/results/ (or --results-dir).
Raises an error if a required query is missing.

Usage
-----
    python3 experiments/plot_rq1.py
    python3 experiments/plot_rq1.py --results-dir experiments/results --out figures/rq1_scatter
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Representative queries (ordered simple → complex) ────────────────
# (label, template, query_id, x_position)
QUERIES = [
    ("T1·q1\nEinstein\nspouse",      "T1", "T1_q1",  1),
    ("T1·q4\nCurie\nawards",         "T1", "T1_q4",  2),
    ("T1·q5\nTuring\nwinners",       "T1", "T1_q5",  3),
    ("T2·q1\nCapitals",              "T2", "T2_q1",  5),
    ("T2·q2\nKubrick\nfilms",        "T2", "T2_q2",  6),
    ("T2·q5\nPop.\n>10M",            "T2", "T2_q5",  7),
    ("T3·q1\nNolan\nthriller",       "T3", "T3_q1",  9),
    ("T3·q2\nDC\nPMs",               "T3", "T3_q2", 10),
    ("T4·q1\nWilde\ncomedy",         "T4", "T4_q1", 12),
    ("T4·q3\nShakespeare\nItaly",    "T4", "T4_q3", 13),
    ("T4·q5\nNobel\nAsia",           "T4", "T4_q5", 14),
    ("T5·q1\nPius XII\n+fwd",        "T5", "T5_q1", 16),
    ("T5·q7\nMartin V\n49-hop",      "T5", "T5_q7", 17),
    ("T5·q9\nEliz. II\nDAG",         "T5", "T5_q9", 18),
    ("T5·q10\nBaseExc.\nDAG",        "T5", "T5_q10",19),
    ("T6·q1\nEinstein\n|sibling",    "T6", "T6_q1", 21),
    ("T6·q4\nNolan\ndirected|",      "T6", "T6_q4", 22),
    ("T6·q9\nFIFA\nwon|lost|3rd",    "T6", "T6_q9", 23),
    ("T7·q4\nThatcher\ncons.univ",   "T7", "T7_q4", 25),
    ("T7·q1\nPopes\nItaly",          "T7", "T7_q1", 26),
    ("T7·q8\nCasino R.\nsequel*",    "T7", "T7_q8", 27),
]

MODELS = {
    "NL_naive":     {"color": "#4C72B0", "marker": "o", "label": r"NL$_\mathrm{naive}$", "lw": 1.5},
    "µ-Galois_Hol": {"color": "#DD8452", "marker": "s", "label": r"$\mu$-Galois\_Hol",  "lw": 1.0},
    "µ-Galois_Dec": {"color": "#55A868", "marker": "^", "label": r"$\mu$-Galois\_Dec",  "lw": 1.0},
    "µ-Galois_F-M": {"color": "#C44E52", "marker": "D", "label": r"$\mu$-Galois\_F-M", "lw": 2.0},
}

TEMPLATE_BANDS = {
    "T1": (0.5,  3.5,  "#EEF2FF"),
    "T2": (4.5,  7.5,  "#F0FFF4"),
    "T3": (8.5,  10.5, "#FFFBEB"),
    "T4": (11.5, 14.5, "#FFF5F5"),
    "T5": (15.5, 19.5, "#F0F9FF"),
    "T6": (20.5, 23.5, "#FDF4FF"),
    "T7": (24.5, 27.5, "#FFF8F0"),
}


def load_results(results_dir: Path) -> dict:
    """
    Load all *_ablation.json files.
    Returns {query_id: {model: f1_mean}}.
    """
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Run experiments/run_general.py first, or pass --results-dir."
        )
    data = {}
    for f in sorted(results_dir.glob("T*_ablation.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        qid = rec["query_id"]
        data[qid] = {m: v["f1_mean"] for m, v in rec["results"].items()}

    if not data:
        raise ValueError(
            f"No *_ablation.json files found in {results_dir}."
        )
    return data


def validate(data: dict) -> None:
    """Raise if any required query or model is missing."""
    required_qids   = [q[2] for q in QUERIES]
    required_models = list(MODELS.keys())
    missing_q, missing_m = [], []

    for qid in required_qids:
        if qid not in data:
            missing_q.append(qid)
        else:
            for m in required_models:
                if m not in data[qid]:
                    missing_m.append(f"{qid}/{m}")

    errors = []
    if missing_q:
        errors.append(f"Missing queries: {missing_q}")
    if missing_m:
        errors.append(f"Missing model results: {missing_m}")
    if errors:
        raise KeyError(
            "Cannot build figure — incomplete results:\n  " +
            "\n  ".join(errors) +
            "\nRe-run the ablation runner for the missing entries."
        )


def make_figure(data: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    xs     = [q[3] for q in QUERIES]
    labels = [q[0] for q in QUERIES]
    qids   = [q[2] for q in QUERIES]

    for tmpl, (x0, x1, color) in TEMPLATE_BANDS.items():
        ax.axvspan(x0, x1, color=color, alpha=0.6, zorder=0)
        ax.text((x0 + x1) / 2, 1.03, tmpl,
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#555",
                transform=ax.get_xaxis_transform())

    for model, cfg in MODELS.items():
        ys = [data[qid][model] for qid in qids]
        ax.plot(xs, ys,
                color=cfg["color"], marker=cfg["marker"],
                linewidth=cfg["lw"], markersize=6,
                label=cfg["label"], alpha=0.85,
                linestyle="-", zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5, ha="center")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    for _, (_, x1, _) in TEMPLATE_BANDS.items():
        ax.axvline(x1, color="#ccc", linewidth=0.7, zorder=1)

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9,
              ncol=4, bbox_to_anchor=(0.0, -0.38))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 28)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.pdf')} and .png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot RQ1 — F1 per query ordered by complexity"
    )
    parser.add_argument(
        "--results-dir", default="experiments/results",
        help="Directory containing T*_ablation.json files (default: experiments/results)"
    )
    parser.add_argument(
        "--out", default="figures/rq1_scatter",
        help="Output path without extension (default: figures/rq1_scatter)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading results from {results_dir} ...")
    data = load_results(results_dir)
    print(f"  Found {len(data)} queries.")

    validate(data)
    print("  Validation passed.")

    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()