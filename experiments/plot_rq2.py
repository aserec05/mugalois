"""
experiments/plot_rq2.py
=======================
Two figures for RQ2 — Confidence vs Structure signals.

Figure 1 — α sensitivity (3 subplots): F1 / Tokens / Time vs α
           T3/T4/T5/T6 only. T7 excluded (confidence controls meta-execution,
           not routing). No cross-template F-M reference line.

Figure 2 — Side-by-side: bar comparison (_S/_C/_F-M) + per-query scatter
           Left  : grouped bars per template + macro
           Right : F1 per query for _F-M, _S, _C (best/low/high α)

Usage
-----
    python3 experiments/plot_rq2.py
    python3 experiments/plot_rq2.py --tau-dir experiments/results_tau
                                    --s-dir   experiments/results_s
                                    --ref-dir experiments/results
                                    --out     figures/rq2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Palette ───────────────────────────────────────────────────────────────────

TEMPLATE_COLOR = {
    "T3": "#4C72B0", "T4": "#8B5CF6",
    "T5": "#C44E52", "T6": "#2DA44E", "T7": "#B8860B",
}
MODEL_COLOR = {
    "µ-Galois_S":           "#2DA44E",
    "µ-Galois_C":           "#DD8452",
    "µ-Galois_C (best α)": "#DD8452",
    "µ-Galois_F-M":         "#C44E52",
}
ALPHA_VALUES = [0.20, 0.35, 0.50, 0.60, 0.70, 0.80]
TEMPLATES_TAU = ["T3", "T4", "T5", "T6", "T7"]
TEMPLATES_ALL = ["T3", "T4", "T5", "T6", "T7"]

QUERIES_RQ2 = [
    ("T3·q1\nNolan\nthriller",    "T3", "T3_q1",  1),
    ("T3·q2\nDC\nPMs",            "T3", "T3_q2",  2),
    ("T3·q3\nAfrica\nCommonw.",   "T3", "T3_q3",  3),
    ("T4·q1\nWilde\ncomedy",      "T4", "T4_q1",  5),
    ("T4·q3\nShakespeare\nItaly", "T4", "T4_q3",  6),
    ("T4·q5\nNobel\nAsia",        "T4", "T4_q5",  7),
    ("T5·q2\n264 popes\nbwd",     "T5", "T5_q2",  9),
    ("T5·q7\nMartin V\n49-hop",   "T5", "T5_q7", 10),
    ("T5·q9\nEliz. II\nDAG",      "T5", "T5_q9", 11),
    ("T6·q1\nEinstein\n|sibling", "T6", "T6_q1", 13),
    ("T6·q9\nFIFA\nwon|lost|3rd", "T6", "T6_q9", 14),
    ("T6·q10\nEinstein\nfamily",  "T6", "T6_q10",15),
    ("T7·q1\nPopes\nItaly",        "T7", "T7_q1", 17),
    ("T7·q4\nThatcher\ncons.",     "T7", "T7_q4", 18),
    ("T7·q8\nCasino R.\nsequel*",  "T7", "T7_q8", 19),
]
TEMPLATE_BANDS = {
    "T3": (0.5,  3.5,  "#EEF2FF"),
    "T4": (4.5,  7.5,  "#F5F3FF"),
    "T5": (8.5,  11.5, "#FFF5F5"),
    "T6": (12.5, 15.5, "#F0FFF4"),
    "T7": (16.5, 19.5, "#FFF8F0"),
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_tau_results(tau_dir: Path) -> dict:
    if not tau_dir.exists():
        raise FileNotFoundError(f"τ results not found: {tau_dir}")
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(tau_dir.glob("tau_T*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        tmpl = rec["template"]
        if tmpl not in TEMPLATES_TAU:
            continue
        for key, m in rec["results"].items():
            try:
                alpha = float(key.split("=")[1])
            except (IndexError, ValueError):
                continue
            data[tmpl][alpha]["f1"].append(m["f1_mean"])
            data[tmpl][alpha]["tokens"].append(m["tokens_mean"])
            data[tmpl][alpha]["time"].append(m["time_mean"])
    return {
        tmpl: {
            alpha: {k: sum(v)/len(v) if v else float("nan") for k, v in vals.items()}
            for alpha, vals in by_a.items()
        }
        for tmpl, by_a in data.items()
    }


def load_tau_per_query(tau_dir: Path) -> dict:
    if not tau_dir.exists():
        raise FileNotFoundError(f"τ results not found: {tau_dir}")
    data = {}
    for f in sorted(tau_dir.glob("tau_T*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        raw = rec["query_id"]
        qid = raw.removeprefix("tau_").removeprefix("s_")
        data[qid] = {}
        for key, m in rec["results"].items():
            try:
                data[qid][float(key.split("=")[1])] = m["f1_mean"]
            except (IndexError, ValueError):
                continue
    return data


def load_ref_results(ref_dir: Path, model: str) -> dict:
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference results not found: {ref_dir}")
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(ref_dir.glob("T*_ablation.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        tmpl = rec["template"]
        m = rec["results"].get(model, {})
        if not m:
            continue
        data[tmpl]["f1"].append(m.get("f1_mean", float("nan")))
        data[tmpl]["tokens"].append(m.get("tokens_mean", float("nan")))
        data[tmpl]["time"].append(m.get("time_mean", float("nan")))
    return {tmpl: {k: sum(v)/len(v) if v else float("nan") for k, v in vals.items()}
            for tmpl, vals in data.items()}


def load_ref_per_query(ref_dir: Path, model: str) -> dict:
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference results not found: {ref_dir}")
    data = {}
    for f in sorted(ref_dir.glob("T*_ablation.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        m = rec["results"].get(model, {})
        raw = rec["query_id"]
        qid = raw.removeprefix("tau_").removeprefix("s_")
        data[qid] = m.get("f1_mean", float("nan"))
    return data


def load_s_results(s_dir: Path, ref_dir: Path = None) -> dict:
    if not s_dir.exists():
        raise FileNotFoundError(f"µ-Galois_S results not found: {s_dir}")
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(s_dir.glob("s_T*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        tmpl = rec["template"]
        m = rec["results"].get("µ-Galois_S", {})
        data[tmpl]["f1"].append(m.get("f1_mean", float("nan")))
        data[tmpl]["tokens"].append(m.get("tokens_mean", float("nan")))
        data[tmpl]["time"].append(m.get("time_mean", float("nan")))
    result = {tmpl: {k: sum(v)/len(v) if v else float("nan") for k, v in vals.items()}
              for tmpl, vals in data.items()}
    # T7: _S = _F-M (structural planner, confidence unused)
    if ref_dir and ref_dir.exists() and "T7" not in result:
        fm = load_ref_results(ref_dir, "µ-Galois_F-M")
        if "T7" in fm:
            result["T7"] = fm["T7"]
    return result


def load_s_per_query(s_dir: Path, ref_dir: Path = None) -> dict:
    if not s_dir.exists():
        raise FileNotFoundError(f"µ-Galois_S results not found: {s_dir}")
    data = {}
    for f in sorted(s_dir.glob("s_T*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        m = rec["results"].get("µ-Galois_S", {})
        raw = rec["query_id"]
        qid = raw.removeprefix("tau_").removeprefix("s_")
        data[qid] = m.get("f1_mean", float("nan"))
    # T7 fallback
    if ref_dir and ref_dir.exists():
        for qid, f1 in load_ref_per_query(ref_dir, "µ-Galois_F-M").items():
            if qid.startswith("T7_") and qid not in data:
                data[qid] = f1
    return data


# ── α selection ───────────────────────────────────────────────────────────────

def select_alpha_trio(tau_data: dict) -> tuple[float, float, float]:
    global_f1 = {}
    for alpha in ALPHA_VALUES:
        vals = [tau_data.get(t, {}).get(alpha, {}).get("f1", float("nan"))
                for t in TEMPLATES_TAU]
        vals = [v for v in vals if not np.isnan(v)]
        global_f1[alpha] = sum(vals)/len(vals) if vals else float("nan")
    alpha_best = max(
        (a for a in ALPHA_VALUES if not np.isnan(global_f1.get(a, float("nan")))),
        key=lambda a: global_f1[a]
    )
    print(f"  α_best={alpha_best} (F1={global_f1[alpha_best]:.3f}), "
          f"α_low=0.20, α_high=0.80")
    return alpha_best, 0.20, 0.80


# ── Figure 1 — α sensitivity ──────────────────────────────────────────────────

def plot_alpha_sensitivity(tau_data: dict, out_path: Path):
    """
    Heatmap table: rows = templates + Macro, columns = alpha values.
    Cell = F1, colored red→green. Best per row in bold + dark border.
    """
    import matplotlib.colors as mcolors

    alphas    = ALPHA_VALUES
    templates = [t for t in TEMPLATES_TAU if t in tau_data]

    # Collect F1 values
    row_data = []
    all_f1   = []
    for tmpl in templates:
        row = [tau_data.get(tmpl, {}).get(a, {}).get("f1", float("nan")) for a in alphas]
        row_data.append((tmpl, row))
        all_f1 += [v for v in row if not np.isnan(v)]

    macro_row = []
    for a in alphas:
        vals = [tau_data.get(t, {}).get(a, {}).get("f1", float("nan")) for t in templates]
        vals = [v for v in vals if not np.isnan(v)]
        macro_row.append(sum(vals)/len(vals) if vals else float("nan"))
    all_f1 += [v for v in macro_row if not np.isnan(v)]

    vmin = min(all_f1) - 0.02
    vmax = max(all_f1) + 0.02
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    n_rows = len(templates) + 1
    n_cols = len(alphas)
    fig, ax = plt.subplots(figsize=(n_cols * 1.5 + 2.0, n_rows * 0.7 + 1.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    fig.suptitle(
        "F1 of µ-Galois_C across alpha (T3-T7)\n"
        "Default: tau_simple=0.80 (T3); tau_chain=tau_rec=tau_choice=tau_hybrid=0.50 (T4-T7)",
        fontsize=8.5, y=1.04
    )

    # Column headers
    for ci, a in enumerate(alphas):
        ax.text(ci + 0.5, n_rows - 0.08,
                f"\u03b1={a}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#444")

    def draw_row(ri, label, values, lcolor="#333", is_macro=False):
        y = n_rows - ri - 1
        h = 0.82
        ax.text(-0.12, y + h/2, label,
                ha="right", va="center", fontsize=9,
                fontweight="bold" if is_macro else "normal",
                color=lcolor)
        valid   = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        best_i  = max(valid, key=lambda x: x[1])[0] if valid else -1
        for ci, (a, val) in enumerate(zip(alphas, values)):
            x = ci
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((x, y), 1, h,
                    facecolor="#f5f5f5", edgecolor="#ddd", lw=0.4))
                ax.text(x+0.5, y+h/2, "—", ha="center", va="center",
                        fontsize=8, color="#bbb")
            else:
                fc  = cmap(norm(val))
                lw  = 2.0 if ci == best_i else 0.4
                ec  = "#222" if ci == best_i else "#ccc"
                ax.add_patch(plt.Rectangle((x, y), 1, h,
                    facecolor=fc, edgecolor=ec, lw=lw))
                mid  = (vmin + vmax) / 2
                tcol = "white" if val < mid else "#1a1a1a"
                fw   = "bold" if ci == best_i else "normal"
                ax.text(x+0.5, y+h/2, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=8.5, color=tcol, fontweight=fw)

    for ri, (tmpl, values) in enumerate(row_data):
        draw_row(ri, tmpl, values, lcolor=TEMPLATE_COLOR.get(tmpl, "#333"))

    # Separator line before Macro
    ax.plot([0, n_cols], [1, 1], color="#888", lw=0.8, ls="--")
    draw_row(len(row_data), "Macro", macro_row, lcolor="#444", is_macro=True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.01, aspect=18)
    cbar.set_label("F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, out_path)

# ── Figure 2 — Bars + per-query side by side ──────────────────────────────────

def _best_alpha_for(tmpl, tau_data):
    best = tau_data.get(tmpl, {})
    if not best:
        return None
    cands = {a: v.get("f1", float("nan")) for a, v in best.items()
             if not np.isnan(v.get("f1", float("nan")))}
    return max(cands, key=cands.get) if cands else None


def plot_combined(s_data, ref_c, ref_fm,
                  tau_pq, s_pq, ref_fm_pq,
                  alpha_best, out_path, tau_data=None):
    fig, (ax_bar, ax_pq) = plt.subplots(
        1, 2, figsize=(18, 5.5),
        gridspec_kw={"width_ratios": [1, 2.2]}
    )

    # ── LEFT: grouped bars ────────────────────────────────────────────────────
    templates = [t for t in TEMPLATES_ALL]
    groups    = templates + ["Macro"]
    models    = ["µ-Galois_S", "µ-Galois_C", "µ-Galois_C (best α)", "µ-Galois_F-M"]
    bar_w     = 0.16
    gap       = 0.04
    x         = np.arange(len(groups))
    offset    = np.array([-1.5, -0.5, 0.5, 1.5]) * (bar_w + gap/2)

    def get_f1(model, tmpl):
        if model == "µ-Galois_S":
            return s_data.get(tmpl, {}).get("f1", float("nan"))
        elif model == "µ-Galois_C":
            return ref_c.get(tmpl, {}).get("f1", float("nan"))
        elif model == "µ-Galois_C (best α)":
            best = tau_data.get(tmpl, {}) if tau_data else {}
            if best:
                best_f1 = max(
                    (v.get("f1", float("nan")) for v in best.values()
                     if not np.isnan(v.get("f1", float("nan")))),
                    default=float("nan")
                )
                if not np.isnan(best_f1):
                    return best_f1
            return ref_c.get(tmpl, {}).get("f1", float("nan"))
        else:
            return ref_fm.get(tmpl, {}).get("f1", float("nan"))

    def macro(model):
        vals = [get_f1(model, t) for t in templates
                if not np.isnan(get_f1(model, t))]
        return sum(vals)/len(vals) if vals else float("nan")

    for i, model in enumerate(models):
        ys = [get_f1(model, t) for t in templates] + [macro(model)]
        bars = ax_bar.bar(x + offset[i], ys, width=bar_w,
                          color=MODEL_COLOR[model], alpha=0.85,
                          label=(r"$\mu$-Galois\_C (best $\alpha$, may differ from default)"
                          if "best" in model
                          else model.replace("µ", r"$\mu$")))
        for jj, (bar, val) in enumerate(zip(bars, ys)):
            if not np.isnan(val):
                ax_bar.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=5.5, color="#333")
                if "best" in model and jj < len(templates) and bar.get_height() > 0.15:
                    ba = _best_alpha_for(templates[jj], tau_data or {})
                    if ba is not None:
                        ax_bar.text(bar.get_x() + bar.get_width()/2,
                                    bar.get_height() / 2,
                                    f"α={ba}", ha="center", va="center",
                                    fontsize=5.0, color="white",
                                    fontweight="bold", rotation=90)

    ax_bar.axvline(len(groups) - 1 - 0.5, color="#aaa",
                   linewidth=0.8, linestyle=":")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(groups, fontsize=9)
    for i, tmpl in enumerate(templates):
        ax_bar.get_xticklabels()[i].set_color(TEMPLATE_COLOR[tmpl])
        ax_bar.get_xticklabels()[i].set_fontweight("bold")
    ax_bar.set_ylabel("Macro-average F1", fontsize=10)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bar.legend(fontsize=7.5, framealpha=0.9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax_bar.set_title("(a) Model comparison (T3–T7)", fontsize=10)

    # ── RIGHT: per-query scatter ───────────────────────────────────────────────
    xs     = [q[3] for q in QUERIES_RQ2]
    labels = [q[0] for q in QUERIES_RQ2]
    qids   = [q[2] for q in QUERIES_RQ2]

    for tmpl, (x0, x1, color) in TEMPLATE_BANDS.items():
        ax_pq.axvspan(x0, x1, color=color, alpha=0.6, zorder=0)
        ax_pq.text((x0+x1)/2, 1.05, tmpl, ha="center", va="bottom",
                   fontsize=8.5, fontweight="bold",
                   color=TEMPLATE_COLOR[tmpl],
                   transform=ax_pq.get_xaxis_transform())

    def s_val(qid):
        return s_pq.get(qid, float("nan"))

    lines = [
        (r"$\mu$-Galois\_F-M",                    "#C44E52", "D", 2.0,
         [ref_fm_pq.get(q, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_S",                       "#2DA44E", "s", 1.5,
         [s_val(q) for q in qids]),
        (fr"$\mu$-Galois\_C ($\alpha$={alpha_best})",  "#DD8452", "o", 1.5,
         [tau_pq.get(q, {}).get(alpha_best, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_C ($\alpha$=0.50, default)", "#F5A623", "^", 1.0,
         [tau_pq.get(q, {}).get(0.50, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_C ($\alpha$=0.80, holistic)", "#B0B0B0", "v", 1.0,
         [tau_pq.get(q, {}).get(0.80, float("nan")) for q in qids]),
    ]

    for label, color, marker, lw, ys in lines:
        ax_pq.plot(xs, ys, color=color, marker=marker, linewidth=lw,
                   markersize=4.5, label=label, alpha=0.85, zorder=3)

    for _, (_, x1, _) in TEMPLATE_BANDS.items():
        ax_pq.axvline(x1, color="#ccc", linewidth=0.7, zorder=1)

    ax_pq.set_xticks(xs)
    ax_pq.set_xticklabels(labels, fontsize=5.8, ha="center")
    ax_pq.tick_params(axis="x", length=0, pad=5)
    ax_pq.set_ylim(-0.05, 1.12)
    ax_pq.set_ylabel("F1", fontsize=10)
    ax_pq.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_pq.spines["top"].set_visible(False)
    ax_pq.spines["right"].set_visible(False)
    ax_pq.set_xlim(0, 20.5)
    ax_pq.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                 ncol=3, bbox_to_anchor=(0.0, -0.40))
    ax_pq.set_title("(b) Per-query F1 — T3–T7", fontsize=10)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, out_path)


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    plt.close(fig)



# ── Unified RQ2 figure ────────────────────────────────────────────────────────

def plot_rq2_unified(tau_data, s_data, ref_c, ref_fm,
                     tau_pq, s_pq, ref_fm_pq,
                     alpha_best, out_path, tau_data_bars=None):
    """
    Single figure: heatmap (top-left) | bars (top-right) | scatter (bottom).
    """
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.1, 1.0],
                            width_ratios=[1.0, 1.2],
                            hspace=0.38, wspace=0.28)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar  = fig.add_subplot(gs[0, 1])
    ax_pq   = fig.add_subplot(gs[1, :])

    td = tau_data_bars or tau_data

    # ── (a) Heatmap ───────────────────────────────────────────────────────────
    alphas    = ALPHA_VALUES
    templates = [t for t in TEMPLATES_TAU if t in tau_data]

    row_data, all_f1 = [], []
    for tmpl in templates:
        row = [tau_data.get(tmpl, {}).get(a, {}).get("f1", float("nan")) for a in alphas]
        row_data.append((tmpl, row))
        all_f1 += [v for v in row if not np.isnan(v)]

    macro_row = []
    for a in alphas:
        vals = [tau_data.get(t, {}).get(a, {}).get("f1", float("nan")) for t in templates]
        vals = [v for v in vals if not np.isnan(v)]
        macro_row.append(sum(vals)/len(vals) if vals else float("nan"))
    all_f1 += [v for v in macro_row if not np.isnan(v)]

    vmin = min(all_f1) - 0.02 if all_f1 else 0.4
    vmax = max(all_f1) + 0.02 if all_f1 else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    n_rows = len(templates) + 1
    n_cols = len(alphas)
    ax_heat.set_xlim(0, n_cols); ax_heat.set_ylim(0, n_rows); ax_heat.axis("off")
    ax_heat.set_title("(a) F1 of µ-Galois_C across alpha", fontsize=9, pad=6)

    for ci, a in enumerate(alphas):
        ax_heat.text(ci+0.5, n_rows-0.06, f"a={a}", ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color="#444")

    def heat_row(ri, label, values, lcolor="#333", is_macro=False):
        y, h = n_rows - ri - 1, 0.82
        ax_heat.text(-0.1, y+h/2, label, ha="right", va="center",
                     fontsize=8, fontweight="bold" if is_macro else "normal", color=lcolor)
        valid  = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        best_i = max(valid, key=lambda x: x[1])[0] if valid else -1
        for ci, val in enumerate(values):
            if np.isnan(val):
                ax_heat.add_patch(plt.Rectangle((ci, y), 1, h, facecolor="#f5f5f5", edgecolor="#ddd", lw=0.4))
                ax_heat.text(ci+0.5, y+h/2, "-", ha="center", va="center", fontsize=7, color="#bbb")
            else:
                fc = cmap(norm(val)); lw = 1.8 if ci==best_i else 0.4; ec = "#222" if ci==best_i else "#ccc"
                ax_heat.add_patch(plt.Rectangle((ci, y), 1, h, facecolor=fc, edgecolor=ec, lw=lw))
                tc = "white" if val < (vmin+vmax)/2 else "#1a1a1a"
                ax_heat.text(ci+0.5, y+h/2, f"{val:.3f}", ha="center", va="center",
                             fontsize=7.5, color=tc, fontweight="bold" if ci==best_i else "normal")

    for ri, (tmpl, values) in enumerate(row_data):
        heat_row(ri, tmpl, values, lcolor=TEMPLATE_COLOR.get(tmpl, "#333"))
    ax_heat.plot([0, n_cols], [1, 1], color="#888", lw=0.7, ls="--")
    heat_row(len(row_data), "Macro", macro_row, lcolor="#444", is_macro=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_heat, orientation="vertical", fraction=0.04, pad=0.02, aspect=16)
    cb.set_label("F1", fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # ── (b) Bar chart ─────────────────────────────────────────────────────────
    models    = ["µ-Galois_S", "µ-Galois_C", "µ-Galois_C (best alpha)", "µ-Galois_F-M"]
    colors_m  = ["#2DA44E", "#DD8452", "#DD8452", "#C44E52"]
    bar_tmpls = list(TEMPLATES_ALL)
    groups    = bar_tmpls + ["Macro"]
    bar_w, gap = 0.16, 0.04
    x      = np.arange(len(groups))
    offset = np.array([-1.5, -0.5, 0.5, 1.5]) * (bar_w + gap/2)

    def get_f1_b(model, tmpl):
        if model == "µ-Galois_S":
            return s_data.get(tmpl, {}).get("f1", float("nan"))
        elif model == "µ-Galois_C":
            return ref_c.get(tmpl, {}).get("f1", float("nan"))
        elif "best" in model:
            best = td.get(tmpl, {})
            if best:
                bv = max((v.get("f1", float("nan")) for v in best.values()
                          if not np.isnan(v.get("f1", float("nan")))), default=float("nan"))
                if not np.isnan(bv): return bv
            return ref_c.get(tmpl, {}).get("f1", float("nan"))
        else:
            return ref_fm.get(tmpl, {}).get("f1", float("nan"))

    def macro_b(model):
        vals = [get_f1_b(model, t) for t in bar_tmpls if not np.isnan(get_f1_b(model, t))]
        return sum(vals)/len(vals) if vals else float("nan")

    hatches = [None, None, "///", None]
    for i, (model, color, hatch) in enumerate(zip(models, colors_m, hatches)):
        ys = [get_f1_b(model, t) for t in bar_tmpls] + [macro_b(model)]
        lbl = (r"$\mu$-Galois\_C (best $\alpha$)" if "best" in model
               else model.replace("µ", r"$\mu$"))
        bars = ax_bar.bar(x + offset[i], ys, width=bar_w, color=color,
                          alpha=0.85, label=lbl, hatch=hatch,
                          edgecolor="white" if hatch else None)
        for bar, val in zip(bars, ys):
            if not np.isnan(val):
                ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=4.8, color="#333")

    ax_bar.axvline(len(groups)-1-0.5, color="#aaa", lw=0.7, ls=":")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(groups, fontsize=8)
    for i, tmpl in enumerate(bar_tmpls):
        ax_bar.get_xticklabels()[i].set_color(TEMPLATE_COLOR.get(tmpl, "#333"))
        ax_bar.get_xticklabels()[i].set_fontweight("bold")
    ax_bar.set_ylabel("Macro-average F1", fontsize=9)
    ax_bar.set_ylim(0, 1.10)
    ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bar.legend(fontsize=6.5, framealpha=0.9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax_bar.set_title("(b) Model comparison (T3-T7)", fontsize=9, pad=6)

    # ── (c) Per-query scatter ─────────────────────────────────────────────────
    xs     = [q[3] for q in QUERIES_RQ2]
    labels = [q[0] for q in QUERIES_RQ2]
    qids   = [q[2] for q in QUERIES_RQ2]

    for tmpl, (x0, x1, color) in TEMPLATE_BANDS.items():
        ax_pq.axvspan(x0, x1, color=color, alpha=0.6, zorder=0)
        ax_pq.text((x0+x1)/2, 1.05, tmpl, ha="center", va="bottom",
                   fontsize=8.5, fontweight="bold", color=TEMPLATE_COLOR[tmpl],
                   transform=ax_pq.get_xaxis_transform())

    pq_lines = [
        (r"$\mu$-Galois\_F-M",                   "#C44E52", "D", 2.0,
         [ref_fm_pq.get(q, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_S",                      "#2DA44E", "s", 1.5,
         [s_pq.get(q, float("nan")) for q in qids]),
        (fr"$\mu$-Galois\_C (best a={alpha_best})", "#DD8452", "o", 1.5,
         [tau_pq.get(q, {}).get(alpha_best, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_C (a=0.50, default)",   "#F5A623", "^", 1.0,
         [tau_pq.get(q, {}).get(0.50, float("nan")) for q in qids]),
        (r"$\mu$-Galois\_C (a=0.80, holistic)",  "#B0B0B0", "v", 1.0,
         [tau_pq.get(q, {}).get(0.80, float("nan")) for q in qids]),
    ]
    for label, color, marker, lw, ys in pq_lines:
        ax_pq.plot(xs, ys, color=color, marker=marker, linewidth=lw,
                   markersize=4.5, label=label, alpha=0.85, zorder=3)
    for _, (_, x1, _) in TEMPLATE_BANDS.items():
        ax_pq.axvline(x1, color="#ccc", linewidth=0.7, zorder=1)

    ax_pq.set_xticks(xs); ax_pq.set_xticklabels(labels, fontsize=5.8, ha="center")
    ax_pq.tick_params(axis="x", length=0, pad=5)
    ax_pq.set_ylim(-0.05, 1.12); ax_pq.set_ylabel("F1", fontsize=10)
    ax_pq.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_pq.spines["top"].set_visible(False); ax_pq.spines["right"].set_visible(False)
    ax_pq.set_xlim(0, 20.5)
    ax_pq.legend(loc="lower left", fontsize=7.5, framealpha=0.9,
                 ncol=3, bbox_to_anchor=(0.0, -0.38))
    ax_pq.set_title("(c) Per-query F1 - T3-T7", fontsize=9, pad=6)

    fig.savefig(out_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-dir", default="experiments/results_tau")
    parser.add_argument("--s-dir",   default="experiments/results_s")
    parser.add_argument("--ref-dir", default="experiments/results")
    parser.add_argument("--out",     default="figures/rq2")
    args = parser.parse_args()

    tau_dir = Path(args.tau_dir)
    s_dir   = Path(args.s_dir)
    ref_dir = Path(args.ref_dir)
    out     = Path(args.out)

    print("Loading τ ablation results...")
    tau_data = load_tau_results(tau_dir)

    print("Loading µ-Galois_S results...")
    s_data   = load_s_results(s_dir, ref_dir)

    print("Loading reference results...")
    ref_c    = load_ref_results(ref_dir, "µ-Galois_C")
    ref_fm   = load_ref_results(ref_dir, "µ-Galois_F-M")


    tau_pq    = load_tau_per_query(tau_dir)
    s_pq      = load_s_per_query(s_dir, ref_dir)
    ref_fm_pq = load_ref_per_query(ref_dir, "µ-Galois_F-M")
    alpha_best, _, _ = select_alpha_trio(tau_data)

    print("Generating unified RQ2 figure...")
    plot_rq2_unified(tau_data, s_data, ref_c, ref_fm,
                     tau_pq, s_pq, ref_fm_pq,
                     alpha_best, out.parent / (out.name + "_unified"),
                     tau_data_bars=tau_data)
    print("Done.")


if __name__ == "__main__":
    main()