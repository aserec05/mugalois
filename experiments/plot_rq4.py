"""
plot_rq4_final.py — RQ4 : 3×2 figure, 6 panels
=================================================
Models (no µ-Galois_F, no µ-Galois_FM+C, no µ-Galois_F-M original):
  NL_naive        → results/   key: NL_naive
  NL_precise      → results/   key: NL_precise
  SPARQL          → results/   key: SPARQL
  µ-Galois_Hol    → results/   key: µ-Galois_Hol
  µ-Galois_Dec    → results/   key: µ-Galois_Dec
  µ-Galois_S      → results_s/ key: µ-Galois_S
  µ-Galois_C_05   → T1-T2: results_new/ key: µ-Galois_C_05
                    T3-T7: results_tau/ key: C_α=0.5
  µ-Galois_FM_05  → T1-T6: results_new/ key: µ-Galois_FM_05
                    T7:     results/     key: µ-Galois_F-M  (fallback)

Panels (3 rows × 2 cols):
  (a) ΔF1 over NL_naive per template
  (b) F1 vs Tokens scatter + Pareto
  (c) Token cost heatmap (models × templates)
  (d) Radar (F1, P, R, efficiency, speed)
  (e) Token cost per template (grouped bars)
  (f) F1 per template (line chart)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE if (_HERE / "results").exists() else _HERE.parent / "experiments"
DIRS  = {
    "main": ROOT / "results",
    "tau":  ROOT / "results_tau",
    "new":  ROOT / "results_new",
    "s":    ROOT / "results_s",
}
FIGDIR = ROOT / "figures"

TEMPLATES  = ["T1","T2","T3","T4","T5","T6","T7"]
TMPL_LABEL = {
    "T1":"T1\nSimple","T2":"T2\nSeed","T3":"T3\n2-hop",
    "T4":"T4\nMulti-hop","T5":"T5\nRecursive",
    "T6":"T6\nChoice","T7":"T7\nConjunct.",
}

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "NL_naive":        {"color":"#555555","marker":"o","ls":"-"},
    "NL_precise":      {"color":"#888888","marker":"s","ls":"--"},
    "SPARQL":          {"color":"#AAAAAA","marker":"^","ls":":"},
    "µ-Galois_Hol":    {"color":"#E07B39","marker":"D","ls":"-"},
    "µ-Galois_Dec":    {"color":"#C8970A","marker":"P","ls":"--"},
    "µ-Galois_C_05":   {"color":"#4A7A4A","marker":"v","ls":":"},
    "µ-Galois_FM_05":  {"color":"#8B1A1A","marker":"*","ls":"-"},
}
MODEL_ORDER = list(MODELS.keys())


def _source(mname, tmpl):
    """Returns (dir_key, result_key) for model × template."""
    if mname == "NL_naive":   return ("main", "NL_naive")
    if mname == "NL_precise": return ("main", "NL_precise")
    if mname == "SPARQL":     return ("main", "SPARQL")
    if mname == "µ-Galois_Hol": return ("main", "µ-Galois_Hol")
    if mname == "µ-Galois_Dec": return ("main", "µ-Galois_Dec")
    if mname == "µ-Galois_C_05":
        return ("new", "µ-Galois_C_05") if tmpl in ("T1","T2") \
               else ("tau", "C_α=0.5")
    if mname == "µ-Galois_FM_05":
        return ("main", "µ-Galois_F-M") if tmpl == "T7" \
               else ("new", "µ-Galois_FM_05")
    return ("main", mname)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _norm(s):
    return s.replace("\u00b5","µ").replace("\u03bc","µ").replace("\u03b1","α").strip()

def _load_ablation(d):
    out = {}
    if not d.exists(): return out
    for f in sorted(d.glob("T*_q*_ablation.json")):
        m = re.match(r"(T\d+)_(q[\w]+)_ablation", f.stem)
        if not m: continue
        data = json.loads(f.read_text())
        out[(m.group(1), m.group(2))] = {_norm(k): v
                                          for k, v in data["results"].items()}
    return out

def _load_tau(d):
    out = {}
    if not d.exists(): return out
    for f in sorted(d.glob("tau_T*_q*.json")):
        m = re.match(r"tau_(T\d+)_(q[\w]+)", f.stem)
        if not m: continue
        data = json.loads(f.read_text())
        out[(m.group(1), m.group(2))] = {_norm(k): v
                                          for k, v in data["results"].items()}
    return out

def load_all():
    raw = {
        "main": _load_ablation(DIRS["main"]),
        "tau":  _load_tau(DIRS["tau"]),
        "new":  _load_ablation(DIRS["new"]),
        "s":    _load_ablation(DIRS["s"]),
    }
    result = {m: {} for m in MODEL_ORDER}
    all_keys = set()
    for d in raw.values(): all_keys |= set(d.keys())
    for (tmpl, qid) in all_keys:
        for mname in MODEL_ORDER:
            dk, mk = _source(mname, tmpl)
            scores = raw[dk].get((tmpl, qid), {}).get(mk)
            if scores is not None:
                result[mname][(tmpl, qid)] = scores
    return result

def aggregate(data):
    out = {}
    for mname, qdict in data.items():
        by_t = defaultdict(list)
        for (t, q), sc in qdict.items(): by_t[t].append(sc)
        out[mname] = {}
        for t, qs in by_t.items():
            mn = lambda k: float(np.mean([q.get(k,0) for q in qs]))
            out[mname][t] = {
                "f1":  round(mn("f1_mean"),3),   "f1s": round(mn("f1_std"),3),
                "tok": round(mn("tokens_mean"),0),"time":round(mn("time_mean"),2),
                "p":   round(mn("precision_mean"),3),
                "r":   round(mn("recall_mean"),3),
            }
    return out

def global_avg(agg):
    out = {}
    for m, td in agg.items():
        if not td: continue
        out[m] = {
            "f1":  np.mean([v["f1"]  for v in td.values()]),
            "tok": np.mean([v["tok"] for v in td.values()]),
            "time":np.mean([v["time"] for v in td.values()]),
            "p":   np.mean([v["p"]   for v in td.values()]),
            "r":   np.mean([v["r"]   for v in td.values()]),
        }
    return out

def present(agg):
    return [m for m in MODEL_ORDER if agg.get(m)]

def lh(models):
    return [mlines.Line2D([],[],
        color=MODELS[m]["color"], marker=MODELS[m]["marker"],
        linestyle=MODELS[m]["ls"], linewidth=1.8, markersize=6, label=m)
        for m in models]


# ── 3×2 Figure ────────────────────────────────────────────────────────────────

def make_figure(agg, glb):
    models = present(agg)
    if not models: print("  [error] no data"); return

    x   = np.arange(len(TEMPLATES))
    fig = plt.figure(figsize=(20, 14))

    # 3 panels top row + 2 panels bottom row (wider)
    gs_top = fig.add_gridspec(1, 3, left=0.05, right=0.97,
                               top=0.93, bottom=0.52,
                               wspace=0.32)
    gs_bot = fig.add_gridspec(1, 2, left=0.06, right=0.97,
                               top=0.44, bottom=0.05,
                               wspace=0.38)

    ax_a = fig.add_subplot(gs_top[0])
    ax_b = fig.add_subplot(gs_top[1])
    ax_c = fig.add_subplot(gs_top[2])
    ax_d = fig.add_subplot(gs_bot[0], projection="polar")
    ax_e = fig.add_subplot(gs_bot[1])

    fig.suptitle("RQ4 — Quality vs. Cost Analysis", fontsize=14, fontweight="bold")

    # ── (a) ΔF1 over NL_naive ────────────────────────────────────────────────
    ax = ax_a
    comp = [m for m in models if m != "NL_naive"]
    n_c  = len(comp)
    w    = 0.7 / n_c
    offs = np.linspace(-(n_c-1)/2, (n_c-1)/2, n_c) * w
    for mi, m in enumerate(comp):
        cfg = MODELS[m]
        deltas = []
        for t in TEMPLATES:
            fm  = agg[m].get(t,{}).get("f1", np.nan)
            fnl = agg.get("NL_naive",{}).get(t,{}).get("f1", np.nan)
            deltas.append(fm - fnl if not (np.isnan(fm) or np.isnan(fnl)) else np.nan)
        ax.bar(x + offs[mi], deltas, w, color=cfg["color"], alpha=0.85,
               edgecolor="#333", linewidth=0.3)
    ax.axhline(0, color="#333", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels([TMPL_LABEL[t] for t in TEMPLATES], fontsize=8.5)
    ax.set_ylabel("ΔF1 vs. NL_naive", fontsize=10)
    ax.set_title("(a) F1 gain over NL_naive", fontsize=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(handles=lh(comp), fontsize=7, ncol=1, loc="lower right", framealpha=0.9)

    # ── (b) F1 vs Tokens scatter ──────────────────────────────────────────────
    ax = ax_b
    pts = []
    for m in models:
        if m not in glb: continue
        cfg = MODELS[m]; g = glb[m]
        ax.scatter(g["tok"], g["f1"], color=cfg["color"], marker=cfg["marker"],
                   s=160, zorder=5, edgecolors="#333", linewidths=0.6)
        ax.annotate(m, (g["tok"], g["f1"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=7.5, color=cfg["color"], fontweight="bold")
        pts.append((g["tok"], g["f1"]))
    pts_s = sorted(pts); pareto, best = [], -1
    for tok, f1 in pts_s:
        if f1 > best: pareto.append((tok, f1)); best = f1
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.step(px, py, where="post", color="#CC0000",
                linewidth=1.5, linestyle="--", alpha=0.6, label="Pareto frontier")
        ax.legend(fontsize=8)
    ax.set_xlabel("Mean tokens per query", fontsize=10)
    ax.set_ylabel("Global macro-average F1", fontsize=10)
    ax.set_title("(b) Quality vs. Token cost\nupper-left = better", fontsize=10)
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # ── (c) Token heatmap ─────────────────────────────────────────────────────
    ax = ax_c
    matrix = np.full((len(models), len(TEMPLATES)), np.nan)
    for mi, m in enumerate(models):
        for ti, t in enumerate(TEMPLATES):
            matrix[mi, ti] = agg[m].get(t,{}).get("tok", np.nan)
    vmin = np.nanmin(matrix); vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Mean tokens", shrink=0.85)
    ax.set_xticks(range(len(TEMPLATES)))
    ax.set_xticklabels([TMPL_LABEL[t] for t in TEMPLATES], fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    for mi in range(len(models)):
        for ti in range(len(TEMPLATES)):
            v = matrix[mi, ti]
            if not np.isnan(v):
                ax.text(ti, mi, f"{int(v)}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > vmin + 0.6*(vmax-vmin) else "#222")
    ax.set_title("(c) Token cost heatmap", fontsize=10)

    # ── (d) Radar — large ─────────────────────────────────────────────────────
    ax = ax_d
    cats   = ["F1","Precision","Recall","Efficiency\n(1-tokens)","Speed\n(1-latency)"]
    n_cats = len(cats)
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    max_tok  = max(glb[m]["tok"]  for m in models if m in glb)
    min_tok  = min(glb[m]["tok"]  for m in models if m in glb)
    max_time = max(glb[m]["time"] for m in models if m in glb)
    for m in models:
        if m not in glb: continue
        cfg = MODELS[m]; g = glb[m]
        eff  = 1 - (g["tok"]-min_tok) / max(max_tok-min_tok, 1)
        spd  = 1 - g["time"] / max_time
        vals = [g["f1"], g["p"], g["r"], eff, spd] + [g["f1"]]
        ax.plot(angles, vals, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2.0, marker=cfg["marker"], markersize=6, label=m)
        ax.fill(angles, vals, color=cfg["color"], alpha=0.07)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("(d) Model profile radar\nF1 · Precision · Recall · Efficiency · Speed",
                 fontsize=10, pad=28)
    ax.legend(handles=lh(models), fontsize=8, loc="upper right",
              bbox_to_anchor=(1.55, 1.18), framealpha=0.92, ncol=1)

    # ── (e) F1 per template ───────────────────────────────────────────────────
    ax = ax_e
    for m in models:
        cfg  = MODELS[m]
        y    = [agg[m].get(t,{}).get("f1",  np.nan) for t in TEMPLATES]
        yerr = [agg[m].get(t,{}).get("f1s", 0)      for t in TEMPLATES]
        ax.plot(x, y, color=cfg["color"], marker=cfg["marker"],
                linestyle=cfg["ls"], linewidth=2, markersize=7)
        ax.fill_between(x,
            [v-e if not np.isnan(v) else np.nan for v,e in zip(y,yerr)],
            [v+e if not np.isnan(v) else np.nan for v,e in zip(y,yerr)],
            color=cfg["color"], alpha=0.08)
    ax.set_xticks(x); ax.set_xticklabels([TMPL_LABEL[t] for t in TEMPLATES], fontsize=9)
    ax.set_ylabel("Macro-average F1", fontsize=10)
    ax.set_title("(e) F1 per template  (ordered by complexity →)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(handles=lh(models), fontsize=8, ncol=2, loc="lower left", framealpha=0.92)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf","png"):
        p = FIGDIR / f"rq4_final.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=180)
        print(f"  Saved → {p}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"ROOT = {ROOT}")
    data = load_all()
    for m, qd in data.items():
        print(f"  {m}: {len(qd)} queries")
    agg = aggregate(data)
    glb = global_avg(agg)
    print("\nGlobal F1:")
    for m, g in glb.items():
        print(f"  {m}: F1={g['f1']:.3f}  tok={g['tok']:.0f}  time={g['time']:.1f}s")
    make_figure(agg, glb)
    print("Done.")