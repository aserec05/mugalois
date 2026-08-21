"""
experiments/aggregate_results.py
=================================
Aggregate results from run_general.py JSON outputs into two LaTeX tables:
  Table 1 — F1 macro-average per template (one row per model, one col per template)
  Table 2 — Global macro-average (P / R / F1 / Cardinality / Tokens / Time)
             one row per model, averaged over ALL queries across ALL templates
Usage
-----
    python3 experiments/aggregate_results.py
    python3 experiments/aggregate_results.py --results-dir experiments/results
    python3 experiments/aggregate_results.py --out tables/
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

MODEL_ORDER = [
    "NL_naive", "NL_precise", "SPARQL",
    "µ-Galois_Hol", "µ-Galois_Dec", "µ-Galois_C",
    "µ-Galois_F-M", "µ-Galois_F",
]
MODEL_LATEX = {
    "NL_naive":      r"\textbf{NL\textsubscript{naive}}",
    "NL_precise":    r"\textbf{NL\textsubscript{precise}}",
    "SPARQL":        r"\textbf{SPARQL}",
    "µ-Galois_Hol":  r"\sys{}\_Hol",
    "µ-Galois_Dec":  r"\sys{}\_Dec",
    "µ-Galois_C":    r"\sys{}\_C",
    "µ-Galois_F-M":  r"\sys{}\_F-M",
    "µ-Galois_F":    r"\sys{}\_F",
}
TEMPLATE_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


def load_results(results_dir: Path) -> dict:
    """
    Returns: {template: {query_id: {"n_expected": int, "models": {model: metrics}}}}
    """
    data = defaultdict(dict)
    files = sorted(results_dir.glob("T*_ablation.json"))
    if not files:
        print(f"[warn] No result files found in {results_dir}")
        return data
    for f in files:
        try:
            rec      = json.loads(f.read_text(encoding="utf-8"))
            template = rec["template"]
            qid      = rec["query_id"]
            # n_expected is at record level, not inside each model's metrics
            data[template][qid] = {
                "n_expected": rec.get("n_expected", 0),
                "models":     rec["results"],
            }
        except Exception as e:
            print(f"[warn] Could not load {f.name}: {e}")
    print(f"[aggregate] Loaded {sum(len(v) for v in data.values())} queries "
          f"across {len(data)} templates from {results_dir}")
    return data


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def per_template_f1(data: dict) -> dict:
    """Returns: {template: {model: (f1_mean, f1_std_mean)}}"""
    result = {}
    for template in TEMPLATE_ORDER:
        queries = data.get(template, {})
        by_model     = defaultdict(list)
        by_model_std = defaultdict(list)
        for qid, qdata in queries.items():
            for model, metrics in qdata["models"].items():
                by_model[model].append(metrics.get("f1_mean", 0.0))
                by_model_std[model].append(metrics.get("f1_std", 0.0))
        result[template] = {
            m: (mean(by_model[m]), mean(by_model_std[m]))
            for m in by_model
        }
    return result


def global_metrics(data: dict) -> dict:
    """
    Returns: {model: {metric: value}}
    Macro-averaged over ALL queries across ALL templates.
    Cardinality = mean(n_generated / n_expected) — n_expected read from record level.
    """
    by_model = defaultdict(lambda: defaultdict(list))
    for template in TEMPLATE_ORDER:
        for qid, qdata in data.get(template, {}).items():
            n_expected = qdata["n_expected"]
            for model, m in qdata["models"].items():
                by_model[model]["precision"].append(m.get("precision_mean", 0.0))
                by_model[model]["recall"].append(m.get("recall_mean", 0.0))
                by_model[model]["f1"].append(m.get("f1_mean", 0.0))
                by_model[model]["f1_std"].append(m.get("f1_std", 0.0))
                by_model[model]["tokens"].append(m.get("tokens_mean", 0.0))
                by_model[model]["time"].append(m.get("time_mean", 0.0))
                n_gen = m.get("n_generated", 0.0)
                if n_expected == 0:
                    card = 1.0 if n_gen == 0 else float(n_gen)
                else:
                    card = n_gen / n_expected
                by_model[model]["cardinality"].append(card)
    return {model: {k: mean(v) for k, v in metrics.items()}
            for model, metrics in by_model.items()}


def table_f1_per_template(per_tmpl: dict) -> str:
    # Best F1 per template column
    best = {}
    for t in TEMPLATE_ORDER:
        vals = {m: v[0] for m, v in per_tmpl.get(t, {}).items() if v[0] > 0}
        if vals:
            best[t] = max(vals, key=vals.get)

    cols = "c" * len(TEMPLATE_ORDER)
    header_tmpl = " & ".join(f"\\textbf{{{t}}}" for t in TEMPLATE_ORDER)

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l{cols}}}",
        r"\toprule",
        rf"\textbf{{Model}} & {header_tmpl} \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        name  = MODEL_LATEX.get(model, model)
        cells = []
        for t in TEMPLATE_ORDER:
            entry = per_tmpl.get(t, {}).get(model)
            if entry is None:
                cells.append("--")
            else:
                f1_m, _ = entry
                cell = f"{f1_m:.3f}"
                if best.get(t) == model:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    # Macro-avg row per template
    lines.append(r"\midrule")
    macro_cells = []
    for t in TEMPLATE_ORDER:
        f1s = [per_tmpl.get(t, {}).get(m, (0,0))[0]
               for m in MODEL_ORDER if per_tmpl.get(t, {}).get(m)]
        macro_cells.append(f"{mean(f1s):.3f}" if f1s else "--")
    lines.append(r"\textit{Macro} & " + " & ".join(macro_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Macro-average F1 per template. Best per column in \textbf{bold}. "
        r"Bottom row: average across all models.}",
        r"\label{tab:f1_per_template}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_global(glob: dict) -> str:
    metrics_cols = [
        ("precision",   "P",       "{:.3f}",  True),
        ("recall",      "R",       "{:.3f}",  True),
        ("f1",          "F1",      "{:.3f}",  True),
        ("f1_std",      r"$\pm$std", "{:.3f}", False),
        ("cardinality", "Card.",   "{:.2f}",  None),   # closest to 1
        ("tokens",      "Tokens",  "{:.0f}",  False),
        ("time",        "Time(s)", "{:.1f}",  False),
    ]

    best = {}
    for key, _, _, higher in metrics_cols:
        present = {m: glob[m].get(key, 0) for m in MODEL_ORDER if m in glob}
        if not present:
            continue
        if higher is None:
            best[key] = min(present, key=lambda m: abs(present[m] - 1.0))
        elif higher:
            best[key] = max(present, key=present.get)
        else:
            best[key] = min(present, key=present.get)

    col_spec = "l" + "r" * len(metrics_cols)
    header   = " & ".join(f"\\textbf{{{h}}}" for _, h, _, _ in metrics_cols)

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"\textbf{{Model}} & {header} \\",
        r"\midrule",
    ]

    for i, model in enumerate(MODEL_ORDER):
        if i == 3:
            lines.append(r"\midrule")
        name = MODEL_LATEX.get(model, model)
        if model not in glob:
            lines.append(f"{name} & " + " & ".join(["--"]*len(metrics_cols)) + r" \\")
            continue
        cells = []
        for key, _, fmt, _ in metrics_cols:
            val  = glob[model].get(key, 0.0)
            cell = fmt.format(val)
            if best.get(key) == model:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Global macro-average across all queries (T1--T7). "
        r"Card.\,=\,$|\mathrm{gen}|/|\mathrm{GT}|$. Best per column in \textbf{bold}.}",
        r"\label{tab:global}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_summary(glob: dict) -> None:
    print("\n=== GLOBAL F1 SUMMARY ===")
    print(f"{'Model':<20} {'P':>6} {'R':>6} {'F1':>6} {'±std':>6} "
          f"{'Card.':>6} {'Tok':>7} {'Time':>7}")
    print("-" * 72)
    for model in MODEL_ORDER:
        if model not in glob:
            continue
        m = glob[model]
        print(f"{model:<20} "
              f"{m.get('precision',0):>6.3f} "
              f"{m.get('recall',0):>6.3f} "
              f"{m.get('f1',0):>6.3f} "
              f"{m.get('f1_std',0):>6.3f} "
              f"{m.get('cardinality',0):>6.2f} "
              f"{m.get('tokens',0):>7.0f} "
              f"{m.get('time',0):>7.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments/results",
                        help="Directory with T*_ablation.json files")
    parser.add_argument("--out", default=".",
                        help="Output directory for .tex files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data     = load_results(results_dir)
    per_tmpl = per_template_f1(data)
    glob     = global_metrics(data)

    tex1 = table_f1_per_template(per_tmpl)
    tex2 = table_global(glob)

    (out_dir / "table_f1_per_template.tex").write_text(tex1, encoding="utf-8")
    (out_dir / "table_global.tex").write_text(tex2, encoding="utf-8")
    print(f"[aggregate] Saved table_f1_per_template.tex and table_global.tex → {out_dir}")

    print_summary(glob)


if __name__ == "__main__":
    main()
