"""
rq3_extra.py — T2q5 + T5q2 only, appends to results_rq3_final.json
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM
from mugalois.core.types import TriplePattern, RecursivePattern, Environment
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.closures.motivational import MotivationalClosure
from mugalois.closures.contrast     import ContrastClosure
from mugalois.closures.socratic     import SocraticClosure
from mugalois.closures.alphabet     import AlphabetClosure
from evaluation.metrics import Metrics


def load_gt(qkey):
    paths = {
        "T2q5": (ROOT / "experiments/template2/queries.json",      "q5"),
        "T5q2": (ROOT / "experiments/template5/queries_t5.json",   "q2"),
    }
    p, qid = paths[qkey]
    data = json.loads(p.read_text())
    return set(data[qid]["ground_truth"])


def run_with_closures(initial_fn, tp, gt, closures, llm,
                      n_break=2, max_iter=6, label="?"):
    t0    = time.time()
    found = {str(v) for v in initial_fn()}
    n_calls = 1
    empty_streak = 0
    for i in range(max_iter if closures else 0):
        closure  = closures[i % len(closures)]
        prompt   = closure.prompt(tp, found, llm)
        new_vals = json_to_values(
            llm.chat(build_value_messages(prompt)).text
        ) - found
        n_calls += 1
        if not new_vals:
            empty_streak += 1
            if empty_streak >= n_break: break
        else:
            empty_streak = 0
            found |= new_vals
    scores = Metrics.compute(found, gt, mode="values")
    return {"label": label, "n": len(found),
            "P": scores.precision, "R": scores.recall, "F1": scores.f1,
            "calls": n_calls, "time_s": round(time.time()-t0, 1)}


def run_T2q5(llm, gt, closures, label):
    """?country population >10M — open NL scan, GT=90"""
    def initial():
        prompt = (
            "List all countries in the world whose population exceeds 10 million. "
            "Be exhaustive — include every single country. "
            "Respond ONLY in valid JSON: {\"values\": [\"Country1\", ...]}"
        )
        return json_to_values(llm.chat(build_value_messages(prompt)).text)
    tp = TriplePattern("?country", "population > 10M", "true")
    return run_with_closures(initial, tp, gt, closures, llm, label=label)


def run_T5q2(llm, gt, closures, label):
    """All predecessors of John Paul II — backward, GT=264"""
    from mugalois.rec.simple_rec import LLMRecScan
    pattern = RecursivePattern(
        "?a", "came immediately before", "Pope John Paul II", operator="*"
    )
    def initial():
        return LLMRecScan(pattern, llm, motivational=False)
    tp = TriplePattern("?a", "came immediately before", "Pope John Paul II")
    return run_with_closures(initial, tp, gt, closures, llm,
                             n_break=2, max_iter=6, label=label)


QUERY_RUNNERS = {
    "T2q5": (run_T2q5, 90),
    "T5q2": (run_T5q2, 264),
}

CLOSURE_CONFIGS = {
    "none":         [],
    "motivational": [MotivationalClosure()],
    "alphabet":     [AlphabetClosure()],
    "mot+socratic": [MotivationalClosure(), SocraticClosure()],
    "contrast":     [ContrastClosure()],
}


def run_n(qkey, llm, n_runs=3, closures_filter=None):
    runner_fn, gt_size = QUERY_RUNNERS[qkey]
    gt = load_gt(qkey)
    print(f"\n{'='*72}")
    print(f"  {qkey}  |  GT={len(gt)}  |  {n_runs} runs each")
    print(f"{'='*72}")
    configs = {k: v for k, v in CLOSURE_CONFIGS.items()
               if closures_filter is None or k in closures_filter}
    summary = {}
    for label, closures in configs.items():
        f1s, rs, ns, calls_list = [], [], [], []
        print(f"\n  [{label}]")
        for run_i in range(n_runs):
            fresh = [type(c)() for c in closures]
            r = runner_fn(llm, gt, fresh, label=label)
            f1s.append(r["F1"]); rs.append(r["R"])
            ns.append(r["n"]); calls_list.append(r["calls"])
            print(f"    run {run_i+1}/{n_runs}  "
                  f"n={r['n']:>3}  P={r['P']:.3f}  R={r['R']:.3f}  "
                  f"F1={r['F1']:.3f}  calls={r['calls']}  t={r['time_s']}s")
        f1_mean = round(mean(f1s), 3)
        f1_std  = round(stdev(f1s) if n_runs > 1 else 0.0, 3)
        r_mean  = round(mean(rs),  3)
        r_std   = round(stdev(rs)  if n_runs > 1 else 0.0, 3)
        n_mean  = round(mean(ns),  1)
        c_mean  = round(mean(calls_list), 1)
        print(f"    → F1={f1_mean}±{f1_std}  R={r_mean}±{r_std}  "
              f"n≈{n_mean}  calls≈{c_mean}")
        summary[label] = {
            "F1_mean": f1_mean, "F1_std": f1_std,
            "R_mean":  r_mean,  "R_std":  r_std,
            "n_mean":  n_mean,  "calls_mean": c_mean,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--query",    default="T2q5,T5q2")
    parser.add_argument("--n-runs",   type=int, default=3)
    parser.add_argument("--closures", default=None)
    args = parser.parse_args()

    llm     = MockLLM() if args.dry_run else AzureOpenAIClient()
    queries = [q.strip() for q in args.query.split(",")]
    cf      = ([c.strip() for c in args.closures.split(",")]
               if args.closures else None)

    # ── Load existing results ─────────────────────────────────────────────────
    out_path = ROOT / "experiments/results_rq3_final.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"[info] Loaded existing results from {out_path}")
    else:
        existing = {}

    # ── Run new queries ───────────────────────────────────────────────────────
    for qkey in queries:
        if qkey not in QUERY_RUNNERS:
            print(f"[skip] {qkey}"); continue
        existing[qkey] = run_n(qkey, llm, n_runs=args.n_runs, closures_filter=cf)

    # ── Save merged ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  Merged → {out_path}")

    # ── Final table — all keys ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("FULL SUMMARY (all queries)")
    print(f"{'='*72}")
    print(f"  {'Query':<8} {'Closure':<16} {'F1':>8} {'±':>5} "
          f"{'R':>7} {'±':>5} {'n':>6} {'calls':>6}")
    print(f"  {'-'*65}")
    for qkey, summ in existing.items():
        for label, s in summ.items():
            print(f"  {qkey:<8} {label:<16} "
                  f"{s['F1_mean']:>8.3f} {s['F1_std']:>5.3f} "
                  f"{s['R_mean']:>7.3f} {s['R_std']:>5.3f} "
                  f"{s['n_mean']:>6.1f} {s['calls_mean']:>6.1f}")