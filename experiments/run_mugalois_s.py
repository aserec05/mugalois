"""
experiments/run_mugalois_s.py
==============================
Runs µ-Galois_S (structure & cardinality only, no confidence) on the
same representative queries as run_ablation_tau.py.

Results saved to experiments/results_s/ — does NOT overwrite
experiments/results/ (main ablation) or experiments/results_tau/.

Usage
-----
    python3 -m experiments.run_mugalois_s
    python3 -m experiments.run_mugalois_s --template T5
    python3 -m experiments.run_mugalois_s --query T5_q7
    python3 -m experiments.run_mugalois_s --dry-run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from typing import Set
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

# Same queries as run_ablation_tau.py
QUERIES_BY_TEMPLATE = {
    "T3": ["q1", "q2", "q3"],
    "T4": ["q1", "q2", "q3", "q4", "q5", "q7"],
    "T5": ["q1", "q2", "q3", "q4", "q5", "q7", "q8", "q9", "q10"],
    "T6": ["q1", "q2", "q3", "q4", "q5", "q7", "q8", "q9", "q10"],
    "T7": ["q1", "q4", "q5", "q7", "q8"],
}

N_RUNS      = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results_s"


# ── Reuse adapter from run_ablation_tau ───────────────────────────────────────

def _get_ctx_and_gt(template: str, qid: str):
    """Load QueryContext and ground truth — identical to run_ablation_tau."""
    from run_ablation_tau import _get_ctx_and_gt as _get
    return _get(template, qid)


# ── TrackingLLM ───────────────────────────────────────────────────────────────

class TrackingLLM(BaseLLM):
    def __init__(self, llm):
        self._llm         = llm
        self.total_tokens = 0
        self.total_time_s = 0.0

    def reset(self):
        self.total_tokens = 0
        self.total_time_s = 0.0

    def chat(self, messages):
        resp = self._llm.chat(messages)
        self.total_tokens += resp.usage_tokens
        self.total_time_s += resp.latency_s
        return resp


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, n, tracker, ctx_factory):
    scores = []
    last   = None
    for i in range(n):
        tracker.reset()
        ctx_i  = ctx_factory()
        values = {str(v) for v in run_fn(ctx_i)}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s  = tracker.total_time_s
        score.tokens  = tracker.total_tokens
        print(f"      run {i+1}/{n}  "
              f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}"
              f"  t={score.time_s:.1f}s  tok={score.tokens}")
        if i == n - 1:
            last = values
        scores.append(score)
    return AggregatedScores.from_runs(scores), last


# ── Main ──────────────────────────────────────────────────────────────────────

def run_query(template: str, qid: str, base_llm, n_runs: int):
    from models import MuGaloisS
    tracker  = TrackingLLM(base_llm)
    ctx0, gt, q = _get_ctx_and_gt(template, qid)
    full_qid = f"{template}_{qid}"

    if template == "T6":
        p1 = ROOT / "experiments/template6/queries_t6.json"
        p2 = ROOT / "experiments/template6/queries_t6_hybrid.json"
        queries = {
            **json.loads(p1.read_text(encoding="utf-8")),
            **json.loads(p2.read_text(encoding="utf-8")),
        }
    else:
        adapters_path = {
            "T3": ROOT / "experiments/template3/queries_t3.json",
            "T4": ROOT / "experiments/template4/queries_t4.json",
            "T5": ROOT / "experiments/template5/queries_t5.json",
            "T6": ROOT / "experiments/template6/queries_t6.json",
            "T7": ROOT / "experiments/template7/queries_t7.json",
        }
        queries = json.loads(adapters_path[template].read_text(encoding="utf-8"))

    print(f"\n{'='*72}")
    print(f" µ-Galois_S | {full_qid} | {q.get('description','')}")
    print(f" GT={len(gt)}")
    print(f"{'='*72}")

    report = Report(
        query_id=f"s_{full_qid}",
        template=template,
        predicate=q.get("description", ""),
        n_expected=len(gt),
    )

    def ctx_factory():
        ctx, _, _ = _get_ctx_and_gt(template, qid)
        return ctx

    # ── µ-Galois_S only ──────────────────────────────────────────────────────
    print(f"\n  -- µ-Galois_S --")
    agg, last = _aggregate(
        lambda ctx_i: MuGaloisS(ctx_i, tracker),
        gt, n_runs, tracker, ctx_factory,
    )
    report.add("µ-Galois_S", agg, actual=last or set(), expected=gt)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"s_{full_qid}.json"
    report.save_json(out)
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser(
        description="µ-Galois_S — structure & cardinality only routing"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--template", default=None,
                        help="Single template (T3/T4/T5/T6/T7)")
    parser.add_argument("--query",    default=None,
                        help="Single query ID, e.g. T5_q7")
    parser.add_argument("--n-runs",   type=int, default=N_RUNS)
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()

    if args.query:
        parts = args.query.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"--query must be T5_q7 format, got {args.query!r}")
        to_run = [(parts[0], parts[1])]
    elif args.template:
        if args.template not in QUERIES_BY_TEMPLATE:
            raise ValueError(f"Template {args.template} not in "
                             f"{list(QUERIES_BY_TEMPLATE.keys())}")
        to_run = [(args.template, q) for q in QUERIES_BY_TEMPLATE[args.template]]
    else:
        to_run = [
            (t, q)
            for t, qs in QUERIES_BY_TEMPLATE.items()
            for q in qs
        ]

    print(f"[µ-Galois_S] {len(to_run)} queries × {N_RUNS} runs  "
          f"|  dry_run={args.dry_run}")
    print(f"  Queries : {[f'{t}_{q}' for t,q in to_run]}")
    print(f"  Results → {RESULTS_DIR}")
    print(f"  Model: µ-Galois_S")

    for template, qid in to_run:
        run_query(template, qid, base_llm, args.n_runs)

    print(f"\n[µ-Galois_S] Done. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()