"""
experiments/run_rq3_coverage.py
================================
RQ3 coverage ablation — tests µ-Galois_F with different coverage thresholds.

Does NOT overwrite experiments/results/.
Results → experiments/results_coverage/

Variants:
  F_cov050 : coverage=0.50 (very aggressive — activates almost always)
  F_cov065 : coverage=0.65 (calibrated)
  F_cov080 : coverage=0.80 (default _F — reference)
  F_cov095 : coverage=0.95 (conservative — almost never activates)

Usage
-----
    python3 -m experiments.run_rq3_coverage
    python3 -m experiments.run_rq3_coverage --template T5
    python3 -m experiments.run_rq3_coverage --query q2
    python3 -m experiments.run_rq3_coverage --dry-run
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

N_RUNS      = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results_coverage"

# Coverage variants to test
COVERAGE_VARIANTS = {
    "F_cov050": 0.50,
    "F_cov065": 0.65,
    "F_cov080": 0.80,   # default _F
    "F_cov095": 0.95,
}

# Templates and queries to run — same as main ablation
TEMPLATES = {
    "T1": ROOT / "experiments/template1/queries.json",
    "T2": ROOT / "experiments/template2/queries.json",
    "T3": ROOT / "experiments/template3/queries_t3.json",
    "T4": ROOT / "experiments/template4/queries_t4.json",
    "T5": ROOT / "experiments/template5/queries_t5.json",
    "T6": ROOT / "experiments/template6/queries_t6.json",
    "T7": ROOT / "experiments/template7/queries_t7.json",
}

# ── TrackingLLM ───────────────────────────────────────────────────────────────

class TrackingLLM(BaseLLM):
    def __init__(self, llm):
        self._llm = llm; self.total_tokens = 0; self.total_time_s = 0.0
    def reset(self): self.total_tokens = 0; self.total_time_s = 0.0
    def chat(self, messages):
        resp = self._llm.chat(messages)
        self.total_tokens += resp.usage_tokens
        self.total_time_s += resp.latency_s
        return resp

# ── Coverage-aware _F runner ──────────────────────────────────────────────────

def run_f_with_coverage(ctx, llm, coverage: float):
    """
    Run µ-Galois_F with a specific coverage threshold.
    Patches COVERAGE_THRESHOLD in simple_rec and COVERAGE in mugalois_multi
    temporarily — restores originals after the call.
    """
    import mugalois.rec.simple_rec as _rec
    import mugalois.paths.mugalois_multi as _multi
    import mugalois.choice.mugalois_choice as _choice

    orig_rec    = _rec.COVERAGE_THRESHOLD
    orig_multi  = _multi.COVERAGE
    orig_choice = _choice.COVERAGE

    _rec.COVERAGE_THRESHOLD = coverage
    _multi.COVERAGE         = coverage
    _choice.COVERAGE        = coverage

    try:
        from models import MuGaloisF
        return MuGaloisF(ctx, llm)
    finally:
        _rec.COVERAGE_THRESHOLD = orig_rec
        _multi.COVERAGE         = orig_multi
        _choice.COVERAGE        = orig_choice

# ── Reuse adapters from run_ablation ─────────────────────────────────────────

def _get_adapter(template, queries_path):
    """Import adapter from run_ablation without re-running main."""
    # We inline a simplified version to avoid circular imports
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    if template in ("T1", "T2"):
        def to_ctx(q):
            from mugalois.core.types import TriplePattern, Environment, ConditionIN, Condition
            from mugalois.core.rewrite import RW1
            from models import QueryContext
            raw = q.get("T1a", q).get("pattern_nl", q.get("pattern_nl", {}))
            triple = TriplePattern(s=raw["s"], p=raw["p"], o=raw["o"])
            NUMERIC = {">","<",">=","<=","!="}
            in_conds, num_conds = [], []
            for c in q.get("conditions", []):
                if c["type"] == "IN":
                    in_conds.append(ConditionIN(c["var"], frozenset(c["values"])))
                elif c["type"] in NUMERIC:
                    num_conds.append(Condition(c["var"], c["type"], c["val"]))
            env, gamma = Environment(), Environment()
            if in_conds: env, gamma = RW1(frozenset(in_conds), env, gamma)
            rv = raw["o"] if raw["o"].startswith("?") else raw["s"]
            return QueryContext(
                template=template, triple=triple, env=env, gamma=gamma,
                nl_naive=q.get("nl_prompt",""), nl_precise=q.get("nl_prompt_precise",""),
                sparql=q.get("sparql_prompt",""),
                extra={"return_var": rv, "post_filter_conds": num_conds},
            )
        gt_fn = lambda q: set(q["ground_truth"])

    elif template == "T3":
        def to_ctx(q):
            from models import QueryContext
            raw_dir = q.get("optimal_direction","left").lower()
            direction = {"gd":"left","dg":"right","join":"join",
                         "left":"left","right":"right"}.get(raw_dir,"left")
            return QueryContext(template="T3", s=q["s"], p1=q["p1"], p2=q["p2"], t=q["t"],
                                nl_naive=q.get("nl_prompt",""), nl_precise=q.get("nl_prompt_precise",""),
                                sparql=q.get("sparql_prompt",""),
                                extra={"direction": direction})
        gt_fn = lambda q: set(q["ground_truth"])

    elif template == "T4":
        def to_ctx(q):
            from models import QueryContext
            from mugalois.paths.path_query import PathQuery
            from mugalois.core.types import Condition
            path = PathQuery.from_chain(q["anchors"], q["predicates"], q["target_var"])
            labels = {}
            for c in q.get("conditions", []):
                path.conditions.append(Condition(var=c["var"], op=c["op"], val=c["val"]))
                if "label" in c: labels[c["var"]] = c["label"]
            path.condition_labels = labels
            return QueryContext(template="T4", path=path,
                                nl_naive=q.get("nl_prompt_naive",""),
                                nl_precise=q.get("nl_prompt_precise",""),
                                sparql=q.get("sparql_prompt",""),
                                extra={"description": q.get("description","")})
        gt_fn = lambda q: set(q["ground_truth"])

    elif template == "T5":
        def to_ctx(q):
            from models import QueryContext
            from mugalois.core.types import RecursivePattern
            pat = RecursivePattern(s=q["s"], p=q["p"], o=q.get("o","?b"),
                                   operator=q.get("operator","+"))
            return QueryContext(template="T5", pattern=pat,
                                nl_naive=q.get("nl_prompt_naive",""),
                                nl_precise=q.get("nl_prompt_precise",""),
                                sparql=q.get("sparql_prompt",""),
                                extra={"max_depth": q.get("max_depth",300)})
        gt_fn = lambda q: set(q["ground_truth"])

    elif template == "T6":
        def to_ctx(q):
            from models import QueryContext
            return QueryContext(template="T6", nl_naive=q.get("nl",""),
                                branches=q.get("branches",[]),
                                extra={"tau_high":0.60,"tau_low":0.40})
        gt_fn = lambda q: set(q["ground_truth"])

    elif template == "T7":
        def to_ctx(q):
            from models import QueryContext
            from mugalois.hybrid.hybrid_plan import from_query_json
            plan = from_query_json(q)
            return QueryContext(template="T7", nl_naive=q.get("nl",""),
                                sparql=q.get("sparql_semantic",""), plan=plan,
                                extra={"description": q.get("description","")})
        gt_fn = lambda q: set(q.get("ground_truth",[]))

    else:
        raise ValueError(f"Unknown template: {template}")

    return list(queries.keys()), queries, to_ctx, gt_fn

# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, n, tracker, ctx_factory):
    scores, last = [], None
    for i in range(n):
        tracker.reset()
        ctx_i  = ctx_factory()
        values = {str(v) for v in run_fn(ctx_i)}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s  = tracker.total_time_s
        score.tokens  = tracker.total_tokens
        print(f"      run {i+1}/{n}  "
              f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}"
              f"  n={len(values)}/GT={len(gt)}"
              f"  t={score.time_s:.1f}s  tok={score.tokens}")
        if i == n-1: last = values
        scores.append(score)
    return AggregatedScores.from_runs(scores), last

# ── Main runner ───────────────────────────────────────────────────────────────

def run_template(template, base_llm, query_filter=None, n_runs=N_RUNS):
    if template not in TEMPLATES:
        print(f"[skip] {template} not configured")
        return
    queries_path = TEMPLATES[template]
    if not queries_path.exists():
        print(f"[skip] {queries_path} not found")
        return

    query_ids, queries, to_ctx, gt_fn = _get_adapter(template, queries_path)
    tracker = TrackingLLM(base_llm)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for qid in query_ids:
        if query_filter and qid not in query_filter:
            continue
        q  = queries[qid]
        gt = gt_fn(q)

        print(f"\n{'='*72}")
        print(f" Coverage ablation | {template}/{qid} | {q.get('description','')}")
        print(f" GT={len(gt)}")
        print(f"{'='*72}")

        report = Report(
            query_id=f"{template}_{qid}",
            template=template,
            predicate=q.get("description", q.get("nl_prompt","")),
            n_expected=len(gt),
        )

        for variant_name, coverage in COVERAGE_VARIANTS.items():
            print(f"\n  -- {variant_name} (coverage={coverage}) --")
            ctx_factory = lambda q_=q: to_ctx(q_)
            agg, last = _aggregate(
                lambda ctx_i, cov=coverage: run_f_with_coverage(ctx_i, tracker, cov),
                gt, n_runs, tracker, ctx_factory,
            )
            report.add(variant_name, agg, actual=last or set(), expected=gt)

        report.print_table()
        out = RESULTS_DIR / f"coverage_{template}_{qid}.json"
        report.save_json(out)
        print(f"  Saved → {out}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 coverage ablation — _F with 4 coverage thresholds"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--template", default=None,
                        help=f"Single template. Available: {list(TEMPLATES.keys())}")
    parser.add_argument("--query",    default=None, help="Single query ID, e.g. q2")
    parser.add_argument("--n-runs",   type=int, default=N_RUNS)
    args = parser.parse_args()

    base_llm  = MockLLM() if args.dry_run else AzureOpenAIClient()
    templates = [args.template] if args.template else list(TEMPLATES.keys())
    qfilter   = [args.query] if args.query else None

    print(f"[coverage ablation] templates={templates}  n_runs={args.n_runs}")
    print(f"  Variants: {list(COVERAGE_VARIANTS.items())}")
    print(f"  Results → {RESULTS_DIR}  (does NOT overwrite results/)")

    for t in templates:
        run_template(t, base_llm, query_filter=qfilter, n_runs=args.n_runs)

    print("\n[coverage ablation] Done.")

if __name__ == "__main__":
    main()
