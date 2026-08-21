"""
run_ablation.py
===============
Unified ablation runner — evaluates the 5 µ-Galois models across T1–T7.

Usage
-----
    python3 -m experiments.ablation.run_ablation --template T5
    python3 -m experiments.ablation.run_ablation --template T7 --query q4
    python3 -m experiments.ablation.run_ablation --all
    python3 -m experiments.ablation.run_ablation --dry-run

Each template adapter (see TEMPLATE_ADAPTERS below) converts the template's
query JSON into a QueryContext, reusing ALL existing runners.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # ← ajoute ça


from dotenv import load_dotenv
load_dotenv()

from typing import Set
from models import (
    QueryContext, MODELS,
    MuGaloisHol, MuGaloisDec, MuGaloisC, MuGaloisFM, MuGaloisF,
)
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

N_RUNS     = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Baseline runners ───────────────────────────────────────────────────────────

def _run_nl_naive(ctx: QueryContext, llm) -> Set[str]:
    from mugalois.core.prompts import build_value_messages, genNLPrompt
    from mugalois.core.parser import json_to_values
    nl = ctx.nl_naive or ctx.extra.get("description", "")
    resp = llm.chat(build_value_messages(genNLPrompt(nl)))
    return json_to_values(resp.text)

def _run_nl_precise(ctx: QueryContext, llm) -> Set[str]:
    from mugalois.core.prompts import build_value_messages, genNLPrompt
    from mugalois.core.parser import json_to_values
    nl = ctx.nl_precise or ctx.nl_naive or ctx.extra.get("description", "")
    resp = llm.chat(build_value_messages(genNLPrompt(nl)))
    return json_to_values(resp.text)

def _run_sparql(ctx: QueryContext, llm) -> Set[str]:
    from mugalois.core.prompts import build_value_messages, genSPARQLPrompt
    from mugalois.core.parser import json_to_values
    sparql = ctx.sparql or ctx.extra.get("sparql_prompt", "")
    if not sparql:
        return set()
    resp = llm.chat(build_value_messages(genSPARQLPrompt(sparql)))
    return json_to_values(resp.text)


# Full strategy registry: baselines + 5 µ-Galois models
ALL_MODELS = {
    "nl_naive":     (_run_nl_naive,   "NL_naive"),
    "nl_precise":   (_run_nl_precise, "NL_precise"),
    "sparql":       (_run_sparql,     "SPARQL"),
    **{k: v for k, v in MODELS.items()},
}
ALL_STRATEGIES = list(ALL_MODELS.keys())

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


# ── Template adapters ─────────────────────────────────────────────────────────
# Each adapter reads the template's query JSON and returns
# (list_of_query_ids, fn(query_dict) -> QueryContext, gt_fn(query_dict) -> set)

def _parse_conditions_t1_t2(q):
    """
    Separate T1/T2 conditions into:
    - env/gamma : IN conditions → seeds for LLMScan (via RW1)
    - num_conds : numeric conditions (>, <, !=) → post-filter in memory
    Numeric conditions must NEVER go into env as bindings —
    they are not seeds, they are filters on the output variable.
    """
    from mugalois.core.types import Environment, ConditionIN, Condition
    from mugalois.core.rewrite import RW1
    NUMERIC = {">", "<", ">=", "<=", "!="}
    in_conds, num_conds = [], []
    for c in q.get("conditions", []):
        if c["type"] == "IN":
            in_conds.append(ConditionIN(c["var"], frozenset(c["values"])))
        elif c["type"] in NUMERIC:
            num_conds.append(Condition(c["var"], c["type"], c["val"]))
    env, gamma = Environment(), Environment()
    if in_conds:
        env, gamma = RW1(frozenset(in_conds), env, gamma)
    return env, gamma, num_conds


def _adapter_t1(queries_path):
    """
    T1 JSON structure (new):
      T1a.pattern_nl: {s, p, o}   — main variant (one fixed anchor)
      T1a.sparql_prompt            — semantic SPARQL (no dbo:/dbr:)
      T1b.pattern_nl, T1b.sparql_prompt — filter variant
      nl_prompt, nl_prompt_precise, ground_truth
    Uses T1a as the primary pattern for µ-Galois models.
    """
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        from mugalois.core.types import TriplePattern
        # Use T1a as primary (fixed anchor, one free variable)
        t1a = q.get("T1a", {})
        raw = t1a.get("pattern_nl", q.get("pattern_nl", {}))
        triple = TriplePattern(s=raw["s"], p=raw["p"], o=raw["o"])
        env, gamma, num_conds = _parse_conditions_t1_t2(q)
        # return_var: whichever side is a variable
        if raw["o"].startswith("?"):
            return_var = raw["o"]
        elif raw["s"].startswith("?"):
            return_var = raw["s"]
        else:
            return_var = q.get("return_var", raw["o"])
        return QueryContext(
            template="T1",
            nl_naive=q.get("nl_prompt", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=t1a.get("sparql_prompt", q.get("sparql_prompt", "")),
            triple=triple,
            env=env,
            gamma=gamma,
            extra={
                "description":       q.get("nl_prompt", ""),
                "return_var":        return_var,
                "post_filter_conds": num_conds,
            },
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


def _adapter_t3(queries_path):
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        raw_dir   = q.get("optimal_direction", "left").lower()
        # Normalise: GD→left, DG→right, join→join
        direction = {"gd": "left", "dg": "right", "join": "join",
                     "left": "left", "right": "right"}.get(raw_dir, "left")
        # "join" maps to the Join strategy in models.py
        return QueryContext(
            template="T3",
            nl_naive=q.get("nl_prompt", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            s=q["s"],
            p1=q["p1"],
            p2=q["p2"],
            t=q["t"],
            extra={
                "description": q.get("description", ""),
                "direction":   direction,
            },
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


def _adapter_t4(queries_path):
    """
    T4 JSON structure:
      anchors: [source, ?intermediate..., target]
      predicates: [p1, p2, ...]
      target_var: "?b"
      conditions: [{var, op, val}]  — optional numeric/equality
      nl_prompt_naive, nl_prompt_precise, sparql_prompt
    Uses PathQuery.from_chain to build the path object.
    """
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        from mugalois.paths.path_query import PathQuery
        from mugalois.core.types import Condition, Environment
        path = PathQuery.from_chain(
            q["anchors"], q["predicates"], q["target_var"]
        )
        # Attach conditions
        labels = {}
        for c in q.get("conditions", []):
            path.conditions.append(
                Condition(var=c["var"], op=c["op"], val=c["val"])
            )
            if "label" in c:
                labels[c["var"]] = c["label"]
        path.condition_labels = labels  # type: ignore[attr-defined]
        return QueryContext(
            template="T4",
            nl_naive=q.get("nl_prompt_naive", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            path=path,
            extra={"description": q.get("description", "")},
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


def _adapter_t5(queries_path):
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        from mugalois.core.types import RecursivePattern
        pat = RecursivePattern(
            s=q["source"], predicate=q["predicate"],
            o="?o", operator=q.get("operator", "+"),
        )
        return QueryContext(
            template="T5",
            nl_naive=q.get("nl_prompt_naive", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            pattern=pat,
            operator=q.get("operator", "+"),
            extra={"description":  q.get("description", ""),
                   "max_depth":    q.get("max_depth", 300),
                   "tau_high":     q.get("tau_high", 0.50),
                   "tau_low":      q.get("tau_low",  0.35)},
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


def _adapter_t6(queries_path):
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        return QueryContext(
            template="T6",
            nl_naive=q.get("nl", ""),
            branches=q.get("branches", []),
            extra={"description": q.get("description", ""),
                   "tau_high": 0.60, "tau_low": 0.40},
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


def _adapter_t7(queries_path):
    from mugalois.hybrid.hybrid_plan import from_query_json
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        plan = from_query_json(q)
        return QueryContext(
            template="T7",
            nl_naive=q.get("nl", ""),
            sparql=q.get("sparql_semantic", ""),
            plan=plan,
            extra={"description": q.get("description", "")},
        )
    gt = lambda q: set(q.get("ground_truth", []))
    return list(queries.keys()), queries, to_ctx, gt



def _adapter_t2(queries_path):
    """
    T2 JSON structure: same as T1.
    IN conditions → env (seeds), numeric conditions → post-filter.
    """
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        from mugalois.core.types import TriplePattern
        raw    = q["pattern_nl"]
        triple = TriplePattern(s=raw["s"], p=raw["p"], o=raw["o"])
        env, gamma, num_conds = _parse_conditions_t1_t2(q)
        return_var = q.get("return_var",
                           raw["o"] if raw["o"].startswith("?") else raw["s"])
        return QueryContext(
            template="T2",
            nl_naive=q.get("nl_prompt", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            triple=triple,
            env=env,
            gamma=gamma,
            extra={
                "description":       q.get("description", q.get("nl_prompt", "")),
                "return_var":        return_var,
                "post_filter_conds": num_conds,
            },
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt


TEMPLATE_ADAPTERS = {
    "T1": (ROOT / "experiments/template1/queries.json",         _adapter_t1),
    "T2": (ROOT / "experiments/template2/queries.json",         _adapter_t2),
    "T3": (ROOT / "experiments/template3/queries_t3.json",      _adapter_t3),
    "T4": (ROOT / "experiments/template4/queries_t4.json",      _adapter_t4),
    "T5": (ROOT / "experiments/template5/queries_t5.json",      _adapter_t5),
    "T6": (ROOT / "experiments/template6/queries_t6.json",      _adapter_t6),
    "T7": (ROOT / "experiments/template7/queries_t7.json",      _adapter_t7),
}


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, n, tracker, ctx_factory=None):
    """
    ctx_factory: callable () -> QueryContext, called before each run.
    Ensures env/gamma are fresh (not mutated by updateEnv from previous run).
    """
    scores = []
    last   = None
    null_q = len(gt) == 0
    for i in range(n):
        tracker.reset()
        ctx_i  = ctx_factory() if ctx_factory is not None else None
        values = {str(v) for v in (run_fn(ctx_i) if ctx_i is not None else run_fn())}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s  = tracker.total_time_s
        score.tokens  = tracker.total_tokens
        tag = " [null]" if null_q else ""
        print(f"      run {i+1}/{n}  "
              f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}"
              f"  returned={len(values)}  expected={len(gt)}{tag}"
              f"  t={score.time_s:.1f}s  tok={score.tokens}")
        if i == n - 1:
            last = values
        scores.append(score)
    return AggregatedScores.from_runs(scores), last


# ── Main ──────────────────────────────────────────────────────────────────────

def run_template(template_name, base_llm, query_filter=None,
                 model_filter=None, n_runs=N_RUNS, verbose=False):
    if template_name not in TEMPLATE_ADAPTERS:
        print(f"[skip] {template_name} not configured")
        return

    queries_path, adapter_fn = TEMPLATE_ADAPTERS[template_name]
    if not queries_path.exists():
        print(f"[skip] {queries_path} not found")
        return

    query_ids, queries, to_ctx, gt_fn = adapter_fn(queries_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    models_to_run = {
        k: v for k, v in ALL_MODELS.items()
        if model_filter is None or k in model_filter
    }

    for qid in query_ids:
        if query_filter and qid not in query_filter:
            continue

        q   = queries[qid]
        gt  = gt_fn(q)
        ctx = to_ctx(q)

        print(f"\n{'='*72}")
        print(f" {template_name} / {qid} | {q.get('description', '')}")
        print(f" GT={len(gt)}")
        print(f" Models: {list(models_to_run.keys())}")
        print(f"{'='*72}")

        report = Report(
            query_id=f"{template_name}_{qid}",
            template=template_name,
            predicate=q.get("description", ""),
            n_expected=len(gt),
        )

        for model_key, (model_fn, display_name) in models_to_run.items():
            print(f"\n  -- {display_name} --")
            # ctx_factory rebuilds env/gamma fresh each run — prevents
            # updateEnv mutation from leaking across runs (T2 q2 bug).
            ctx_factory = lambda q_=q: to_ctx(q_)
            fn  = lambda ctx_i, mf=model_fn: mf(ctx_i, tracker)
            agg, last = _aggregate(fn, gt, n_runs, tracker, ctx_factory=ctx_factory)
            report.add(display_name, agg, actual=last or set(), expected=gt)

        report.print_table()
        out = RESULTS_DIR / f"{template_name}_{qid}_ablation.json"
        report.save_json(out)
        print(f"  Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="µ-Galois ablation — 5 models across T1–T7"
    )
    parser.add_argument("--template",  default=None,
                        help="Single template, e.g. T5")
    parser.add_argument("--all",       action="store_true",
                        help="Run all templates")
    parser.add_argument("--query",     default=None,
                        help="Single query ID, e.g. q1")
    parser.add_argument("--model",     default=None,
                        help=f"Comma-separated or 'all'. Available: {chr(10).join(ALL_STRATEGIES)}")
    parser.add_argument("--n-runs",    type=int, default=N_RUNS)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()

    templates = list(TEMPLATE_ADAPTERS.keys()) if args.all \
                else ([args.template] if args.template else list(TEMPLATE_ADAPTERS.keys()))

    query_filter = [args.query] if args.query else None
    model_filter = (
        None if (args.model is None or args.model == "all")
        else [m.strip() for m in args.model.split(",")]
    )

    print(f"[ablation] templates={templates}  models={model_filter or 'all'}"
          f"  n_runs={args.n_runs}  dry_run={args.dry_run}")

    for t in templates:
        run_template(t, base_llm,
                     query_filter=query_filter,
                     model_filter=model_filter,
                     n_runs=args.n_runs,
                     verbose=args.verbose)