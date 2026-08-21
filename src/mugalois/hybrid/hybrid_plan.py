"""
run_new_models.py
=================
Adds 3 new models to the ablation runner:

  µ-Galois_C_05   : MuGaloisC with α=0.50 (τ_high=0.50, τ_low=0.30)
  µ-Galois_FM_05  : MuGaloisFM with α=0.50 calibration
  µ-Galois_F      : FM_05 + motivational closure when estimated cardinality ≥ 40

Results saved in experiments/results_new/ (does NOT overwrite existing results)

Usage:
    python3 -m experiments.run_new_models --template T4
    python3 -m experiments.run_new_models --all
    python3 -m experiments.run_new_models --all --dry-run
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from typing import Set
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from mugalois.core.types import Environment
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

# Reuse all adapters and infrastructure from run_general
from run_general import (
    TEMPLATE_ADAPTERS as _BASE_ADAPTERS, TrackingLLM, _aggregate,
)
import copy
TEMPLATE_ADAPTERS = dict(_BASE_ADAPTERS)

# ── Fix T5 adapter : JSON uses s/p not source/predicate ──────────────────────
def _adapter_t5_fixed(queries_path):
    queries = json.loads(queries_path.read_text())
    def to_ctx(q):
        from mugalois.core.types import RecursivePattern
        pat = RecursivePattern(
            q["s"], q["p"], "?o", operator=q.get("operator", "+"),
        )
        return __import__("models").QueryContext(
            template="T5",
            nl_naive=q.get("nl_prompt_naive", q.get("nl_prompt", "")),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            pattern=pat,
            extra={"description": q.get("description", ""),
                   "max_depth":   q.get("max_depth", 300),
                   "tau_high":    q.get("tau_high", 0.50),
                   "tau_low":     q.get("tau_low",  0.35),
                   "operator":    q.get("operator", "+")},
        )
    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt

_t5_path = _BASE_ADAPTERS["T5"][0]
TEMPLATE_ADAPTERS["T5"] = (_t5_path, _adapter_t5_fixed)

# ── Fix T6 adapter : builds ChoicePath and stores in extra ───────────────────
def _adapter_t6_fixed(queries_path):
    from mugalois.choice.choice_path import ChoicePath

    # Load both T6 files and merge
    q6_base   = json.loads(queries_path.read_text())
    q6_hybrid = queries_path.parent / "queries_t6_hybrid.json"
    queries   = dict(q6_base)
    if q6_hybrid.exists():
        queries.update(json.loads(q6_hybrid.read_text()))

    def to_ctx(q):
        from models import QueryContext
        source     = q.get("source", "")
        target_var = q.get("target_var", "?b")
        direction  = q.get("direction", "forward")

        # Build branches list
        # Simple queries (q1-q6): branches from branches_nl as plain strings
        # Hybrid queries (q7-q10): branches_simple + branches_complex
        if "branches_nl" in q:
            # T6 simple: use predicate labels as branch strings
            branches = q["branches_nl"]
        else:
            # T6 hybrid: simple branches + complex stubs
            branches = list(q.get("branches_simple", []))
            # complex branches: store as dicts for MuGaloisChoice to handle
            for cb in q.get("branches_complex", []):
                branches.append(cb)

        choice_path = ChoicePath(
            source=source,
            target_var=target_var,
            branches=branches,
            direction=direction,
        )

        return QueryContext(
            template="T6",
            nl_naive=q.get("nl_prompt_naive", q.get("nl", "")),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", q.get("sparql", "")),
            extra={
                "description": q.get("description", ""),
                "choice_path": choice_path,
                "tau_high": 0.60, "tau_low": 0.40,
            },
        )

    gt = lambda q: set(q["ground_truth"])
    return list(queries.keys()), queries, to_ctx, gt

_t6_path = _BASE_ADAPTERS["T6"][0]
TEMPLATE_ADAPTERS["T6"] = (_t6_path, _adapter_t6_fixed)

# ── Fix T7 adapter : use from_query_json (now we have the source) ─────────────
def _adapter_t7_fixed(queries_path):
    queries = json.loads(queries_path.read_text())

    def to_ctx(q):
        from models import QueryContext
        from mugalois.hybrid.hybrid_plan import from_query_json
        plan = from_query_json(q)
        return QueryContext(
            template="T7",
            nl_naive=q.get("nl", ""),
            nl_precise=q.get("nl_precise", ""),
            sparql=q.get("sparql_semantic", ""),
            extra={
                "description": q.get("description", ""),
                "plan":        plan,
                "source":      q.get("source", ""),
                "target_var":  q.get("target_var", "?x"),
            },
        )
    gt = lambda q: set(q.get("ground_truth", []))
    return list(queries.keys()), queries, to_ctx, gt

_t7_path = _BASE_ADAPTERS["T7"][0]
TEMPLATE_ADAPTERS["T7"] = (_t7_path, _adapter_t7_fixed)

N_RUNS      = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results_new"

# ── α=0.50 thresholds ────────────────────────────────────────────────────────
ALPHA     = 0.50
TAU_HIGH  = ALPHA
TAU_LOW   = max(0, ALPHA - 0.20)
TAU_SCAN  = ALPHA
TAU_SIMPLE = ALPHA


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 : µ-Galois_C_05
# MuGaloisC with α=0.50 everywhere — no structural signal, only confidence
# ═══════════════════════════════════════════════════════════════════════════════

def _mugalois_c05(ctx, tracker) -> Set[str]:
    from models import MuGaloisC
    # Patch thresholds via a thin wrapper
    import models as _m
    _orig_tau_s = getattr(_m, "TAU_SCAN_C",  0.70)
    _orig_tau_p = getattr(_m, "TAU_SIMPLE_C", 0.80)
    # MuGaloisC already reads tau_high/tau_low from ctx.extra for T4-T6
    ctx.extra["tau_high"]   = TAU_HIGH
    ctx.extra["tau_low"]    = TAU_LOW
    ctx.extra["tau_scan"]   = TAU_SCAN
    ctx.extra["tau_simple"] = TAU_SIMPLE
    ctx.extra["use_structure"] = False  # C = confidence only
    return MuGaloisC(ctx, tracker)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 : µ-Galois_FM_05
# MuGaloisFM (no motivational) with α=0.50 calibration
# ═══════════════════════════════════════════════════════════════════════════════

def _mugalois_fm05(ctx, tracker) -> Set[str]:
    from models import MuGaloisFM
    ctx.extra["tau_high"]   = TAU_HIGH
    ctx.extra["tau_low"]    = TAU_LOW
    ctx.extra["tau_scan"]   = TAU_SCAN
    ctx.extra["tau_simple"] = TAU_SIMPLE
    return MuGaloisFM(ctx, tracker)


# ── Model registry ────────────────────────────────────────────────────────────

NEW_MODELS = {
    "mugalois_c05":  (_mugalois_c05,   "µ-Galois_C_05"),
    "mugalois_fm05": (_mugalois_fm05,  "µ-Galois_FM_05"),
}

ALL_MODELS = {**NEW_MODELS}

# results_tau/ directory — C_05 already computed there at α=0.50
RESULTS_TAU = Path(__file__).resolve().parent / "results_tau"


def _c05_already_computed(template_name: str, qid: str) -> bool:
    """Check if C_05 (α=0.50) already computed — tau files are tau_Tx_qx.json"""
    f = RESULTS_TAU / f"tau_{template_name}_{qid}.json"
    return f.exists()


# ── Runner ────────────────────────────────────────────────────────────────────

def run_template(template_name, base_llm, query_filter=None,
                 model_filter=None, n_runs=N_RUNS):
    if template_name not in TEMPLATE_ADAPTERS:
        print(f"[skip] {template_name} not configured"); return

    queries_path, adapter_fn = TEMPLATE_ADAPTERS[template_name]
    if not queries_path.exists():
        print(f"[skip] {queries_path} not found"); return

    query_ids, queries, to_ctx, gt_fn = adapter_fn(queries_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Only run queries that already exist in results/
    existing_results = Path(__file__).resolve().parent / "results"
    existing_qids = set()
    for f in existing_results.glob(f"{template_name}_q*_ablation.json"):
        m2 = re.match(rf"{template_name}_(q[\w]+)_ablation", f.stem)
        if m2: existing_qids.add(m2.group(1))
    if existing_qids:
        query_ids = [q for q in query_ids if q in existing_qids]
        print(f"  [filter] existing queries: {sorted(query_ids)}")
    else:
        print(f"  [warn] no existing results for {template_name}, running all")

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
        print(f" {template_name} / {qid} | {q.get('description','')}")
        print(f" GT={len(gt)}")
        print(f"{'='*72}")

        report = Report(
            query_id=f"{template_name}_{qid}",
            template=template_name,
            predicate=q.get("description", ""),
            n_expected=len(gt),
        )

        for model_key, (model_fn, display_name) in models_to_run.items():
            # C_05 : skip if already computed in results_tau/ at α=0.50
            if model_key == "mugalois_c05" and _c05_already_computed(template_name, qid):
                print(f"  -- {display_name} -- [skip: already in results_tau]")
                continue
            print(f"\n  -- {display_name} --")
            ctx_factory = lambda q_=q: to_ctx(q_)
            fn  = lambda ctx_i, mf=model_fn: mf(ctx_i, tracker)
            agg, last = _aggregate(fn, gt, n_runs, tracker,
                                   ctx_factory=ctx_factory)
            report.add(display_name, agg, actual=last or set(), expected=gt)

        report.print_table()
        out = RESULTS_DIR / f"{template_name}_{qid}_ablation.json"
        report.save_json(out)
        print(f"  Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=None)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--query",    default=None)
    parser.add_argument("--model",    default=None,
                        help=f"Comma-sep from: {list(ALL_MODELS.keys())}")
    parser.add_argument("--n-runs",   type=int, default=N_RUNS)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    templates = (list(TEMPLATE_ADAPTERS.keys()) if args.all
                 else ([args.template] if args.template
                       else list(TEMPLATE_ADAPTERS.keys())))
    query_filter = [args.query] if args.query else None
    model_filter = (None if (args.model is None or args.model == "all")
                    else [m.strip() for m in args.model.split(",")])

    print(f"[new models] templates={templates}  models={model_filter or 'all'}"
          f"  n_runs={args.n_runs}  dry_run={args.dry_run}")
    print(f"  ALPHA={ALPHA}  TAU_HIGH={TAU_HIGH}  TAU_LOW={TAU_LOW}")

    for t in templates:
        run_template(t, base_llm,
                     query_filter=query_filter,
                     model_filter=model_filter,
                     n_runs=args.n_runs)

def from_query_json(q: dict) -> "HybridPlan":
    from mugalois.hybrid.hybrid_plan import HopNode, ClosureNode, FilterNode, ChoiceNode, HybridPlan
    from mugalois.core.types import TriplePattern
    nodes = []
    for n in q.get("nodes", []):
        t = n["type"]
        if t == "hop":
            nodes.append(HopNode(predicate=n["predicate"], inverse=n.get("inverse", False), s=n.get("s",""), o=n.get("o","")))
        elif t == "closure":
            nodes.append(ClosureNode(predicate=n["predicate"], operator=n.get("operator","+"), inverse=n.get("inverse", False), s=n.get("s",""), o=n.get("o","")))
        elif t == "filter":
            nodes.append(FilterNode(predicate=n.get("predicate","rdf:type"), value=n["value"], nl_type=n.get("nl_type", n["value"])))
        elif t == "choice":
            nodes.append(ChoiceNode(branches=n["branches"]))
    where_triples = [TriplePattern(s=wt.get("s","?x"), p=wt.get("p",""), o=wt.get("o","")) for wt in q.get("where_triples", [])]
    return HybridPlan(nodes=nodes, source=q.get("source","?x"), target=q.get("target") or q.get("anchor","?target"), target_var=q.get("target_var","?x"), anchor_side=q.get("anchor_side","right"), result_node_idx=q.get("result_node_idx",0), nl=q.get("nl",""), expression=q.get("expression",""), gt_size=q.get("gt_size",-1), where_triples=where_triples)
