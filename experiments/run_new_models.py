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
# ── Monkey patch : use run_general adapters as-is, just patch taus ────────────
TEMPLATE_ADAPTERS = dict(_BASE_ADAPTERS)

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

def _patch_taus(ctx):
    """Inject α=0.50 thresholds into ctx.extra — works for all templates."""
    ctx.extra["tau_high"]   = TAU_HIGH
    ctx.extra["tau_low"]    = TAU_LOW
    ctx.extra["tau_scan"]   = TAU_SCAN
    ctx.extra["tau_simple"] = TAU_SIMPLE


def _mugalois_c05(ctx, tracker) -> Set[str]:
    from models import MuGaloisC
    _patch_taus(ctx)
    ctx.extra["use_structure"] = False
    return MuGaloisC(ctx, tracker)


def _mugalois_fm05(ctx, tracker) -> Set[str]:
    from models import MuGaloisFM
    _patch_taus(ctx)
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