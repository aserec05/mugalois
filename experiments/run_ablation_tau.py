"""
experiments/run_ablation_tau.py
================================
Ablation study of τ_high / τ_low on µ-Galois_C (confidence-only routing).

µ-Galois_C is the only model where τ_high and τ_low directly control
the routing decision without interference from structural or cardinality signals.
Varying τ on this model isolates the contribution of the confidence threshold.

Queries tested (representative subset, 3 templates × 3-4 queries):
  T3 : q1 (GD optimal), q2 (DG optimal), q3 (Join)
  T4 : q1 (2-hop), q3 (3-hop inter.), q5 (4-hop)
  T5 : q7 (49-hop chain), q9 (DAG), q2 (264 backward)
  T6 : q1 (similar branches), q9 (distinct branches)

τ grid (7 valid combinations, τ_low < τ_high):
  τ_high ∈ {0.30, 0.50, 0.70}
  τ_low  ∈ {0.20, 0.35, 0.50}
  → (0.30,0.20) (0.50,0.20) (0.50,0.35) (0.70,0.20) (0.70,0.35) (0.70,0.50)
  + current default (0.50, 0.35) already covered above

Usage
-----
    python3 -m experiments.run_ablation_tau
    python3 -m experiments.run_ablation_tau --dry-run
    python3 -m experiments.run_ablation_tau --template T5
    python3 -m experiments.run_ablation_tau --query T5_q7
"""
from __future__ import annotations
import argparse
import json
import copy
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

# ── τ grid ────────────────────────────────────────────────────────────────────

# α = global LLM trust parameter.
# A single α encodes how much we trust the LLM to answer holistically.
# High α → prefer holistic; low α → prefer decomposition.
# All internal τ thresholds are derived from α — zero extra LLM calls.
ALPHA_GRID = [0.20, 0.35, 0.50, 0.60, 0.70, 0.80]
# Default α corresponds to the current system defaults
ALPHA_DEFAULT = 0.50


DELTA = 0.20   # zone grise width: τ_low = max(0, α - Δ)


def alpha_to_taus(alpha: float) -> dict:
    """
    Derive all internal τ thresholds from a single trust parameter α.

    α encodes the global LLM confidence personality:
      α → 1.0 : trust the LLM, prefer holistic prompts
      α → 0.0 : distrust the LLM, prefer decomposition

    For binary decisions (single threshold):
      τ = α

    For three-way decisions (τ_high / τ_low zone):
      τ_high = α
      τ_low  = max(0, α - Δ)   with Δ = 0.20

    This preserves the current system defaults at α = 0.50:
      T5 MuGaloisRec  : τ_high=0.50, τ_low=0.30  (default: 0.50 / 0.35 — close)
      T6 MuGaloisChoice: τ_high=0.50, τ_low=0.30  (default: 0.60 / 0.40 — close)
      T4 MuGaloisMultiPath: τ_high=0.50, τ_low=0.30 (default: 0.50 / 0.35 — close)

    tau_C = α * 0.9 : condition inject vs post-filter — slightly more conservative
                      (injecting a condition into the prompt risks hallucination).
    """
    tau_low = round(max(0.0, alpha - DELTA), 3)
    return {
        "tau_scan":        alpha,
        "tau_C":           round(alpha * 0.9, 3),
        "tau_simple":      alpha,               # T3 : holistic vs GD/DG/Join
        "tau_chain_high":  alpha,               # T4 : holistic vs chain (high)
        "tau_chain_low":   tau_low,             # T4 : chain vs decomposed (low)
        "tau_rec_high":    alpha,               # T5 : holistic vs fixpoint (high)
        "tau_rec_low":     tau_low,             # T5 : fixpoint vs forced (low)
        "tau_choice_high": alpha,               # T6 : holistic OR vs parallel (high)
        "tau_choice_low":  tau_low,             # T6 : parallel vs forced (low)
        "tau_hybrid":      alpha,               # T7 : holistic steps vs full planner
    }

# ── Representative queries ────────────────────────────────────────────────────

QUERIES_BY_TEMPLATE = {
    "T3": ["q1", "q2", "q3"],
    "T4": ["q1", "q2", "q3", "q4", "q5", "q7"],
    "T5": ["q1", "q2", "q3", "q4", "q5", "q7", "q8", "q9", "q10"],
    "T6": ["q1", "q2", "q3", "q4", "q5", "q7", "q8", "q9", "q10"],
    "T7": ["q1", "q4", "q5", "q7", "q8"],
}

N_RUNS     = 5
RESULTS_DIR = Path(__file__).resolve().parent / "results_tau"


# ── Adapters (reuse from run_general) ─────────────────────────────────────────

def _get_ctx_and_gt(template: str, qid: str):
    """Load QueryContext and ground truth for a single query."""
    from models import QueryContext

    adapters_path = {
        "T3": ROOT / "experiments/template3/queries_t3.json",
        "T4": ROOT / "experiments/template4/queries_t4.json",
        "T5": ROOT / "experiments/template5/queries_t5.json",
        "T6": ROOT / "experiments/template6/queries_t6.json",
        "T6H": ROOT / "experiments/template6/queries_t6_hybrid.json",
        "T7": ROOT / "experiments/template7/queries_t7.json",
    }

    # T6 queries may be split across two files (simple: q1-q6, hybrid: q7-q10)
    if template == "T6":
        path_simple = adapters_path["T6"]
        path_hybrid = adapters_path["T6H"]
        q_simple = json.loads(path_simple.read_text(encoding="utf-8")) if path_simple.exists() else {}
        q_hybrid = json.loads(path_hybrid.read_text(encoding="utf-8")) if path_hybrid.exists() else {}
        queries = {**q_simple, **q_hybrid}
        if qid not in queries:
            raise KeyError(f"Query {qid} not found in T6 files. "
                           f"Available: {list(queries.keys())}")
    else:
        if template not in adapters_path:
            raise ValueError(f"Template {template} not supported. "
                             f"Supported: {list(adapters_path.keys())}")
        queries_path = adapters_path[template]
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        if qid not in queries:
            raise KeyError(f"Query {qid} not found in {queries_path.name}. "
                           f"Available: {list(queries.keys())}")

    q = queries[qid]

    # ── T3 ──
    if template == "T3":
        raw_dir   = q.get("optimal_direction", "left").lower()
        direction = {"gd": "left", "dg": "right", "join": "join",
                     "left": "left", "right": "right"}.get(raw_dir, "left")
        ctx = QueryContext(
            template="T3",
            nl_naive=q.get("nl_prompt", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            s=q["s"], p1=q["p1"], p2=q["p2"], t=q["t"],
            extra={"description": q.get("description", ""), "direction": direction},
        )

    # ── T4 ──
    elif template == "T4":
        from mugalois.paths.path_query import PathQuery
        from mugalois.core.types import Condition
        path = PathQuery.from_chain(q["anchors"], q["predicates"], q["target_var"])
        labels = {}
        for c in q.get("conditions", []):
            path.conditions.append(Condition(var=c["var"], op=c["op"], val=c["val"]))
            if "label" in c:
                labels[c["var"]] = c["label"]
        path.condition_labels = labels
        ctx = QueryContext(
            template="T4",
            nl_naive=q.get("nl_prompt_naive", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            path=path,
            extra={"description": q.get("description", "")},
        )

    # ── T5 ──
    elif template == "T5":
        from mugalois.core.types import RecursivePattern
        pat = RecursivePattern(
            s=q["s"], p=q["p"],
            o=q.get("o", "?b"), operator=q.get("operator", "+"),
        )
        ctx = QueryContext(
            template="T5",
            nl_naive=q.get("nl_prompt_naive", ""),
            nl_precise=q.get("nl_prompt_precise", ""),
            sparql=q.get("sparql_prompt", ""),
            pattern=pat,
            extra={"description": q.get("description", ""),
                   "operator":   q.get("operator", "+"),
                   "max_depth":  q.get("max_depth", 300)},
        )

    # ── T6 ──
    elif template == "T6":
        from mugalois.choice.choice_path import ChoicePath
        simple   = q.get("branches_simple", q.get("branches_nl", []))
        complex_ = []
        for b in q.get("branches_complex", []):
            if b["type"] == "T4":
                from mugalois.paths.path_query import PathQuery
                hops    = b["hops"]
                anchors = ([q["source"]]
                           + [f"?inter{i}" for i in range(len(hops)-1)]
                           + ["?b"])
                complex_.append(PathQuery.from_chain(anchors, hops, "?b"))
        cp = ChoicePath(
            source=q.get("source",""),
            target_var=q.get("target_var","?b"),
            branches=simple + complex_,
            direction=q.get("direction","forward"),
        )
        ctx = QueryContext(
            template="T6",
            nl_naive=q.get("nl_prompt_naive", q.get("nl_prompt","")),
            nl_precise=q.get("nl_prompt_precise",""),
            sparql=q.get("sparql_prompt",""),
            extra={"description": q.get("description",""), "choice_path": cp},
        )

    # ── T7 ──
    elif template == "T7":
        from mugalois.hybrid.hybrid_plan import from_query_json
        plan = from_query_json(q)
        ctx = QueryContext(
            template="T7",
            nl_naive=q.get("nl", ""),
            sparql=q.get("sparql_semantic", ""),
            plan=plan,
            extra={"description": q.get("description", "")},
        )

    gt = set(q.get("ground_truth", []))
    return ctx, gt, q


# ── µ-Galois_C with variable τ ────────────────────────────────────────────────

def run_mugalois_c_alpha(ctx, llm, alpha: float) -> Set[str]:
    """
    µ-Galois_C with a single trust parameter α.
    All internal τ thresholds are derived from α via alpha_to_taus().
    α encodes the global LLM confidence personality — no extra LLM calls.
    """
    from models import MuGaloisC
    taus = alpha_to_taus(alpha)
    # MuGaloisC uses a single threshold per template (binary decision).
    # The τ_low from alpha_to_taus is recorded in results for reference
    # but does not affect _C — it will affect _F-M and _S in future experiments.
    return MuGaloisC(
        ctx, llm,
        tau_scan   = taus["tau_scan"],
        tau_simple = taus["tau_simple"],
        tau_chain  = taus["tau_chain_high"],
        tau_rec    = taus["tau_rec_high"],
        tau_choice = taus["tau_choice_high"],
        tau_hybrid = taus["tau_hybrid"],
    )


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
    tracker = TrackingLLM(base_llm)
    ctx0, gt, q = _get_ctx_and_gt(template, qid)
    full_qid = f"{template}_{qid}"

    print(f"\n{'='*72}")
    print(f" τ ablation | {full_qid} | {q.get('description','')}")
    print(f" GT={len(gt)}  α values={len(ALPHA_GRID)}")
    print(f"{'='*72}")

    report = Report(
        query_id=f"tau_{full_qid}",
        template=template,
        predicate=q.get("description", ""),
        n_expected=len(gt),
    )

    # T6 merges two JSON files
    if template == "T6":
        p1 = ROOT / "experiments/template6/queries_t6.json"
        p2 = ROOT / "experiments/template6/queries_t6_hybrid.json"
        queries = {
            **json.loads(p1.read_text(encoding="utf-8")),
            **json.loads(p2.read_text(encoding="utf-8")),
        }
    else:
        queries_path = {
            "T3": ROOT / "experiments/template3/queries_t3.json",
            "T4": ROOT / "experiments/template4/queries_t4.json",
            "T5": ROOT / "experiments/template5/queries_t5.json",
            "T7": ROOT / "experiments/template7/queries_t7.json",
        }[template]
        queries = json.loads(queries_path.read_text(encoding="utf-8"))

    for alpha in ALPHA_GRID:
        label = f"C_α={alpha}"
        default_marker = " [default]" if alpha == ALPHA_DEFAULT else ""
        print(f"\n  -- {label}{default_marker} --")
        taus = alpha_to_taus(alpha)
        print(f"     τ = {taus}")

        def ctx_factory(q_=queries[qid], t=template):
            ctx, _, _ = _get_ctx_and_gt(t, qid)
            return ctx

        agg, last = _aggregate(
            lambda ctx_i, a=alpha:
                run_mugalois_c_alpha(ctx_i, tracker, a),
            gt, n_runs, tracker, ctx_factory,
        )
        report.add(label, agg, actual=last or set(), expected=gt)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"tau_{full_qid}.json"
    report.save_json(out)
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser(
        description="τ ablation on µ-Galois_C — confidence-only routing"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--template", default=None,
                        help="Single template (T3/T4/T5/T6)")
    parser.add_argument("--query",    default=None,
                        help="Single query ID, e.g. T5_q7")
    parser.add_argument("--n-runs",   type=int, default=N_RUNS)
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()

    # Build list of (template, qid) to run
    if args.query:
        # e.g. "T5_q7"
        parts = args.query.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"--query must be in format T5_q7, got {args.query!r}")
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

    print(f"[alpha ablation] {len(to_run)} queries × {len(ALPHA_GRID)} α values "
          f"× {args.n_runs} runs  |  dry_run={args.dry_run}")
    print(f"  Queries : {[f'{t}_{q}' for t,q in to_run]}")
    print(f"  α grid  : {ALPHA_GRID}")
    print(f"  Example taus for α=0.50: {alpha_to_taus(0.50)}")

    for template, qid in to_run:
        run_query(template, qid, base_llm, args.n_runs)

    print("\n[tau ablation] Done. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()