# experiments/template7/run.py
"""
T7 — Hybrid path queries — final evaluation.

Strategies
----------
  nl_naive      : 1 NL prompt (q["nl"])                                [baseline]
  nl_precise    : 1 NL prompt (q["nl_precise"], linguistically precise) [baseline]
  sparql        : semantic SPARQL prompt (human-readable predicates)    [baseline]
  holistic      : single structured prompt from plan expression         [ours]
  holistic_psych: holistic + psychological priming (CoT persona)        [ours]
  segment       : node-by-node evaluation, direction auto-detected      [ours]
  mugalois      : HybridPlannerV4 — dependency-graph planner            [ours]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.hybrid.hybrid_plan import from_query_json
from mugalois.hybrid.mugalois_hybrid import (
    _holistic_prompt,
    _segment_eval,
    _direction,
)
from mugalois.hybrid.hybrid_planner_v4 import plan_and_execute_v4
from mugalois.core.types import Environment
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import AzureOpenAIClient, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report


QUERIES_PATH = Path(__file__).resolve().parent / "queries_t7.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
N_RUNS = 5

QUERIES = ["q1", "q1b", "q4", "q5", "q6", "q7", "q8"]

STRATEGIES = [
    "nl_naive",
    "nl_precise",
    "sparql",
    "holistic",
    "holistic_psych",
    "segment",
    "mugalois",
]


# ── Tracking wrapper ──────────────────────────────────────────────────

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
        self.total_tokens += getattr(resp, "usage_tokens", 0)
        self.total_time_s += getattr(resp, "latency_s",   0.0)
        return resp


# ── Strategy functions ────────────────────────────────────────────────

def run_nl_naive(q, plan, tracker):
    """
    Baseline 1 — NL naive.
    Single prompt from the natural language description of the query.
    """
    prompt = (
        f"By your knowledge: {q['nl']}\n"
        f"Be exhaustive. Return ONLY a JSON array of entity names."
    )
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_nl_precise(q, plan, tracker):
    """
    Baseline 2 — NL precise.
    Single prompt from the linguistically precise NL formulation,
    including all filters and constraints.
    """
    nl = q.get("nl_precise") or q.get("nl", "")
    prompt = (
        f"By your knowledge: {nl}\n"
        f"Be exhaustive. Include all you are confident about.\n"
        f"Return ONLY a JSON array of entity names."
    )
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_sparql(q, plan, tracker):
    """
    Baseline 3 — Semantic SPARQL.
    SPARQL query with human-readable semantic predicates (no Wikidata IDs).
    The LLM evaluates the query against its parametric knowledge.
    """
    sparql = (
        q.get("sparql_semantic")
        or q.get("sparql_wikidata")
    )
    if not sparql:
        var    = q.get("target_var", "?x")
        expr   = q.get("expression", "")
        anchor = q.get("target") or q.get("anchor", "")
        sparql = (
            f"SELECT DISTINCT {var} WHERE {{\n"
            f"  {var} {expr} {anchor} .\n"
            f"}}"
        )
    prompt = (
        f"Using your encyclopedic knowledge, evaluate this query and "
        f"return all results:\n\n"
        f"{sparql}\n\n"
        f"Be exhaustive. Return ONLY a JSON array of entity names."
    )
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_holistic(q, plan, tracker):
    """
    Ours 1 — Holistic.
    Single structured prompt built from the plan expression and NL.
    No decomposition.
    """
    prompt = _holistic_prompt(plan)
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_holistic_psych(q, plan, tracker):
    """
    Ours 2 — Holistic + psychological priming.
    Assigns an expert persona, asks for explicit mental enumeration,
    counters saliency bias ("not just the famous ones").
    Based on Kojima et al. (2022) chain-of-thought priming.
    """
    nl    = plan.nl or (
        f"Find all {plan.target_var} such that "
        f"{plan.source} {plan.expression} {plan.target}."
    )
    expr_lower = (plan.expression or "").lower()

    if any(w in expr_lower for w in ["pope", "bishop", "church"]):
        persona = "expert historian of the Catholic Church with encyclopedic knowledge"
    elif any(w in expr_lower for w in ["president", "prime minister", "successor", "politician"]):
        persona = "expert political historian with encyclopedic knowledge of world leaders"
    elif any(w in expr_lower for w in ["film", "director", "sequel", "oscar", "award"]):
        persona = "expert film historian with encyclopedic knowledge of cinema"
    elif any(w in expr_lower for w in ["child", "descendant", "royal", "victoria", "born"]):
        persona = "expert genealogist and royal historian"
    else:
        persona = "expert knowledge graph reasoner with encyclopedic factual knowledge"

    prompt = (
        f"You are an {persona}.\n\n"
        f"Your task: {nl}\n\n"
        f"Before writing your answer, do the following mentally:\n"
        f"  1. Recall ALL entities that could be relevant — "
        f"not just the famous or well-known ones.\n"
        f"  2. For each candidate, verify it satisfies every condition.\n"
        f"  3. Only then write your final answer.\n\n"
        f"Be exhaustive. Do not stop at the first few obvious answers. "
        f"Include lesser-known entities you are confident about.\n"
        f"Do not guess or hallucinate. Only include what you know to be true.\n\n"
        f"Return ONLY a JSON array of entity names."
    )
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_segment(q, plan, tracker, verbose=False):
    """
    Ours 3 — Segment.
    Node-by-node evaluation, direction auto-detected from anchor_side.
    No confidence routing — always decomposes segment by segment.
    """
    direction = _direction(plan)
    return _segment_eval(
        plan, direction, tracker, Environment(),
        motivational=False, verbose=verbose,
    )


def run_mugalois(q, plan, tracker, verbose=False):
    """
    µ-Galois — HybridPlannerV4.
    Dependency-graph planner: TYPE ANCHOR → EXPAND → CHAIN → FILTER.
    Always decomposes. Handles anchor_side=both via split/join.
    """
    return plan_and_execute_v4(plan, tracker, Environment(), verbose=verbose)


# Strategies that accept a verbose parameter
_VERBOSE_STRATEGIES = {"segment", "mugalois"}

STRATEGY_REGISTRY = {
    "nl_naive":      (run_nl_naive,      "NL_naive"),
    "nl_precise":    (run_nl_precise,    "NL_precise"),
    "sparql":        (run_sparql,        "SPARQL"),
    "holistic":      (run_holistic,      "Holistic"),
    "holistic_psych":(run_holistic_psych,"Holistic_psych"),
    "segment":       (run_segment,       "Segment"),
    "mugalois":      (run_mugalois,      "µ-Galois"),
}


# ── Evaluation ────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, n, tracker):
    """
    Evaluate a strategy over N runs.

    Null queries (GT=∅) handled by Metrics.compute:
      pred=∅  → P=1, R=1, F1=1  (correctly returns nothing)
      pred≠∅  → P=0, R=1, F1=0  (hallucinates on empty GT)
    """
    scores = []
    last   = None
    null_q = len(gt) == 0

    for i in range(n):
        tracker.reset()
        values = {str(v) for v in run_fn()}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s  = tracker.total_time_s
        score.tokens  = tracker.total_tokens
        tag = "  [null query]" if null_q else ""
        print(
            f"      run {i+1}/{n}  "
            f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}  "
            f"returned={len(values)}  expected={len(gt)}{tag}  "
            f"t={score.time_s:.1f}s  tok={score.tokens}"
        )
        if i == n - 1:
            last = values
        scores.append(score)
    return AggregatedScores.from_runs(scores), last


# ── Main ──────────────────────────────────────────────────────────────

def main(base_llm, queries_to_run, strategies_to_run, verbose=False,
         n_runs=N_RUNS):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        if qid not in queries:
            print(f"[skip] {qid} not in queries file")
            continue

        q    = queries[qid]
        gt   = set(q.get("ground_truth", []))
        plan = from_query_json(q)

        print(f"\n{'='*72}")
        print(f" {qid} | {q.get('description','')}")
        print(f" Expression : {q.get('expression','')}")
        print(f" NL         : {q.get('nl','')}")
        print(f" Anchor     : {q.get('anchor','')}  "
              f"Side: {q.get('anchor_side','right')}")
        print(f" Complexity : {plan.complexity}")
        print(f" GT={len(gt)}  (gt_size={q.get('gt_size',len(gt))})")
        print(f" Strategies : {', '.join(strategies_to_run)}")
        print(f"{'='*72}")

        report = Report(
            query_id=qid,
            template="T7",
            predicate=q.get("expression", ""),
            n_expected=len(gt),
        )

        for strat_key in strategies_to_run:
            run_fn, display_name = STRATEGY_REGISTRY[strat_key]
            print(f"\n  -- {display_name} --")

            if strat_key in _VERBOSE_STRATEGIES:
                fn = lambda fn=run_fn: fn(q, plan, tracker, verbose)
            else:
                fn = lambda fn=run_fn: fn(q, plan, tracker)

            agg, last = _aggregate(fn, gt, n_runs, tracker)
            report.add(display_name, agg, actual=last or set(), expected=gt)

        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}.json")
        report.save_csv( RESULTS_DIR / f"{qid}.csv")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="T7 hybrid path query evaluation"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--query",    default=None,
                        help="Single query ID, e.g. q1")
    parser.add_argument("--strategy", default="all",
                        help=f"Comma-separated or 'all'. "
                             f"Available: {', '.join(STRATEGIES)}")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--n-runs",   type=int, default=N_RUNS)
    args = parser.parse_args()

    queries_to_run    = [args.query] if args.query else QUERIES
    strategies_to_run = (
        STRATEGIES if args.strategy == "all"
        else [s.strip() for s in args.strategy.split(",")]
    )

    unknown = set(strategies_to_run) - set(STRATEGY_REGISTRY)
    if unknown:
        print(f"[error] Unknown strategies: {unknown}")
        print(f"Available: {list(STRATEGY_REGISTRY)}")
        sys.exit(1)

    if args.dry_run:
        from mugalois.llm.llm_client import MockLLM
        base_llm = MockLLM()
    else:
        base_llm = AzureOpenAIClient()

    print(f"[run_t7] queries={queries_to_run}  strategies={strategies_to_run}  "
          f"dry_run={args.dry_run}  n_runs={args.n_runs}")
    main(base_llm, queries_to_run, strategies_to_run,
         verbose=args.verbose, n_runs=args.n_runs)