"""
experiments/template5/run.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.core.types import RecursivePattern
from mugalois.core.prompts import (
    genNLPrompt,
    genSPARQLPrompt,
    build_value_messages,
)
from mugalois.core.parser import json_to_values

from mugalois.rec.simple_rec import LLMRecScan
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.rec.mugalois_rec import MuGaloisRec

from mugalois.llm.llm_client import (
    AzureOpenAIClient,
    MockLLM,
    BaseLLM,
)

from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report


QUERIES_PATH = Path(__file__).resolve().parent / "queries_t5.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS    = 5
MAX_DEPTH = 300
VERBOSE   = False

QUERIES = ["q1", "q2", "q3", "q4", "q5", "q7", "q8", "q9", "q10"]

STRATEGIES = ["nl_naive", "nl_precise", "sparql", "recscan", "fixpoint", "mugalois"]


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _strip_titles(name: str) -> str:
    """Strip royal/noble title prefixes and suffixes generically.

    'King Charles III'                    -> 'Charles III'
    'Prince William, Prince of Wales'     -> 'William'
    'Lady Davina Lewis'                   -> 'Davina Lewis'
    'Lord Frederick Windsor'              -> 'Frederick Windsor'
    'Princess Margaret, Countess of Snowdon' -> 'Margaret'
    'George Lascelles, 7th Earl of Harewood' -> 'George Lascelles'

    Applied to both returned values and GT when query has 'strip_titles': true.
    No manual alias map needed.
    """
    s = name.strip()
    s = re.sub(
        r'^(King|Queen|Prince|Princess|Lady|Lord|The\s+Hon(?:ourable)?|HRH|HH|Sir|Dame|The)\s+',
        '', s
    )
    s = re.sub(
        r',\s+(Duke|Duchess|Earl|Countess|Viscount|Viscountess|Baron|Baroness|'
        r'Princess\s+Royal|Prince|Princess|Marquess|Lord|Lady).*$',
        '', s
    )
    s = re.sub(r',\s+\d+(st|nd|rd|th)\s+.*$', '', s)
    return s.strip()


def _normalize(values: set[str], aliases: dict, do_strip_titles: bool = False) -> set[str]:
    """Normalize entity names before metric computation.

    Two independent mechanisms, both optional:
    - do_strip_titles: strips royal/noble title prefixes/suffixes generically
    - aliases: maps specific name variants to canonical forms
    """
    if do_strip_titles:
        values = {_strip_titles(v) for v in values}
    if not aliases:
        return values
    return {aliases.get(v, v) for v in values}


# ─────────────────────────────────────────────────────────────────────────────
# TrackingLLM
# ─────────────────────────────────────────────────────────────────────────────

class TrackingLLM(BaseLLM):
    def __init__(self, llm):
        self._llm = llm
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


# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────

def run_nl_naive(q, pattern, tracker):
    resp = tracker.chat(build_value_messages(genNLPrompt(q["nl_prompt_naive"])))
    return json_to_values(resp.text)


def run_nl_precise(q, pattern, tracker):
    resp = tracker.chat(build_value_messages(genNLPrompt(q["nl_prompt_precise"])))
    return json_to_values(resp.text)


def run_sparql(q, pattern, tracker):
    resp = tracker.chat(build_value_messages(genSPARQLPrompt(q["sparql_prompt"])))
    return json_to_values(resp.text)


def run_rec_scan(q, pattern, tracker):
    return LLMRecScan(pattern, tracker)


def run_fixpoint(q, pattern, tracker):
    return LLMFixpointScan(pattern, tracker, max_depth=MAX_DEPTH, verbose=VERBOSE)


def run_mugalois_rec(q, pattern, tracker):
    return MuGaloisRec(
        pattern, tracker,
        tau_high=0.50, tau_low=0.35,
        max_depth=MAX_DEPTH, verbose=VERBOSE,
    )

STRATEGY_REGISTRY = {
    "nl_naive":   (run_nl_naive,     "NL_naive"),
    "nl_precise": (run_nl_precise,   "NL_precise"),
    "sparql":     (run_sparql,       "SPARQL"),
    "recscan":    (run_rec_scan,     "LLMRecScan"),
    "fixpoint":   (run_fixpoint,     "LLMFixpointScan"),
    "mugalois":   (run_mugalois_rec, "MuGaloisRec"),
}


def build_pattern(q):
    return RecursivePattern(q["s"], q["p"], q["o"], q["operator"])


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, aliases, do_strip_titles, n, tracker, last):
    scores = []
    for i in range(n):
        tracker.reset()
        raw    = {str(v) for v in run_fn()}
        values = _normalize(raw, aliases, do_strip_titles)
        print(f"  returned: {sorted(values)}")
        print(f"  expected: {sorted(gt)}")
        score = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        print(f"      run {i+1}/{n} -> returned={len(values)} "
              f"expected={len(gt)} | P={score.precision:.3f} "
              f"R={score.recall:.3f} F1={score.f1:.3f} "
              f"t={score.time_s:.1f}s tok={score.tokens}")
        if i == n - 1:
            last.append(values)
        scores.append(score)
    return AggregatedScores.from_runs(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(base_llm, queries_to_run, strategies_to_run):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q              = queries[qid]
        gt             = set(q["ground_truth"])
        aliases        = q.get("aliases", {})
        do_strip_titles = q.get("strip_titles", False)
        pattern        = build_pattern(q)

        print(f"\n{'='*70}")
        print(f" Running {qid} | {q['description']}")
        print(f" Pattern : {pattern}")
        print(f" GT size : {len(gt)}")
        print(f" Strategies: {', '.join(strategies_to_run)}")
        print(f"{'='*70}")

        report = Report(query_id=qid, template="T5", predicate=q["p"], n_expected=len(gt))

        for strat_key in strategies_to_run:
            run_fn, display_name = STRATEGY_REGISTRY[strat_key]
            print(f"\n  -- {display_name} --")
            last = []
            agg  = _aggregate(
                lambda: run_fn(q, pattern, tracker),
                gt, aliases, do_strip_titles, N_RUNS, tracker, last
            )
            report.add(display_name, agg, actual=last[0], expected=gt)

        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}.json")
        report.save_csv(RESULTS_DIR / f"{qid}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument(
        "--model", default="all",
        help=f"Comma-separated strategies to run, or 'all'. "
             f"Available: {', '.join(STRATEGIES)}"
    )
    args = parser.parse_args()

    queries_to_run = [args.query] if args.query else QUERIES
    if args.model == "all":
        strategies_to_run = STRATEGIES
    else:
        strategies_to_run = [s.strip() for s in args.model.split(",")]
        for s in strategies_to_run:
            if s not in STRATEGIES:
                parser.error(f"Unknown strategy '{s}'. Available: {', '.join(STRATEGIES)}")
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()

    print(f"[run.py] strategy={args.model}  dry_run={args.dry_run}")
    main(base_llm, queries_to_run, strategies_to_run)