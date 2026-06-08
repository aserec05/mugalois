"""
experiments/template1/exp_final/run.py

Final experiments for Template 1.
Best config from preliminaries: ValueScan pattern+nl, SPARQL nl URI.

Per query:
  - NL
  - SPARQL T1a   (bound term directly in pattern)
  - SPARQL T1b   (bound term in FILTER)
  - ValueScan    (pattern + nl encoding, reads pattern_nl from queries.json)

N=5 runs. All queries q1..q5.

Usage
-----
    python run.py           # Azure (default)
    python run.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mugalois.core.types import TriplePattern
from mugalois.core.prompts import (
    genNLPrompt, genSPARQLPrompt,
    genValueScanPrompt, genValueScanIterativePrompt,
    build_value_messages,
)
from mugalois.llm.llm_client import OllamaClient, AzureOpenAIClient, MockLLM
from mugalois.core.parser import json_to_values
from evaluation.metrics import AggregatedScores
from evaluation.evaluator import Evaluator
from evaluation.report import Report


# ── config ────────────────────────────────────────────────────────────────────

QUERIES_PATH = Path(__file__).resolve().parents[1] / "queries.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS   = 5
MAX_ITER = 10
QUERIES  = ["q1", "q2", "q3", "q4", "q5"]
# overridden by --query CLI arg


# ── URI → nl conversion for SPARQL prompts ───────────────────────────────────

import re as _re

_PRED_NL = {
    'dbo:spouse':          'spouse',
    'dbo:doctoralAdvisor': 'doctoral advisor',
    'dbo:award':           'award',
}
_RES_NL = {
    'dbr:Albert_Einstein':  'Albert Einstein',
    'dbr:Werner_Heisenberg':'Werner Heisenberg',
    'dbr:Mileva_Marić':     'Mileva Marić',
    'dbr:Marie_Curie':      'Marie Curie',
    'dbr:Turing_Award':     'Turing Award',
}

def _sparql_to_nl(sparql: str) -> str:
    """Replace all dbr:/dbo: tokens with NL labels in a SPARQL string."""
    result = sparql
    for uri, label in {**_PRED_NL, **_RES_NL}.items():
        result = result.replace(uri, label)
    return result


# ── Runners ───────────────────────────────────────────────────────────────────

def run_nl(nl_prompt: str, llm) -> set[str]:
    resp = llm.chat(build_value_messages(genNLPrompt(nl_prompt)))
    return json_to_values(resp.text)


def run_sparql(sparql_query: str, llm) -> set[str]:
    resp = llm.chat(build_value_messages(genSPARQLPrompt(sparql_query)))
    return json_to_values(resp.text)


def run_value_scan(pattern: TriplePattern, llm) -> set[str]:
    found   = set()
    ctx     = []
    sys_msg = build_value_messages("")[0]
    for i in range(MAX_ITER):
        prompt = (
            genValueScanPrompt(pattern, encoding="pattern")
            if i == 0
            else genValueScanIterativePrompt(found)
        )
        resp = llm.chat([sys_msg, *ctx, {"role": "user", "content": prompt}])
        new  = json_to_values(resp.text)
        if new.issubset(found):
            break
        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        found |= new
    return found


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt: set[str], n: int,
               last_actual: list) -> AggregatedScores:
    scores = []
    for i in range(n):
        actual = run_fn()
        score  = Evaluator(actual, gt, mode="values").evaluate()
        print(f"      run {i+1}/{n} → returned={len(actual)} expected={len(gt)} | "
              f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}")
        print(f"               values: {sorted(actual)[:5]}{'...' if len(actual) > 5 else ''}")
        if i == n - 1:
            last_actual.append(actual)
        scores.append(score)
    return AggregatedScores.from_runs(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(llm):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for qid in QUERIES:
        q     = queries[qid]
        gt    = set(q["ground_truth"])
        tpl_a = q["T1a"]
        tpl_b = q["T1b"]

        nl_prompt  = q["nl_prompt"]
        sparql_a   = _sparql_to_nl(tpl_a["sparql_prompt"])
        sparql_b   = _sparql_to_nl(tpl_b["sparql_prompt"])

        # pattern_nl read directly from queries.json — no hardcoded dicts
        raw_nl  = tpl_a["pattern_nl"]
        pattern = TriplePattern(s=raw_nl["s"], p=raw_nl["p"], o=raw_nl["o"])

        report = Report(
            query_id=qid,
            template="T1a+T1b",
            predicate=tpl_a["pattern"]["p"],
            n_expected=len(gt),
        )

        print(f"\n{'='*60}")
        print(f"  Running {qid}")
        print(f"{'='*60}")

        # ── NL ───────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(lambda: run_nl(nl_prompt, llm), gt, N_RUNS, last)
        report.add("NL", agg, actual=last[0], expected=gt)

        # ── SPARQL T1a ───────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda sq=sparql_a: run_sparql(sq, llm), gt, N_RUNS, last)
        report.add("SPARQL_T1a", agg, actual=last[0], expected=gt)

        # ── SPARQL T1b (w/ FILTER) ───────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda sq=sparql_b: run_sparql(sq, llm), gt, N_RUNS, last)
        report.add("SPARQL_T1b", agg, actual=last[0], expected=gt)

        # ── ValueScan (pattern + nl) ──────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda p=pattern: run_value_scan(p, llm), gt, N_RUNS, last)
        report.add("ValueScan", agg, actual=last[0], expected=gt)

        # ── Save ──────────────────────────────────────────────────────
        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}.json")
        report.save_csv(RESULTS_DIR  / f"{qid}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",   default="azure",
                        choices=["azure", "ollama", "mock"])
    parser.add_argument("--model", default="phi3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None,
                        help="Run a single query e.g. --query q2")
    args = parser.parse_args()
    if args.query:
        QUERIES = [args.query]

    if args.dry_run or args.llm == "mock":
        llm = MockLLM()
    elif args.llm == "ollama":
        llm = OllamaClient(model=args.model)
    else:
        llm = AzureOpenAIClient()

    main(llm)