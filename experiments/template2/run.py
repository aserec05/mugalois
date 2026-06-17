"""
experiments/template2/run.py

Template 2 — 3 models compared:

  NL          : natural language baseline
  SPARQL      : SPARQL baseline
  LLMScan     : full orchestrator (RW1 → KeyScan/SeedScan/TripleScan → LLMConfCond)
  LLMScan_KO  : RW1 + force KeyScan (key_only mode)
  LLMScan_PF  : no RW1, no inject — all conditions in post_filter

Metrics: P/R/F1 ± std, Time(s), Tokens

Usage
-----
    python3 -m experiments.template2.run
    python3 -m experiments.template2.run --query q3
    python3 -m experiments.template2.run --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.core.types import (
    TriplePattern, Triple, Condition, ConditionIN, Environment,
)
from mugalois.core.rewrite import RW1
from mugalois.core.prompts import genNLPrompt, genSPARQLPrompt, build_value_messages
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.scans.triple_scan import LLMTripleScan, LLMValueScan
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from mugalois.core.parser import json_to_values
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report


# ── Config ────────────────────────────────────────────────────────────────────

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS   = 5
MAX_ITER = 5
QUERIES  = ["q1", "q2", "q3", "q4", "q5"]


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


# ── Condition parsing ─────────────────────────────────────────────────────────

def parse_conditions(raw: list) -> frozenset:
    result = set()
    for c in raw:
        if c["type"] == "IN":
            result.add(ConditionIN(c["var"], frozenset(c["values"])))
        else:
            result.add(Condition(c["var"], c["type"], c["val"]))
    return frozenset(result)


# ── Value extraction ──────────────────────────────────────────────────────────

def extract_values(triples: set, return_var: str,
                   pattern: TriplePattern) -> set[str]:
    if return_var == pattern.s:
        return {t.s for t in triples}
    elif return_var == pattern.o:
        return {t.o for t in triples}
    elif pattern.s_is_var():
        return {t.s for t in triples}
    return {t.o for t in triples}


# ── Runners ───────────────────────────────────────────────────────────────────

def run_nl(nl_prompt: str, tracker: TrackingLLM) -> set[str]:
    resp = tracker.chat(build_value_messages(genNLPrompt(nl_prompt)))
    return json_to_values(resp.text)


def run_sparql(sparql_prompt: str, tracker: TrackingLLM) -> set[str]:
    resp = tracker.chat(build_value_messages(genSPARQLPrompt(sparql_prompt)))
    return json_to_values(resp.text)


def run_llmscan(
    pattern:    TriplePattern,
    env:        Environment,
    gamma:      Environment,
    tracker:    TrackingLLM,
    return_var: str,
    mode:       str = "standard",
) -> set[str]:
    """LLMScan — full orchestrator or key_only mode."""
    triples = LLMScan(
        pattern=copy.deepcopy(pattern),
        env=copy.deepcopy(env),
        gamma=copy.deepcopy(gamma),
        llm=tracker,
        max_iter=MAX_ITER,
        mode=mode,
    )
    return extract_values(triples, return_var, pattern)


def run_postfilter(
    pattern:    TriplePattern,
    conditions: frozenset,
    tracker:    TrackingLLM,
    return_var: str,
) -> set[str]:
    """
    PostFilter — no RW1, no seeds, no inject.
    All conditions go to post_filter_conds.
    TripleScan or ValueScan depending on pattern variables.
    """
    env = Environment()

    # All conditions → post_filter
    post_conds = list(conditions)

    s_var = pattern.s_is_var()
    o_var = pattern.o_is_var()

    if s_var and o_var:
        triples = LLMTripleScan(
            pattern, env, tracker,
            post_filter_conds=post_conds,
            max_iter=MAX_ITER,
        )
        return extract_values(triples, return_var, pattern)
    elif s_var:
        values = LLMValueScan(
            pattern, env, tracker,
            post_filter_conds=post_conds,
            max_iter=MAX_ITER,
        )
        return values
    else:
        values = LLMValueScan(
            pattern, env, tracker,
            post_filter_conds=post_conds,
            max_iter=MAX_ITER,
        )
        return values


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, gt, n, tracker, last) -> AggregatedScores:
    scores = []
    for i in range(n):
        tracker.reset()
        values = run_fn()
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        print(f"      run {i+1}/{n} → returned={len(values)} "
              f"expected={len(gt)} | "
              f"P={score.precision:.3f} R={score.recall:.3f} "
              f"F1={score.f1:.3f} "
              f"t={score.time_s:.1f}s tok={score.tokens}")
        if i == n - 1:
            last.append(values)
        scores.append(score)
    return AggregatedScores.from_runs(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_llm, queries_to_run: list[str]):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q  = queries[qid]
        gt = set(q["ground_truth"])

        raw        = q["pattern_nl"]
        pattern    = TriplePattern(s=raw["s"], p=raw["p"], o=raw["o"])
        return_var = q.get("return_var",
                           pattern.o if pattern.o_is_var() else pattern.s)
        nl_prompt     = q["nl_prompt"]
        sparql_prompt = q["sparql_prompt"]

        # Parse conditions
        conditions = parse_conditions(q.get("conditions", []))

        # RW1 for LLMScan and KeyOnly
        env   = Environment()
        gamma = Environment()
        env, gamma = RW1(conditions, env, gamma)

        print(f"\n{'='*60}")
        print(f"  Running {qid} | cardinality={q['cardinality']}")
        print(f"  Pattern    : {pattern}")
        print(f"  Return var : {return_var}")
        print(f"  RW1 env    : {dict(env._bindings)}")
        print(f"  RW1 gamma  : {dict(gamma._bindings)}")
        print(f"{'='*60}")

        report = Report(
            query_id=qid, template="T2",
            predicate=raw["p"], n_expected=len(gt),
        )

        # ── NL ────────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: run_nl(nl_prompt, tracker),
            gt, N_RUNS, tracker, last)
        report.add("NL", agg, actual=last[0], expected=gt)

        # ── SPARQL ────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda sp=sparql_prompt: run_sparql(sp, tracker),
            gt, N_RUNS, tracker, last)
        report.add("SPARQL", agg, actual=last[0], expected=gt)

        # ── LLMScan ───────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: run_llmscan(
                pattern, env, gamma, tracker, return_var, mode="standard"),
            gt, N_RUNS, tracker, last)
        report.add("LLMScan", agg, actual=last[0], expected=gt)

        # ── LLMScan_KeyOnly ───────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: run_llmscan(
                pattern, env, gamma, tracker, return_var, mode="key_only"),
            gt, N_RUNS, tracker, last)
        report.add("LLMScan_KeyOnly", agg, actual=last[0], expected=gt)

        # ── LLMScan_PostFilter ────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: run_postfilter(pattern, conditions, tracker, return_var),
            gt, N_RUNS, tracker, last)
        report.add("LLMScan_PostFilter", agg, actual=last[0], expected=gt)

        # ── Save ──────────────────────────────────────────────────────────────
        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}.json")
        report.save_csv(RESULTS_DIR  / f"{qid}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query",   default=None)
    args = parser.parse_args()

    queries_to_run = [args.query] if args.query else QUERIES
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    main(base_llm, queries_to_run)