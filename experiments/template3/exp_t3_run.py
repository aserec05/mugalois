"""
experiments/template3/run.py

Template 3 — two-hop path: SELECT ?b WHERE (s, p1, ?b) AND (?b, p2, t)

Models:
  NL
  SPARQL
  GD              left-to-right
  GD_lookahead    left-to-right + next hop hint
  DG              right-to-left
  DG_lookahead    right-to-left + next hop hint
  Join            independent scans + path_join
  Join_lookahead  join + cross hints
  SimpleScan      single LLM call for full path
  MuGalois        SimpleConf → DirectionConf → strategy

Metrics: P/R/F1 ± std, Time(s), Tokens

Usage
-----
    python3 -m experiments.template3.run
    python3 -m experiments.template3.run --query q1
    python3 -m experiments.template3.run --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.paths.gd import scan_gd
from mugalois.paths.dg import scan_dg
from mugalois.paths.join import scan_join
from mugalois.paths.simple import LLMSimpleScan
from mugalois.paths.mugalois import MuGaloisPath
from mugalois.core.prompts import genNLPrompt, genSPARQLPrompt, build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report


# ── Config ────────────────────────────────────────────────────────────────────

QUERIES_PATH = Path(__file__).resolve().parent / "queries_t3.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS  = 5
QUERIES = ["q1", "q2"]


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


# ── Runners ───────────────────────────────────────────────────────────────────

def run_nl(nl_prompt: str, tracker: TrackingLLM) -> set[str]:
    resp = tracker.chat(build_value_messages(genNLPrompt(nl_prompt)))
    return json_to_values(resp.text)


def run_sparql(sparql_prompt: str, tracker: TrackingLLM) -> set[str]:
    resp = tracker.chat(build_value_messages(genSPARQLPrompt(sparql_prompt)))
    return json_to_values(resp.text)


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
        s, p1, p2, t = q["s"], q["p1"], q["p2"], q["t"]

        print(f"\n{'='*60}")
        print(f"  Running {qid} | {q['description']}")
        print(f"  Pattern: ({s}, {p1}, ?b) AND (?b, {p2}, {t})")
        print(f"  Optimal: {q['optimal_direction']} | GT size: {len(gt)}")
        print(f"{'='*60}")

        report = Report(
            query_id=qid, template="T3",
            predicate=f"{p1}/{p2}", n_expected=len(gt),
        )

        # ── NL ────────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: run_nl(q["nl_prompt"], tracker),
            gt, N_RUNS, tracker, last)
        report.add("NL", agg, actual=last[0], expected=gt)

        # ── SPARQL ────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda sp=q["sparql_prompt"]: run_sparql(sp, tracker),
            gt, N_RUNS, tracker, last)
        report.add("SPARQL", agg, actual=last[0], expected=gt)

        # ── GD ────────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_gd(s, p1, p2, t, tracker),
            gt, N_RUNS, tracker, last)
        report.add("GD", agg, actual=last[0], expected=gt)

        # ── GD_lookahead ──────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_gd(s, p1, p2, t, tracker, lookahead=True),
            gt, N_RUNS, tracker, last)
        report.add("GD_lookahead", agg, actual=last[0], expected=gt)

        # ── DG ────────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_dg(s, p1, p2, t, tracker),
            gt, N_RUNS, tracker, last)
        report.add("DG", agg, actual=last[0], expected=gt)

        # ── DG_lookahead ──────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_dg(s, p1, p2, t, tracker, lookahead=True),
            gt, N_RUNS, tracker, last)
        report.add("DG_lookahead", agg, actual=last[0], expected=gt)

        # ── GD_motivational ──────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_gd(s, p1, p2, t, tracker, motivational=True),
            gt, N_RUNS, tracker, last)
        report.add("GD_motivational", agg, actual=last[0], expected=gt)

        # ── DG_motivational ──────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_dg(s, p1, p2, t, tracker, motivational=True),
            gt, N_RUNS, tracker, last)
        report.add("DG_motivational", agg, actual=last[0], expected=gt)

        # ── Join ──────────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_join(s, p1, p2, t, tracker),
            gt, N_RUNS, tracker, last)
        report.add("Join", agg, actual=last[0], expected=gt)

        # ── Join_lookahead ────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: scan_join(s, p1, p2, t, tracker, lookahead=True),
            gt, N_RUNS, tracker, last)
        report.add("Join_lookahead", agg, actual=last[0], expected=gt)

        # ── SimpleScan ────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: LLMSimpleScan(s, p1, p2, t, tracker),
            gt, N_RUNS, tracker, last)
        report.add("SimpleScan", agg, actual=last[0], expected=gt)

        # ── MuGalois ──────────────────────────────────────────────────────────
        last = []
        agg  = _aggregate(
            lambda: MuGaloisPath(s, p1, p2, t, tracker, verbose=True),
            gt, N_RUNS, tracker, last)
        report.add("MuGalois", agg, actual=last[0], expected=gt)

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
