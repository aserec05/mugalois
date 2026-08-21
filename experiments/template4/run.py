"""
experiments/template4/run.py - Template 4 — N-hop sequence path queries.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.paths.path_query import PathQuery
from mugalois.core.types import TriplePattern, Environment, Condition
from mugalois.core.prompts import genNLPrompt, genSPARQLPrompt, build_value_messages
from mugalois.core.parser import json_to_values

from mugalois.paths.multi_scan import LLMMultiHopScan
from mugalois.paths.chain import scan_chain
from mugalois.paths.routing import decide_routing
from mugalois.paths.mugalois_multi import MuGaloisMultiPath

from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

QUERIES_PATH = Path(__file__).resolve().parent / "queries_t4.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS   = 5
MAX_ITER = 5
VERBOSE  = False

QUERIES    = ["q1", "q2", "q3", "q4", "q5", "q7", "q8"]
STRATEGIES = ["nl_naive", "nl_precise", "sparql", "multihop", "chain", "mugalois"]


def build_path(q: dict) -> PathQuery:
    path = PathQuery.from_chain(q["anchors"], q["predicates"], q["target_var"])
    # Fix 1: store condition labels for prompt generation
    labels = {}
    if "conditions" in q:
        for c in q["conditions"]:
            path.conditions.append(Condition(var=c["var"], op=c["op"], val=c["val"]))
            if "label" in c:
                labels[c["var"]] = c["label"]
    path.condition_labels = labels  # type: ignore[attr-defined]
    return path


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


def run_nl_naive(q, path, tracker):
    resp = tracker.chat(build_value_messages(genNLPrompt(q["nl_prompt_naive"])))
    return json_to_values(resp.text)

def run_nl_precise(q, path, tracker):
    resp = tracker.chat(build_value_messages(genNLPrompt(q["nl_prompt_precise"])))
    return json_to_values(resp.text)

def run_sparql(q, path, tracker):
    resp = tracker.chat(build_value_messages(genSPARQLPrompt(q["sparql_prompt"])))
    return json_to_values(resp.text)

def run_multihop(q, path, tracker):
    return LLMMultiHopScan(path, tracker)

def run_chain(q, path, tracker):
    gamma   = Environment()
    routing = decide_routing(path, tracker, gamma, verbose=VERBOSE)
    return scan_chain(
        path=path, llm=tracker, direction=routing.get("direction", "left"),
        gamma=gamma, max_iter=MAX_ITER, verbose=VERBOSE,
    )

def run_mugalois(q, path, tracker):
    return MuGaloisMultiPath(
        path=path, llm=tracker, gamma=Environment(),
        tau_high=0.50, tau_low=0.35,
        max_iter=MAX_ITER, verbose=VERBOSE,
    )

STRATEGY_REGISTRY = {
    "nl_naive":  (run_nl_naive,   "NL_naive"),
    "nl_precise": (run_nl_precise, "NL_precise"),
    "sparql":    (run_sparql,     "SPARQL"),
    "multihop":  (run_multihop,   "LLMMultiHopScan"),
    "chain":     (run_chain,      "LLMChainScan"),
    "mugalois":  (run_mugalois,   "MuGaloisMultiPath"),
}


def _aggregate(run_fn, gt, n, tracker, last):
    scores = []
    for i in range(n):
        tracker.reset()
        values = {str(v) for v in run_fn()}
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


def main(base_llm, queries_to_run, strategies_to_run):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q    = queries[qid]
        gt   = set(q["ground_truth"])
        path = build_path(q)

        print(f"\n{'='*72}")
        print(f" {qid} | {q['description']}")
        print(f" Path   : {path}")
        print(f" GT={len(gt)}  source: {q['source']}")
        print(f" Strategies: {', '.join(strategies_to_run)}")
        print(f"{'='*72}")

        report = Report(
            query_id=qid, template="T4",
            predicate="/".join(path.predicates()),
            n_expected=len(gt),
        )

        for strat_key in strategies_to_run:
            run_fn, display_name = STRATEGY_REGISTRY[strat_key]
            print(f"\n  -- {display_name} --")
            last = []
            agg  = _aggregate(
                lambda fn=run_fn: fn(q, path, tracker),
                gt, N_RUNS, tracker, last,
            )
            report.add(display_name, agg, actual=last[0], expected=gt)

        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}.json")
        report.save_csv(RESULTS_DIR / f"{qid}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--model", default="all",
        help=f"Comma-separated or 'all'. Available: {', '.join(STRATEGIES)}")
    args = parser.parse_args()

    queries_to_run = [args.query] if args.query else QUERIES
    strategies_to_run = STRATEGIES if args.model == "all" else [
        s.strip() for s in args.model.split(",")
    ]
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    print(f"[run.py T4] model={args.model}  dry_run={args.dry_run}")
    main(base_llm, queries_to_run, strategies_to_run)