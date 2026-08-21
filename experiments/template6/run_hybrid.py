"""
experiments/template6/run_hybrid.py — Choice with mixed branches (T6 hybrid).

Strategies:
  nl_naive        → direct NL prompt
  nl_precise      → precise NL prompt
  all_parallel    → one call per branch, union
  holistic_only   → force holistic OR on all branches
  mugalois_hybrid → MuGaloisChoice full hybrid execution
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

from mugalois.choice.choice_path import ChoicePath
from mugalois.choice.mugalois_choice import (
    MuGaloisChoice, _eval_simple_branch, _eval_complex_branch,
)
from mugalois.paths.path_query import PathQuery
from mugalois.core.types import Environment, RecursivePattern
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

QUERIES_PATH = Path(__file__).resolve().parent / "queries_t6_hybrid.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results_hybrid"

N_RUNS  = 5
VERBOSE = False

QUERIES    = ["q7", "q8", "q9", "q10"]
STRATEGIES = [
    "nl_naive",
    "nl_precise",
    "all_parallel",
    "holistic_only",
    "mugalois_hybrid",
]


# ── Build ChoicePath ──────────────────────────────────────────────────

def build_complex_branch(b: dict, source: str) -> object:
    if b["type"] == "T4":
        hops    = b["hops"]
        anchors = (
            [source]
            + [f"?inter{i}" for i in range(len(hops) - 1)]
            + ["?b"]
        )
        return PathQuery.from_chain(anchors, hops, "?b")
    elif b["type"] == "T5":
        return RecursivePattern(
            s=source, p=b["predicate"], o="?b", operator=b["operator"]
        )
    raise ValueError(f"Unknown complex branch type: {b['type']}")


def build_choice_path(q: dict) -> ChoicePath:
    simple   = q.get("branches_simple", [])
    complex_ = [
        build_complex_branch(b, q["source"])
        for b in q.get("branches_complex", [])
    ]
    return ChoicePath(
        source     = q["source"],
        target_var = q["target_var"],
        branches   = simple + complex_,
        direction  = q.get("direction", "forward"),
    )


# ── Tracking LLM ──────────────────────────────────────────────────────

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


# ── Strategy functions ────────────────────────────────────────────────

def run_nl_naive(q, path, tracker):
    resp = tracker.chat(build_value_messages(q["nl_prompt_naive"]))
    return json_to_values(resp.text)


def run_nl_precise(q, path, tracker):
    resp = tracker.chat(build_value_messages(q["nl_prompt_precise"]))
    return json_to_values(resp.text)


def run_all_parallel(q, path, tracker):
    """One call per branch with best orchestrator — no holistic."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    result   = set()
    branches = path.branches

    def eval_b(b):
        if isinstance(b, str):
            return _eval_simple_branch(b, path, tracker)
        return _eval_complex_branch(b, path, tracker, Environment())

    with ThreadPoolExecutor(max_workers=len(branches)) as executor:
        futures = [executor.submit(eval_b, b) for b in branches]
        for f in as_completed(futures):
            try:
                result |= f.result()
            except Exception as e:
                print(f"  [parallel] error: {e}")
    return result


def run_holistic_only(q, path, tracker):
    """Force holistic OR on all branches."""
    from mugalois.choice.mugalois_choice import _holistic_prompt
    # Build a simple-only path for the prompt
    all_nl = (
        q.get("branches_simple", [])
        + [b.get("description", "") for b in q.get("branches_complex", [])]
    )
    simple_path = ChoicePath(
        source=path.source, target_var=path.target_var,
        branches=all_nl, direction=path.direction,
    )
    prompt = _holistic_prompt(simple_path)
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_mugalois_hybrid(q, path, tracker):
    return MuGaloisChoice(
        path, tracker,
        tau_high=0.60, tau_low=0.40,
        verbose=True,
    )


STRATEGY_REGISTRY = {
    "nl_naive":        (run_nl_naive,        "NL_naive"),
    "nl_precise":      (run_nl_precise,      "NL_precise"),
    "all_parallel":    (run_all_parallel,    "AllParallel"),
    "holistic_only":   (run_holistic_only,   "HolisticOnly"),
    "mugalois_hybrid": (run_mugalois_hybrid, "MuGaloisHybrid"),
}


# ── Aggregation ───────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────

def main(base_llm, queries_to_run, strategies_to_run):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q    = queries[qid]
        gt   = set(q["ground_truth"])
        path = build_choice_path(q)

        print(f"\n{'='*72}")
        print(f" {qid} | {q['description']}")
        print(f" direction={q.get('direction','forward')} "
              f"simple={q['n_simple']} complex={q['n_complex']}")
        print(f" GT={len(gt)}  optimal={q['optimal_strategy']}")
        print(f" Strategies: {', '.join(strategies_to_run)}")
        print(f"{'='*72}")

        report = Report(
            query_id  = qid,
            template  = "T6-hybrid",
            predicate = q["description"],
            n_expected= len(gt),
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


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--model", default="all",
        help=f"Comma-separated or 'all'. Available: {', '.join(STRATEGIES)}")
    args = parser.parse_args()

    queries_to_run    = [args.query] if args.query else QUERIES
    strategies_to_run = STRATEGIES if args.model == "all" else [
        s.strip() for s in args.model.split(",")
    ]
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    print(f"[run_hybrid.py T6] model={args.model}  dry_run={args.dry_run}")
    main(base_llm, queries_to_run, strategies_to_run)