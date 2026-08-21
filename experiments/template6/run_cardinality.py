"""
experiments/template6/run_cardinality.py — Choice routing by cardinality (T6).

Hypothesis: the right routing signal is estimated cardinality, not semantic
similarity. Small cardinality → Holistic OR; large cardinality → parallel.

Strategies:
  holistic_only      → always Holistic OR (baseline)
  all_parallel       → always parallel LLMScan (baseline)
  card_10            → Holistic if est ≤ 10, parallel otherwise
  card_15            → Holistic if est ≤ 15, parallel otherwise
  card_20            → Holistic if est ≤ 20, parallel otherwise
  mugalois_hybrid    → current model (similarity-based, for comparison)
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

from mugalois.choice.choice_path import (
    ChoicePath, LLMChoiceConf, LLMEstimateChoiceSize,
)
from mugalois.choice.mugalois_choice import (
    MuGaloisChoice,
    _holistic_prompt, _motivational_prompt,
    _parallel_simple, _eval_complex_branch,
)
from mugalois.paths.path_query import PathQuery
from mugalois.core.types import Environment, RecursivePattern
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

QUERIES_PATH = Path(__file__).resolve().parent / "queries_t6_hybrid.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results_cardinality"

N_RUNS   = 5
VERBOSE  = True
COVERAGE = 0.80

QUERIES    = ["q7", "q8", "q9", "q10"]
STRATEGIES = [
    "holistic_only",
    "all_parallel",
    "card_10",
    "card_15",
    "card_20",
    "mugalois_hybrid",
]


# ── Build ChoicePath ──────────────────────────────────────────────────

def build_complex_branch(b: dict, source: str):
    if b["type"] == "T4":
        hops    = b["hops"]
        anchors = [source] + [f"?inter{i}" for i in range(len(hops)-1)] + ["?b"]
        return PathQuery.from_chain(anchors, hops, "?b")
    elif b["type"] == "T5":
        return RecursivePattern(
            s=source, p=b["predicate"], o="?b", operator=b["operator"]
        )
    raise ValueError(f"Unknown type: {b['type']}")


def build_choice_path(q: dict) -> ChoicePath:
    simple   = q.get("branches_simple", [])
    complex_ = [build_complex_branch(b, q["source"])
                for b in q.get("branches_complex", [])]
    return ChoicePath(
        source=q["source"], target_var=q["target_var"],
        branches=simple + complex_,
        direction=q.get("direction", "forward"),
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


# ── Cardinality-based routing ─────────────────────────────────────────

def _card_routing(path: ChoicePath, llm: BaseLLM, card_max: int) -> set[str]:
    """
    Route by cardinality:
      est ≤ card_max → Holistic OR + cardinality check
      est > card_max → parallel LLMScan
    """
    # Only simple branches for routing decision
    simple   = [b for b in path.branches if isinstance(b, str)]
    complex_ = [b for b in path.branches if not isinstance(b, str)]

    result = set()

    if simple:
        simple_path = ChoicePath(
            source=path.source, target_var=path.target_var,
            branches=simple, direction=path.direction,
        )
        est = LLMEstimateChoiceSize(simple_path, llm)
        print(f"  [card] est={est} card_max={card_max}")

        if est <= card_max:
            print(f"  [card] → Holistic OR (est ≤ {card_max})")
            prompt = _holistic_prompt(simple_path)
            r      = json_to_values(llm.chat(build_value_messages(prompt)).text)
            # motivational if below coverage
            if est > 0 and len(r) < COVERAGE * est:
                print(f"  [card] → motivational")
                follow   = _motivational_prompt(simple_path, r)
                new_vals = json_to_values(
                    llm.chat(build_value_messages(follow)).text
                ) - r
                r |= new_vals
            result |= r
        else:
            print(f"  [card] → parallel LLMScan (est > {card_max})")
            result |= _parallel_simple(simple, path, llm)

    # Complex branches always parallel
    if complex_:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(complex_)) as executor:
            futures = [
                executor.submit(_eval_complex_branch, b, path, llm, Environment())
                for b in complex_
            ]
            for f in as_completed(futures):
                try:
                    result |= f.result()
                except Exception as e:
                    print(f"  [complex] error: {e}")

    return result


# ── Strategy functions ────────────────────────────────────────────────

def run_holistic_only(q, path, tracker):
    simple = [b for b in path.branches if isinstance(b, str)]
    simple_path = ChoicePath(
        source=path.source, target_var=path.target_var,
        branches=simple, direction=path.direction,
    )
    prompt = _holistic_prompt(simple_path)
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)


def run_all_parallel(q, path, tracker):
    simple   = [b for b in path.branches if isinstance(b, str)]
    complex_ = [b for b in path.branches if not isinstance(b, str)]
    result   = _parallel_simple(simple, path, tracker)
    for b in complex_:
        result |= _eval_complex_branch(b, path, tracker, Environment())
    return result


def run_card_10(q, path, tracker):
    return _card_routing(path, tracker, card_max=10)

def run_card_15(q, path, tracker):
    return _card_routing(path, tracker, card_max=15)

def run_card_20(q, path, tracker):
    return _card_routing(path, tracker, card_max=20)

def run_mugalois_hybrid(q, path, tracker):
    return MuGaloisChoice(path, tracker, tau_high=0.60, tau_low=0.40,
                          verbose=VERBOSE)


STRATEGY_REGISTRY = {
    "holistic_only":   (run_holistic_only,   "HolisticOnly"),
    "all_parallel":    (run_all_parallel,    "AllParallel"),
    "card_10":         (run_card_10,         "Card≤10"),
    "card_15":         (run_card_15,         "Card≤15"),
    "card_20":         (run_card_20,         "Card≤20"),
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
        print(f" GT={len(gt)}  direction={q.get('direction','forward')}")
        print(f" Strategies: {', '.join(strategies_to_run)}")
        print(f"{'='*72}")

        report = Report(
            query_id=qid, template="T6-card",
            predicate=q["description"], n_expected=len(gt),
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
    print(f"[run_cardinality.py T6] model={args.model}  dry_run={args.dry_run}")
    main(base_llm, queries_to_run, strategies_to_run)
