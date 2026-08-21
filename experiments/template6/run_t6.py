"""
experiments/template6/run.py — Choice path queries (T6).

Ablation study:
  Encodings : list / or / natural
  Taus      : high / mid / low
  Baselines : nl_naive / nl_precise / union_branches / mugalois
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

from mugalois.choice.choice_path import ChoicePath, LLMChoiceConf, LLMEstimateChoiceSize
from mugalois.choice.mugalois_choice import MuGaloisChoice, _eval_branch, CALM_FOLLOW_UP
from mugalois.core.prompts import build_value_messages, build_messages
from mugalois.core.parser import json_to_values
from mugalois.core.types import Environment
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

QUERIES_PATH = Path(__file__).resolve().parent / "queries_t6.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

N_RUNS  = 5
VERBOSE = False

QUERIES = ["q1", "q2", "q3", "q4", "q5", "q6"]

STRATEGIES = [
    "nl_naive",
    "nl_precise",
    "union_branches",
    "encoding_list",
    "encoding_or",
    "encoding_natural",
    "mugalois_high",
    "mugalois_mid",
    "mugalois_low",
    "mugalois",
]


# ── ChoicePath builder ────────────────────────────────────────────────

def build_choice_path(q: dict) -> ChoicePath:
    return ChoicePath(
        source     = q["source"],
        target_var = q["target_var"],
        branches   = q["branches_nl"],
    )


# ── Encoding variants ─────────────────────────────────────────────────

def _prompt_list(path: ChoicePath) -> str:
    lines = "\n".join(
        f"  - {path.target_var} {b} {path.source}"
        for b in path.branches
    )
    return (
        f"Task: List all values of {path.target_var} such that:\n{lines}\n\n"
        f"Be exhaustive — there are likely several results.\n"
        f"List every single value you know.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )

def _prompt_or(path: ChoicePath) -> str:
    """OR encoding."""
    branches_or = " OR ".join(f"[{b}]" for b in path.branches)
    return (
        f"Task: List all values of {path.target_var} such that "
        f"{path.source} {branches_or} {path.target_var}.\n\n"
        f"Be exhaustive — there are likely several results.\n"
        f"List every single value you know.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )

def _prompt_natural(path: ChoicePath) -> str:
    """Natural language encoding — LLM generates the question itself."""
    branches_str = ", ".join(path.branches)
    gen_prompt = (
        f"Convert this into a single natural language question:\n"
        f"Find all {path.target_var} related to {path.source} via "
        f"any of these relations: {branches_str}.\n\n"
        f"Write the question only. No explanation."
    )
    # We use _prompt_list as fallback — natural is generated at runtime
    return _prompt_list(path)  # overridden in run function


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

def run_union_branches(q, path, tracker):
    """One call per branch, union — no confidence, no holistic."""
    result = set()
    for branch in path.branches:
        single = ChoicePath(path.source, path.target_var, [branch])
        prompt = _prompt_list(single)
        vals   = json_to_values(tracker.chat(build_value_messages(prompt)).text)
        result |= vals
    return result

def run_encoding_list(q, path, tracker):
    prompt = _prompt_list(path)
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)

def run_encoding_or(q, path, tracker):
    prompt = _prompt_or(path)
    return json_to_values(tracker.chat(build_value_messages(prompt)).text)

def run_encoding_natural(q, path, tracker):
    """LLM generates its own NL question then answers it."""
    branches_str = ", ".join(path.branches)
    gen_prompt = (
        f"Convert this into a single natural language question:\n"
        f"Find all {path.target_var} related to {path.source} via "
        f"any of these relations: {branches_str}.\n\n"
        f"Write the question only. No explanation."
    )
    nl_q = tracker.chat(build_messages(gen_prompt)).text.strip()
    full = (
        f"{nl_q}\n\n"
        f"Be exhaustive — there are likely several results.\n"
        f"List every single one you know.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    return json_to_values(tracker.chat(build_value_messages(full)).text)

def run_mugalois_high(q, path, tracker):
    return MuGaloisChoice(path, tracker, tau_high=0.70, tau_low=0.50, verbose=VERBOSE)

def run_mugalois_mid(q, path, tracker):
    return MuGaloisChoice(path, tracker, tau_high=0.60, tau_low=0.40, verbose=VERBOSE)

def run_mugalois_low(q, path, tracker):
    return MuGaloisChoice(path, tracker, tau_high=0.50, tau_low=0.30, verbose=VERBOSE)

def run_mugalois(q, path, tracker):
    """Best config — determined after ablation."""
    return MuGaloisChoice(path, tracker, tau_high=0.60, tau_low=0.40, verbose=VERBOSE)


STRATEGY_REGISTRY = {
    "nl_naive":        (run_nl_naive,        "NL_naive"),
    "nl_precise":      (run_nl_precise,      "NL_precise"),
    "union_branches":  (run_union_branches,  "UnionBranches"),
    "encoding_list":   (run_encoding_list,   "Encoding_list"),
    "encoding_or":     (run_encoding_or,     "Encoding_OR"),
    "encoding_natural":(run_encoding_natural,"Encoding_natural"),
    "mugalois_high":   (run_mugalois_high,   "MuGalois_τ_high"),
    "mugalois_mid":    (run_mugalois_mid,    "MuGalois_τ_mid"),
    "mugalois_low":    (run_mugalois_low,    "MuGalois_τ_low"),
    "mugalois":        (run_mugalois,        "MuGaloisChoice"),
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
        print(f" Source : {q['source']}")
        print(f" Branches ({q['n_branches']}) : {' | '.join(q['branches_nl'])}")
        print(f" GT={len(gt)}  optimal={q['optimal_strategy']}")
        print(f" Strategies: {', '.join(strategies_to_run)}")
        print(f"{'='*72}")

        report = Report(
            query_id=qid, template="T6",
            predicate=" | ".join(q["branches_nl"]),
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
    print(f"[run.py T6] model={args.model}  dry_run={args.dry_run}")
    main(base_llm, queries_to_run, strategies_to_run)
