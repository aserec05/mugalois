"""
experiments/template5/ablation.py

Ablation study for LLMRecScan: isolates the contribution of the
motivational follow-up and the proportional cap, independently.

Configurations:
  A) motivational=False, capped=False  -> single-call baseline
  B) motivational=True,  capped=False  -> motivational, unguarded
  C) motivational=False, capped=True   -> no-op (no follow-up to cap)
  D) motivational=True,  capped=True   -> current default (both)

Usage:
    python3 -m experiments.template5.ablation
    python3 -m experiments.template5.ablation --query q1
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

from mugalois.core.types import RecursivePattern
from mugalois.rec.simple_rec import LLMRecScan
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report


QUERIES_PATH = Path(__file__).resolve().parent / "queries_t5.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results_ablation"

N_RUNS = 5
# Exclude q2 (very_high cardinality, different GT format from the
# predicate-rename session) — keep the ablation on directly comparable
# medium/high/very_high queries.
QUERIES = ["q1", "q3", "q4", "q5", "q7"]

CONFIGS = {
    "A_baseline":            dict(motivational=False, capped=False),
    "B_motivational_only":   dict(motivational=True,  capped=False),
    "C_cap_only":            dict(motivational=False, capped=True),
    "D_motivational_capped": dict(motivational=True,  capped=True),
}


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


def build_pattern(q: dict) -> RecursivePattern:
    return RecursivePattern(q["s"], q["p"], q["o"], q["operator"])


def _aggregate(run_fn, gt, n, tracker):
    scores = []
    for i in range(n):
        tracker.reset()
        values = {str(v) for v in run_fn()}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        print(f"      run {i+1}/{n} -> returned={len(values)} "
              f"expected={len(gt)} | P={score.precision:.3f} "
              f"R={score.recall:.3f} F1={score.f1:.3f} "
              f"t={score.time_s:.1f}s tok={score.tokens}")
        scores.append(score)
    return AggregatedScores.from_runs(scores)


def main(base_llm, queries_to_run):
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q  = queries[qid]
        gt = set(q["ground_truth"])
        pattern = build_pattern(q)

        print(f"\n{'='*70}")
        print(f" Ablation | {qid} | {q['description']}")
        print(f" Pattern : {pattern}")
        print(f" GT size : {len(gt)}")
        print(f"{'='*70}")

        report = Report(
            query_id=qid, template="T5-ablation",
            predicate=q["p"], n_expected=len(gt),
        )

        for config_name, kwargs in CONFIGS.items():
            print(f"\n  -- {config_name} ({kwargs}) --")
            agg = _aggregate(
                lambda: LLMRecScan(pattern, tracker, **kwargs),
                gt, N_RUNS, tracker,
            )
            report.add(config_name, agg, actual=set(), expected=gt)

        report.print_table()
        report.save_json(RESULTS_DIR / f"{qid}_ablation.json")
        report.save_csv(RESULTS_DIR / f"{qid}_ablation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    queries_to_run = [args.query] if args.query else QUERIES
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()

    print(f"[ablation.py] dry_run={args.dry_run}")
    main(base_llm, queries_to_run)