"""
experiments/template2/exp_closures.py

Closure pipeline experiment — compare different pipeline configurations.

Pipelines tested:
  P0 : []                          (standard — no closures)
  P1 : [Motivational]
  P2 : [Telephone]
  P3 : [Motivational, Telephone]   (escalate: motiv → telephone)
  P4 : [Telephone, Motivational]   (escalate: telephone → motiv)
  P5 : [Motivational, Coach, Telephone] (full escalation)

Usage
-----
    python3 -m experiments.template2.exp_closures
    python3 -m experiments.template2.exp_closures --exp baroque
    python3 -m experiments.template2.exp_closures --exp oscar
    python3 -m experiments.template2.exp_closures --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.core.types import TriplePattern
from mugalois.closures.motivational import MotivationalClosure
from mugalois.closures.coach import CoachClosure
from mugalois.closures.telephone import TelephoneClosure
from mugalois.closures.pipeline import ClosurePipeline
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

RESULTS_DIR = Path(__file__).resolve().parent / "results"

N_RUNS   = 5
N_BREAK  = 2
MAX_ITER = 15


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


# ── Runner ────────────────────────────────────────────────────────────────────

def run_pipeline(
    label:    str,
    pipeline: ClosurePipeline,
    pattern:  TriplePattern,
    gt:       set[str],
    tracker:  TrackingLLM,
    n_runs:   int,
    report:   Report,
):
    scores   = []
    n_iters  = []
    all_stats = []

    for i in range(n_runs):
        tracker.reset()
        T, n_iter, stats = pipeline.run(pattern, tracker)
        values = {t.s for t in T}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        n_iters.append(n_iter)
        all_stats.append(stats)

        stats_str = " ".join(f"{k}={v}" for k, v in stats.items() if v > 0)
        print(f"    {label} run {i+1}/{n_runs} → "
              f"returned={len(values)} expected={len(gt)} | "
              f"P={score.precision:.3f} R={score.recall:.3f} "
              f"F1={score.f1:.3f} "
              f"iter={n_iter} t={score.time_s:.1f}s tok={score.tokens}"
              + (f" [{stats_str}]" if stats_str else ""))
        scores.append(score)

    agg = AggregatedScores.from_runs(scores)
    print(f"  → {label}: F1={agg.f1_mean:.3f}±{agg.f1_std:.3f} "
          f"tok={agg.tokens_mean:.0f} t={agg.time_mean:.1f}s "
          f"iter={round(statistics.mean(n_iters),1)}")
    report.add(label, agg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_llm, exp: str):
    gt_path = (
        Path(__file__).resolve().parent / "gt_baroque_composers.json"
        if exp == "baroque"
        else Path(__file__).resolve().parent / "gt_oscar_films.json"
    )
    gt = set(json.loads(gt_path.read_text(encoding="utf-8"))) \
         if gt_path.exists() else set()

    pattern = (
        TriplePattern("?composer", "style", "Baroque")
        if exp == "baroque"
        else TriplePattern("?film", "award", "Academy Award")
    )

    tracker = TrackingLLM(base_llm)

    # ── Define pipelines ──────────────────────────────────────────────────────
    M = MotivationalClosure()
    C = CoachClosure()
    T = TelephoneClosure(n_names=5)

    pipelines = [
        ("P0_Standard",         ClosurePipeline([],          n_break=1, max_iter=MAX_ITER)),
        ("P1_Motivational",     ClosurePipeline([M],         n_break=1, max_iter=MAX_ITER)),
        ("P2_Telephone",        ClosurePipeline([T],         n_break=N_BREAK, max_iter=MAX_ITER)),
        ("P3_Motiv_Telephone",  ClosurePipeline([M, T],      n_break=N_BREAK, max_iter=MAX_ITER)),
        ("P4_Tel_Motiv",        ClosurePipeline([T, M],      n_break=N_BREAK, max_iter=MAX_ITER)),
        ("P5_Full",             ClosurePipeline([M, C, T],   n_break=N_BREAK, max_iter=MAX_ITER)),
    ]

    print(f"\n{'='*60}")
    print(f"  Closure Pipeline Experiment — {exp}")
    print(f"  Pattern : {pattern}")
    print(f"  GT size : {len(gt)}")
    print(f"  N_BREAK : {N_BREAK}")
    print(f"{'='*60}")

    report = Report(
        query_id=f"closures_{exp}",
        template="closure_study",
        predicate=pattern.p,
        n_expected=len(gt),
    )

    for label, pipeline in pipelines:
        run_pipeline(label, pipeline, pattern, gt, tracker, N_RUNS, report)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_json(RESULTS_DIR / f"closures_{exp}.json")
    report.save_csv(RESULTS_DIR  / f"closures_{exp}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     default="oscar",
                        choices=["baroque", "oscar"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    main(base_llm, args.exp)
