"""
experiments/template3/exp_mugalois_only.py

Runs Join, MuGalois and MuGalois_lowTau on all three T3 queries.

Queries:
  wilde        : Oscar Wilde comedies (GD optimal, GT=4)
  portuguese   : UN Portuguese-speaking countries (DG optimal, GT=9)
  commonwealth : African Commonwealth countries (DG optimal, GT=21)

Usage
-----
    python3 -m experiments.template3.exp_mugalois_only
    python3 -m experiments.template3.exp_mugalois_only --query wilde
    python3 -m experiments.template3.exp_mugalois_only --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.paths.join import scan_join
from mugalois.paths.mugalois import MuGaloisPath
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_RUNS = 5

# ── Queries ───────────────────────────────────────────────────────────────────

QUERIES = {
    "wilde": {
        "description": "Oscar Wilde comedies — GD optimal",
        "s": "Oscar Wilde", "p1": "wrote", "p2": "genre", "t": "comedy",
        "optimal": "GD",
        "gt": {
            "A Woman of No Importance", "An Ideal Husband",
            "Lady Windermere's Fan", "The Importance of Being Earnest",
        },
    },
    "portuguese": {
        "description": "UN Portuguese-speaking countries — DG optimal",
        "s": "United Nations", "p1": "member", "p2": "official language", "t": "Portuguese",
        "optimal": "DG",
        "gt": {
            "Angola", "Brazil", "Cape Verde", "Equatorial Guinea",
            "Guinea-Bissau", "Mozambique", "Portugal",
            "São Tomé and Príncipe", "Timor-Leste",
        },
    },
    "commonwealth": {
        "description": "African Commonwealth countries — DG optimal, high cardinality",
        "s": "Africa", "p1": "country", "p2": "member of", "t": "Commonwealth",
        "optimal": "DG",
        "gt": {
            "Botswana", "Cameroon", "Eswatini", "Gabon", "Ghana",
            "Kenya", "Lesotho", "Malawi", "Mauritius", "Mozambique",
            "Namibia", "Nigeria", "Rwanda", "Seychelles", "Sierra Leone",
            "South Africa", "Tanzania", "The Gambia", "Togo", "Uganda", "Zambia",
        },
    },
}


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

def main(base_llm, queries_to_run):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    for qid in queries_to_run:
        q  = QUERIES[qid]
        gt = q["gt"]
        s, p1, p2, t = q["s"], q["p1"], q["p2"], q["t"]

        print(f"\n{'='*60}")
        print(f"  {qid} | {q['description']}")
        print(f"  Pattern: ({s}, {p1}, ?b) AND (?b, {p2}, {t})")
        print(f"  Optimal: {q['optimal']} | GT size: {len(gt)}")
        print(f"{'='*60}")

        report = Report(
            query_id=f"mugalois_{qid}",
            template="T3_mugalois",
            predicate=f"{p1}/{p2}",
            n_expected=len(gt),
        )

        # ── Join ──────────────────────────────────────────────────────────────
        print("\n--- Join ---")
        last = []
        agg  = _aggregate(
            lambda: scan_join(s, p1, p2, t, tracker),
            gt, N_RUNS, tracker, last)
        report.add("Join", agg, actual=last[0], expected=gt)

        # ── MuGalois (tau=0.8, with closure) ──────────────────────────────────
        print("\n--- MuGalois ---")
        last = []
        agg  = _aggregate(
            lambda: MuGaloisPath(s, p1, p2, t, tracker,
                                 tau_simple=0.8, use_closure=True, verbose=True),
            gt, N_RUNS, tracker, last)
        report.add("MuGalois", agg, actual=last[0], expected=gt)

        # ── MuGalois_lowTau (tau=0.5, with closure) ───────────────────────────
        print("\n--- MuGalois_lowTau ---")
        last = []
        agg  = _aggregate(
            lambda: MuGaloisPath(s, p1, p2, t, tracker,
                                 tau_simple=0.5, use_closure=True, verbose=True),
            gt, N_RUNS, tracker, last)
        report.add("MuGalois_lowTau", agg, actual=last[0], expected=gt)

        report.print_table()
        report.save_json(RESULTS_DIR / f"mugalois_{qid}.json")
        report.save_csv(RESULTS_DIR  / f"mugalois_{qid}.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query",   default=None,
                        choices=["wilde", "portuguese", "commonwealth"])
    args = parser.parse_args()

    queries_to_run = [args.query] if args.query else list(QUERIES.keys())
    main(MockLLM() if args.dry_run else AzureOpenAIClient(), queries_to_run)