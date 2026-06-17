"""
experiments/template3/exp_wilde.py

Focused experiment on q2 (Oscar Wilde comedies) — GD optimal query.

Models:
  NL
  SimpleScan
  GD                 standard left-to-right
  GD_motiv0          motivational in iter 0 of hop 1
  GD_motiv_closure   motivational as closure (iter > 0) on hop 1
  GD_telephone       telephone arabe on hop 1
  DG                 right-to-left (sub-optimal direction)
  MuGalois           full orchestrator

Usage
-----
    python3 -m experiments.template3.exp_wilde
    python3 -m experiments.template3.exp_wilde --dry-run
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

from mugalois.core.types import TriplePattern, Environment
from mugalois.core.prompts import (
    genNLPrompt, genSimpleScanPathPrompt,
    build_value_messages, build_messages, MOTIVATIONAL,
)
from mugalois.core.parser import json_to_values
from mugalois.core.operators import path_join
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.paths.gd import scan_gd
from mugalois.paths.dg import scan_dg
from mugalois.paths.simple import LLMSimpleScan
from mugalois.paths.mugalois import MuGaloisPath
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

# ── Query ─────────────────────────────────────────────────────────────────────

S  = "Oscar Wilde"
P1 = "wrote"
P2 = "genre"
T  = "comedy"

GT = {
    "A Woman of No Importance",
    "An Ideal Husband",
    "Lady Windermere's Fan",
    "The Importance of Being Earnest",
}

NL_PROMPT = "which plays written by Oscar Wilde belong to the comedy genre?"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_RUNS = 5


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


# ── Telephone arabe on hop 1 ──────────────────────────────────────────────────

def _telephone_hint(found: set, tracker: TrackingLLM) -> str:
    """Coach gives specific names to help find more works by Oscar Wilde."""
    found_str = ", ".join(sorted(found)[:15])
    prompt = (
        f"Your friend is looking for works written by Oscar Wilde.\n"
        f"They already found: {found_str}.\n"
        f"They are stuck. Give them 3-5 specific titles they haven't found yet.\n"
        f"Only titles, comma-separated. No explanation."
    )
    messages = [
        {"role": "system", "content":
         "You are a helpful friend. Give only a comma-separated list of titles."},
        {"role": "user", "content": prompt},
    ]
    return tracker.chat(messages).text.strip()


def scan_gd_telephone(
    s: str, p1: str, p2: str, t: str,
    tracker: TrackingLLM,
    max_iter: int = 5,
    n_break: int = 2,
) -> set[str]:
    """
    GD with telephone arabe on hop 1.
    When hop 1 gets stuck → coach gives specific titles → scanner continues.
    """
    from mugalois.core.prompts import SYSTEM_PROMPT, genTripleScanPrompt, genTripleScanIterativePrompt
    from mugalois.core.parser import json_to_triples

    pat1    = TriplePattern(s, p1, "?b")
    T       = set()
    ctx     = []
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    coached_empty = 0
    n_coach = 0

    for i in range(max_iter):
        if i == 0:
            prompt = genTripleScanPrompt(pat1, encoding="pattern")
        elif coached_empty > 0:
            # telephone arabe
            hint = _telephone_hint({t.o for t in T}, tracker)
            n_coach += 1
            already = ", ".join(sorted(t.o for t in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"A friend suggests these titles: {hint}\n\n"
                f"Find triples for these suggestions and any others you know. "
                f"Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )
        else:
            prompt = genTripleScanIterativePrompt(T, pattern=pat1)

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        resp     = tracker.chat(messages)
        T_new    = json_to_triples(resp.text)

        if T_new.issubset(T):
            coached_empty += 1
            if coached_empty >= n_break:
                break
        else:
            coached_empty = 0
            ctx.append({"role": "user",      "content": prompt})
            ctx.append({"role": "assistant", "content": resp.text})
            T |= T_new

    seeds_b = {t.o for t in T}
    if not seeds_b:
        return set()

    # Hop 2
    pat2 = TriplePattern("?b", p2, t)
    env2 = Environment()
    env2.set("?b", seeds_b)
    T2 = LLMScan(pat2, copy.deepcopy(env2), Environment(), tracker, max_iter=3)
    return {t.s for t in T2}


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(run_fn, n, tracker, last) -> AggregatedScores:
    scores = []
    for i in range(n):
        tracker.reset()
        values = run_fn()
        score  = Metrics.compute(values, GT, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        print(f"      run {i+1}/{n} → returned={len(values)} "
              f"expected={len(GT)} | "
              f"P={score.precision:.3f} R={score.recall:.3f} "
              f"F1={score.f1:.3f} "
              f"t={score.time_s:.1f}s tok={score.tokens}")
        if i == n - 1:
            last.append(values)
        scores.append(score)
    return AggregatedScores.from_runs(scores)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_llm):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = TrackingLLM(base_llm)

    print(f"\n{'='*60}")
    print(f"  Oscar Wilde Comedies — GD optimal experiment")
    print(f"  Pattern: ({S}, {P1}, ?b) AND (?b, {P2}, {T})")
    print(f"  GT size: {len(GT)}")
    print(f"{'='*60}")

    report = Report(
        query_id="wilde_comedies",
        template="T3_focused",
        predicate=f"{P1}/{P2}",
        n_expected=len(GT),
    )

    # ── NL ────────────────────────────────────────────────────────────────────
    print("\n--- NL ---")
    last = []
    agg  = _aggregate(
        lambda: json_to_values(
            tracker.chat(build_value_messages(genNLPrompt(NL_PROMPT))).text),
        N_RUNS, tracker, last)
    report.add("NL", agg, actual=last[0], expected=GT)

    # ── SimpleScan ────────────────────────────────────────────────────────────
    print("\n--- SimpleScan ---")
    last = []
    agg  = _aggregate(
        lambda: LLMSimpleScan(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("SimpleScan", agg, actual=last[0], expected=GT)

    # ── GD ────────────────────────────────────────────────────────────────────
    print("\n--- GD ---")
    last = []
    agg  = _aggregate(
        lambda: scan_gd(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("GD", agg, actual=last[0], expected=GT)

    # ── GD_motiv0 ─────────────────────────────────────────────────────────────
    print("\n--- GD_motiv0 ---")
    last = []
    agg  = _aggregate(
        lambda: scan_gd(S, P1, P2, T, tracker, motivational=True),
        N_RUNS, tracker, last)
    report.add("GD_motiv0", agg, actual=last[0], expected=GT)

    # ── GD_motiv_closure ──────────────────────────────────────────────────────
    print("\n--- GD_motiv_closure ---")
    from mugalois.closures.motivational import MotivationalClosure
    from mugalois.closures.pipeline import ClosurePipeline
    pipeline = ClosurePipeline([MotivationalClosure()], n_break=1, max_iter=8)
    last = []
    agg  = _aggregate(
        lambda: scan_gd(S, P1, P2, T, tracker, pipeline=pipeline),
        N_RUNS, tracker, last)
    report.add("GD_motiv_closure", agg, actual=last[0], expected=GT)

    # ── GD_telephone ──────────────────────────────────────────────────────────
    print("\n--- GD_telephone ---")
    last = []
    agg  = _aggregate(
        lambda: scan_gd_telephone(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("GD_telephone", agg, actual=last[0], expected=GT)

    # ── DG ────────────────────────────────────────────────────────────────────
    print("\n--- DG ---")
    last = []
    agg  = _aggregate(
        lambda: scan_dg(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("DG", agg, actual=last[0], expected=GT)

    # ── MuGalois ──────────────────────────────────────────────────────────────
    print("\n--- MuGalois ---")
    last = []
    agg  = _aggregate(
        lambda: MuGaloisPath(S, P1, P2, T, tracker, verbose=True),
        N_RUNS, tracker, last)
    report.add("MuGalois", agg, actual=last[0], expected=GT)

    # ── Table ─────────────────────────────────────────────────────────────────
    report.print_table()
    report.save_json(RESULTS_DIR / "wilde_comedies.json")
    report.save_csv(RESULTS_DIR  / "wilde_comedies.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(MockLLM() if args.dry_run else AzureOpenAIClient())