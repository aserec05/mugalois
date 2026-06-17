"""
experiments/template3/exp_dc.py

Focused experiment on DC Prime Ministers — DG optimal query.

SELECT ?b WHERE (Democrazia Cristiana, member, ?b) AND (?b, heldPosition, Prime Minister of Italy)

Models:
  NL
  SimpleScan
  GD                 left-to-right (sub-optimal)
  DG                 right-to-left (optimal)
  DG_motiv0          motivational in iter 0 of hop 1
  DG_motiv_closure   motivational as closure on hop 1
  DG_telephone       telephone arabe on hop 1
  MuGalois           full orchestrator — does it choose DG ?

Usage
-----
    python3 -m experiments.template3.exp_dc
    python3 -m experiments.template3.exp_dc --dry-run
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
    genNLPrompt, MOTIVATIONAL,
    build_value_messages, SYSTEM_PROMPT,
    genTripleScanPrompt, genTripleScanIterativePrompt,
)
from mugalois.core.parser import json_to_values, json_to_triples
from mugalois.core.operators import path_join
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.paths.gd import scan_gd
from mugalois.paths.dg import scan_dg
from mugalois.paths.simple import LLMSimpleScan
from mugalois.paths.join import scan_join
from mugalois.paths.mugalois import MuGaloisPath
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

# ── Query ─────────────────────────────────────────────────────────────────────

S  = "United Nations"
P1 = "member"
P2 = "official language"
T  = "Portuguese"

GT = {
    "Angola", "Brazil", "Cape Verde",
    "Equatorial Guinea", "Guinea-Bissau", "Mozambique",
    "Portugal", "São Tomé and Príncipe", "Timor-Leste",
}

NL_PROMPT = "which UN member countries have Portuguese as their official language?"

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


# ── Telephone arabe on hop 1 (DG side) ───────────────────────────────────────

def _telephone_hint_dg(found: set, tracker: TrackingLLM) -> str:
    found_str = ", ".join(sorted(found)[:15])
    prompt = (
        f"Your friend is looking for people who held the position of "
        f"Prime Minister of Italy.\n"
        f"They already found: {found_str}.\n"
        f"They are stuck. Give them 3-5 specific names they haven't found yet.\n"
        f"Only names, comma-separated. No explanation."
    )
    messages = [
        {"role": "system", "content":
         "You are a helpful friend. Give only a comma-separated list of names."},
        {"role": "user", "content": prompt},
    ]
    return tracker.chat(messages).text.strip()


def scan_dg_telephone(
    s: str, p1: str, p2: str, t: str,
    tracker: TrackingLLM,
    max_iter: int = 8,
    n_break: int = 2,
) -> set[str]:
    """DG with telephone arabe on hop 1 (right side scan)."""
    pat1    = TriplePattern("?b", p2, t)
    T       = set()
    ctx     = []
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    coached_empty = 0

    for i in range(max_iter):
        if i == 0:
            prompt = genTripleScanPrompt(pat1, encoding="pattern")
        elif coached_empty > 0:
            hint = _telephone_hint_dg({tr.s for tr in T}, tracker)
            already = ", ".join(sorted(tr.s for tr in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"A friend suggests: {hint}\n\n"
                f"Find triples for these suggestions and any others. "
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

    seeds_b = {tr.s for tr in T}
    if not seeds_b:
        return set()

    # Hop 2 — filter by DC membership
    pat2 = TriplePattern(s, p1, "?b")
    env2 = Environment()
    env2.set("?b", seeds_b)
    T2 = LLMScan(pat2, copy.deepcopy(env2), Environment(), tracker, max_iter=3)
    return {tr.o for tr in T2}


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
    print(f"  UN Portuguese-speaking countries — DG optimal experiment")
    print(f"  Pattern: ({S}, {P1}, ?b) AND (?b, {P2}, {T})")
    print(f"  GT size: {len(GT)}")
    print(f"{'='*60}")

    report = Report(
        query_id="un_portuguese",
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

    # ── DG ────────────────────────────────────────────────────────────────────
    print("\n--- DG ---")
    last = []
    agg  = _aggregate(
        lambda: scan_dg(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("DG", agg, actual=last[0], expected=GT)

    # ── DG_motiv0 ─────────────────────────────────────────────────────────────
    print("\n--- DG_motiv0 ---")
    last = []
    agg  = _aggregate(
        lambda: scan_dg(S, P1, P2, T, tracker, motivational=True),
        N_RUNS, tracker, last)
    report.add("DG_motiv0", agg, actual=last[0], expected=GT)

    # ── DG_motiv_closure ──────────────────────────────────────────────────────
    print("\n--- DG_motiv_closure ---")
    from mugalois.closures.motivational import MotivationalClosure
    from mugalois.closures.pipeline import ClosurePipeline
    pipeline = ClosurePipeline([MotivationalClosure()], n_break=2, max_iter=8)
    last = []
    agg  = _aggregate(
        lambda: scan_dg(S, P1, P2, T, tracker, pipeline=pipeline),
        N_RUNS, tracker, last)
    report.add("DG_motiv_closure", agg, actual=last[0], expected=GT)

    # ── DG_telephone ──────────────────────────────────────────────────────────
    print("\n--- DG_telephone ---")
    last = []
    agg  = _aggregate(
        lambda: scan_dg_telephone(S, P1, P2, T, tracker),
        N_RUNS, tracker, last)
    report.add("DG_telephone", agg, actual=last[0], expected=GT)

    # ── MuGalois ──────────────────────────────────────────────────────────────
    print("\n--- MuGalois ---")
    last = []
    agg  = _aggregate(
        lambda: MuGaloisPath(S, P1, P2, T, tracker, tau_simple=0.8, verbose=True),
        N_RUNS, tracker, last)
    report.add("MuGalois", agg, actual=last[0], expected=GT)

    # ── MuGalois_lowTau (tau_simple=0.5) ─────────────────────────────────────
    print("\n--- MuGalois_lowTau ---")
    last = []
    agg  = _aggregate(
        lambda: MuGaloisPath(S, P1, P2, T, tracker, tau_simple=0.5, verbose=True),
        N_RUNS, tracker, last)
    report.add("MuGalois_lowTau", agg, actual=last[0], expected=GT)

    # ── Table ─────────────────────────────────────────────────────────────────
    report.print_table()
    report.save_json(RESULTS_DIR / "dc_prime_ministers.json")
    report.save_csv(RESULTS_DIR  / "dc_prime_ministers.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(MockLLM() if args.dry_run else AzureOpenAIClient())