"""
experiments/template2/exp_motiv_coach.py

Simple experiment: Motivational vs Motivational+Coach(on empty).

Flow Motivational:
  iter 0   : standard prompt
  iter 1+  : motivational prompt
  empty    : break

Flow Motivational+Coach:
  iter 0   : standard prompt
  iter 1+  : motivational prompt
  empty    : coach provides hint → ONE more attempt → break

Usage
-----
    python3 -m experiments.template2.exp_motiv_coach
    python3 -m experiments.template2.exp_motiv_coach --dry-run
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
from mugalois.core.prompts import genTripleScanPrompt, SYSTEM_PROMPT
from mugalois.core.parser import json_to_triples
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, AggregatedScores
from evaluation.report import Report

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GT_PATH     = Path(__file__).resolve().parent / "gt_baroque_composers.json"

N_RUNS   = 5
MAX_ITER = 10

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)


# ── TrackingLLM ───────────────────────────────────────────────────────────────

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


# ── Coach ─────────────────────────────────────────────────────────────────────

def get_coach_hint(predicate: str, obj: str, already_found: set,
                   tracker: TrackingLLM) -> str:
    found_str = ", ".join(sorted(str(x) for x in already_found)[:20])
    prompt = (
        f"A friend is trying to find all entities with {predicate} \"{obj}\".\n"
        f"They already found: {found_str}.\n"
        f"They are struggling to find rare or less well-known examples.\n"
        f"Help them in ONE sentence starting with: \"You should look for...\""
    )
    messages = [
        {"role": "system", "content":
         "You are a helpful assistant. Respond with exactly one sentence."},
        {"role": "user", "content": prompt},
    ]
    return tracker.chat(messages).text.strip()


# ── Scans ─────────────────────────────────────────────────────────────────────

def motivational_scan(
    pattern:  TriplePattern,
    tracker:  TrackingLLM,
    max_iter: int = MAX_ITER,
) -> tuple[set, int]:
    T   = set()
    ctx = []
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}

    for i in range(max_iter):
        if i == 0:
            prompt = genTripleScanPrompt(pattern, encoding="pattern")
        else:
            already = ", ".join(sorted(str(t) for t in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"{MOTIVATIONAL_PROMPT}\n"
                f"Find more triples. Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        resp     = tracker.chat(messages)
        T_new    = json_to_triples(resp.text)

        if T_new.issubset(T) and i > 0:
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        T |= T_new

    return T, i + 1


def motivational_coach_scan(
    pattern:  TriplePattern,
    tracker:  TrackingLLM,
    max_iter: int = MAX_ITER,
) -> tuple[set, int, int]:
    """
    Motivational + Coach on empty.
    On empty → coach provides ONE hint → ONE more attempt → break.
    """
    T        = set()
    ctx      = []
    sys_msg  = {"role": "system", "content": SYSTEM_PROMPT}
    coached  = False
    n_coach  = 0

    for i in range(max_iter):
        if i == 0:
            prompt = genTripleScanPrompt(pattern, encoding="pattern")
        else:
            already = ", ".join(sorted(str(t) for t in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"{MOTIVATIONAL_PROMPT}\n"
                f"Find more triples. Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        resp     = tracker.chat(messages)
        T_new    = json_to_triples(resp.text)

        if T_new.issubset(T) and i > 0:
            if coached:
                # already tried coach → break
                break
            # call coach once
            hint = get_coach_hint(pattern.p, pattern.o, T, tracker)
            n_coach += 1
            coached = True
            already = ", ".join(sorted(str(t) for t in T))
            coach_prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"Hint from a friend: {hint}\n\n"
                f"Find more triples following the hint. "
                f"Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )
            messages = [sys_msg, *ctx,
                        {"role": "user", "content": coach_prompt}]
            resp  = tracker.chat(messages)
            T_new = json_to_triples(resp.text)

            if T_new.issubset(T):
                break

            ctx.append({"role": "user",      "content": coach_prompt})
            ctx.append({"role": "assistant", "content": resp.text})
            T |= T_new
            coached = False  # reset — allow one more coach if needed
        else:
            coached = False
            ctx.append({"role": "user",      "content": prompt})
            ctx.append({"role": "assistant", "content": resp.text})
            T |= T_new

    return T, i + 1, n_coach


# ── Runner ────────────────────────────────────────────────────────────────────

def run(label, scan_fn, gt, tracker, n_runs, report):
    scores   = []
    n_iters  = []
    n_coachs = []

    for i in range(n_runs):
        tracker.reset()
        result  = scan_fn()
        T       = result[0]
        n_iter  = result[1]
        n_coach = result[2] if len(result) > 2 else 0

        values = {t.s for t in T}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        n_iters.append(n_iter)
        n_coachs.append(n_coach)

        print(f"    {label} run {i+1}/{n_runs} → "
              f"returned={len(values)} expected={len(gt)} | "
              f"P={score.precision:.3f} R={score.recall:.3f} "
              f"F1={score.f1:.3f} "
              f"iter={n_iter} coach={n_coach} "
              f"t={score.time_s:.1f}s tok={score.tokens}")
        scores.append(score)

    agg = AggregatedScores.from_runs(scores)
    print(f"  → {label}: F1={agg.f1_mean:.3f}±{agg.f1_std:.3f} "
          f"tok={agg.tokens_mean:.0f} t={agg.time_mean:.1f}s "
          f"iter={round(statistics.mean(n_iters),1)} "
          f"coach={round(statistics.mean(n_coachs),1)}")
    report.add(label, agg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_llm):
    gt = set(json.loads(GT_PATH.read_text(encoding="utf-8"))) \
         if GT_PATH.exists() else set()

    pattern = TriplePattern("?composer", "style", "Baroque")
    tracker = TrackingLLM(base_llm)

    print(f"\n{'='*60}")
    print(f"  Motivational vs Motivational+Coach")
    print(f"  Pattern : {pattern}")
    print(f"  GT size : {len(gt)}")
    print(f"{'='*60}")

    report = Report(
        query_id="motiv_vs_coach",
        template="motiv_coach_study",
        predicate=pattern.p,
        n_expected=len(gt),
    )

    run("Motivational",
        lambda: motivational_scan(pattern, tracker),
        gt, tracker, N_RUNS, report)

    run("Motivational+Coach",
        lambda: motivational_coach_scan(pattern, tracker),
        gt, tracker, N_RUNS, report)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_json(RESULTS_DIR / "motiv_vs_coach.json")
    report.save_csv(RESULTS_DIR  / "motiv_vs_coach.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    main(base_llm)