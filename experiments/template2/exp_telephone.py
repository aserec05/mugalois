"""
experiments/template2/exp_telephone.py

Telephone arabe experiment — chained LLM coaches help the scanner
find rare entities when it gets stuck.

Flow:
  iter 0   : standard prompt
  iter 1+  : motivational prompt
  on empty : telephone arabe starts
             Coach receives: already found + "help your friend"
             Mode A (names)     : coach gives specific names
             Mode B (directions): coach gives a search direction
             Scanner uses hint → continues
             Stop after N_BREAK consecutive coached empties

Modes:
  standard         : break on first empty (baseline)
  motivational     : motivational prompt, break on first empty
  telephone_names  : telephone with name hints
  telephone_dirs   : telephone with direction hints

Usage
-----
    python3 -m experiments.template2.exp_telephone
    python3 -m experiments.template2.exp_telephone --exp baroque
    python3 -m experiments.template2.exp_telephone --exp oscar
    python3 -m experiments.template2.exp_telephone --dry-run
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

N_RUNS   = 5
MAX_ITER = 12
N_BREAK  = 2   # stop after N_BREAK consecutive coached empties

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)


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


# ── Coach calls ───────────────────────────────────────────────────────────────

def coach_names(
    predicate:     str,
    obj:           str,
    already_found: set,
    tracker:       TrackingLLM,
) -> str:
    """
    Coach gives specific names to help the scanner.
    'Ton ami cherche X. Il a trouvé Y. Aide-le avec des noms.'
    """
    found_str = ", ".join(sorted(str(x) for x in already_found)[:20])
    prompt = (
        f"Your friend is looking for entities with {predicate} \"{obj}\".\n"
        f"They already found: {found_str}.\n"
        f"They are stuck and need help finding more rare examples.\n"
        f"Help your friend by giving 3-5 specific names they haven't found yet. "
        f"Only give names, separated by commas. No explanation."
    )
    messages = [
        {"role": "system", "content":
         "You are a helpful friend. Give only a comma-separated list of names."},
        {"role": "user", "content": prompt},
    ]
    return tracker.chat(messages).text.strip()


def coach_directions(
    predicate:     str,
    obj:           str,
    already_found: set,
    tracker:       TrackingLLM,
) -> str:
    """
    Coach gives a search direction to help the scanner.
    'Ton ami cherche X. Il a trouvé Y. Donne une direction.'
    """
    found_str = ", ".join(sorted(str(x) for x in already_found)[:20])
    prompt = (
        f"Your friend is looking for entities with {predicate} \"{obj}\".\n"
        f"They already found: {found_str}.\n"
        f"They are stuck and need a new direction to explore.\n"
        f"Help your friend in ONE sentence starting with: "
        f"\"You should explore...\"\n"
        f"Give only a direction, no specific names."
    )
    messages = [
        {"role": "system", "content":
         "You are a helpful friend. Respond with exactly one sentence, "
         "giving only a direction without specific names."},
        {"role": "user", "content": prompt},
    ]
    return tracker.chat(messages).text.strip()


# ── Core scan ─────────────────────────────────────────────────────────────────

def telephone_scan(
    pattern:   TriplePattern,
    tracker:   TrackingLLM,
    mode:      str,           # "names" or "directions"
    n_break:   int = N_BREAK,
    max_iter:  int = MAX_ITER,
) -> tuple[set, int, int, list[str]]:
    """
    Telephone arabe scan.
    Returns (triples, n_iter, n_coach_calls, hints_used).
    """
    T             = set()
    ctx           = []
    sys_msg       = {"role": "system", "content": SYSTEM_PROMPT}
    n_iter        = 0
    coached_empty = 0
    n_coach       = 0
    hints         = []

    for i in range(max_iter):
        n_iter = i + 1

        if i == 0:
            prompt = genTripleScanPrompt(pattern, encoding="pattern")
        elif coached_empty > 0:
            # telephone arabe — coach intervenes
            if mode == "names":
                hint = coach_names(pattern.p, pattern.o, T, tracker)
            else:
                hint = coach_directions(pattern.p, pattern.o, T, tracker)
            n_coach += 1
            hints.append(hint)

            already = ", ".join(sorted(str(t) for t in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"A friend suggests: {hint}\n\n"
                f"Find more triples based on this suggestion. "
                f"Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )
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

        if T_new.issubset(T):
            coached_empty += 1
            if coached_empty >= n_break:
                break
        else:
            coached_empty = 0
            ctx.append({"role": "user",      "content": prompt})
            ctx.append({"role": "assistant", "content": resp.text})
            T |= T_new

    return T, n_iter, n_coach, hints


def standard_scan(pattern, tracker, max_iter=MAX_ITER):
    T   = set()
    ctx = []
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    from mugalois.core.prompts import genTripleScanIterativePrompt
    for i in range(max_iter):
        prompt = (
            genTripleScanPrompt(pattern, encoding="pattern")
            if i == 0 else genTripleScanIterativePrompt(T)
        )
        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        resp     = tracker.chat(messages)
        T_new    = json_to_triples(resp.text)
        if T_new.issubset(T) and i > 0:
            break
        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        T |= T_new
    return T, i + 1, 0, []


def motivational_scan(pattern, tracker, max_iter=MAX_ITER):
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
    return T, i + 1, 0, []


# ── Runner ────────────────────────────────────────────────────────────────────

def run_mode(label, scan_fn, gt, tracker, n_runs, report, show_hints=True):
    scores    = []
    n_iters   = []
    n_coachs  = []
    all_hints = []

    for i in range(n_runs):
        tracker.reset()
        T, n_iter, n_coach, hints = scan_fn()
        values = {t.s for t in T}
        score  = Metrics.compute(values, gt, mode="values")
        score.time_s = tracker.total_time_s
        score.tokens = tracker.total_tokens
        n_iters.append(n_iter)
        n_coachs.append(n_coach)
        all_hints.extend(hints)

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

    if show_hints and all_hints:
        print(f"  Sample hints ({label}):")
        for h in all_hints[:3]:
            print(f"    • {h}")

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

    print(f"\n{'='*60}")
    print(f"  Telephone Arabe Experiment — {exp}")
    print(f"  Pattern : {pattern}")
    print(f"  GT size : {len(gt)}")
    print(f"  N_BREAK : {N_BREAK}")
    print(f"{'='*60}")

    report = Report(
        query_id=f"telephone_{exp}",
        template="telephone_study",
        predicate=pattern.p,
        n_expected=len(gt),
    )

    run_mode("Standard",
             lambda: standard_scan(pattern, tracker),
             gt, tracker, N_RUNS, report, show_hints=False)

    run_mode("Motivational",
             lambda: motivational_scan(pattern, tracker),
             gt, tracker, N_RUNS, report, show_hints=False)

    run_mode("Telephone_names",
             lambda: telephone_scan(pattern, tracker, mode="names"),
             gt, tracker, N_RUNS, report)

    run_mode("Telephone_dirs",
             lambda: telephone_scan(pattern, tracker, mode="directions"),
             gt, tracker, N_RUNS, report)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_json(RESULTS_DIR / f"telephone_{exp}.json")
    report.save_csv(RESULTS_DIR  / f"telephone_{exp}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     default="baroque",
                        choices=["baroque", "oscar"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    main(base_llm, args.exp)
