"""
experiments/template2/exp_motivational.py

Dedicated experiment to study the motivational prompt effect.

Exp A — Oscar-winning films (High High Cardinality ~900)
  Pattern : ?film award ?y  FILTER(?y = "Academy Award")
  Test    : Standard vs Motivational — how far can LLM go?

Exp B — Baroque composers (Niche ~50)
  Pattern : ?composer era ?y  FILTER(?y = "Baroque")
  Test    : Standard vs Motivational — can LLM surface obscure composers?

Metrics: F1, Precision, Recall, Tokens, Time, N_iter (mean over N runs)

Usage
-----
    python3 -m experiments.template2.exp_motivational
    python3 -m experiments.template2.exp_motivational --exp A
    python3 -m experiments.template2.exp_motivational --exp B
    python3 -m experiments.template2.exp_motivational --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from mugalois.core.types import (
    TriplePattern, Triple, Condition, ConditionIN, Environment,
)
from mugalois.core.prompts import (
    genTripleScanPrompt, genTripleScanIterativePrompt,
    build_messages, SYSTEM_PROMPT,
)
from mugalois.core.condition_filter import post_filter
from mugalois.core.parser import json_to_triples
from mugalois.llm.llm_client import AzureOpenAIClient, MockLLM, BaseLLM, LLMResponse
from evaluation.metrics import Metrics, MetricScores, AggregatedScores
from evaluation.report import Report

RESULTS_DIR = Path(__file__).resolve().parent / "results"

N_RUNS   = 5
MAX_ITER = 10   # more iterations for high cardinality

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)

# ── Ground truths ─────────────────────────────────────────────────────────────

GT_B = {
    "Alessandro Grandi", "Antoine Francisque", "António Marques Lésbio",
    "Arcangelo Corelli", "Barbara Strozzi", "Charles Mouton",
    "Chiara Margarita Cozzolani", "Claudio Monteverdi", "Ennemond Gaultier",
    "Ernst Gottlieb Baron", "Francesco Geminiani", "Francesco Lambardi",
    "Francesco Maria Zuccari", "François Couperin", "Francesco Maria Zuccari",
    "Giovanni Martino Cesare", "Girolamo Frescobaldi", "Giulio Caccini",
    "Giuseppe Tartini", "Jacob van Eyck", "Jacques Champion de Chambonnières",
    "Jan Dismas Zelenka", "Johan Agrell", "Johann Georg Weichenberger",
    "Johann Hermann Schein", "Johann Joachim Quantz", "Johann Michael Nicolai",
    "Johann Sebastian Bach", "José de Vaquedano", "Juan Mathías",
    "Louis Marchand", "Luís Álvares Pinto", "Michel Richard Delalande",
    "Nikolay Diletsky", "Pierre-César Abeille", "René Mesangeau",
    "Richard Leveridge", "Robert de Visée", "Thomas Baltzar",
    "Thomas Roseingrave",
}

# ── TrackingLLM ───────────────────────────────────────────────────────────────

class TrackingLLM(BaseLLM):
    def __init__(self, llm):
        self._llm         = llm
        self.total_tokens = 0
        self.total_time_s = 0.0
        self.n_calls      = 0

    def reset(self):
        self.total_tokens = 0
        self.total_time_s = 0.0
        self.n_calls      = 0

    def chat(self, messages):
        resp = self._llm.chat(messages)
        self.total_tokens += resp.usage_tokens
        self.total_time_s += resp.latency_s
        self.n_calls      += 1
        return resp


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan(
    pattern:           TriplePattern,
    post_filter_conds: list,
    tracker:           TrackingLLM,
    motivational:      bool = False,
    max_iter:          int  = MAX_ITER,
) -> tuple[set[Triple], int]:
    """
    Run TripleScan with or without motivational prompt.
    Returns (triples_after_filter, n_iterations).
    """
    T   = set()
    ctx = []
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    n_iter  = 0

    for i in range(max_iter):
        n_iter = i + 1
        if i == 0:
            prompt = genTripleScanPrompt(pattern, encoding="pattern")
        elif motivational:
            already = ", ".join(sorted(str(t) for t in T))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"{MOTIVATIONAL_PROMPT}\n"
                f"Find more triples. Do not repeat already retrieved values. "
                f"If there are truly no more, return an empty list.\n"
                f"Respond ONLY in valid JSON following the schema provided."
            )
        else:
            prompt = genTripleScanIterativePrompt(T)

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        resp     = tracker.chat(messages)
        T_new    = json_to_triples(resp.text)

        if T_new.issubset(T) and (not motivational or i > 1):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": resp.text})
        T |= T_new

    T_filtered = post_filter(T, post_filter_conds, pattern)
    return T_filtered, n_iter


# ── Run one experiment ────────────────────────────────────────────────────────

def run_experiment(
    name:              str,
    pattern:           TriplePattern,
    post_filter_conds: list,
    gt:                set[str],
    subject_var:       bool,       # True = extract s, False = extract o
    tracker:           TrackingLLM,
    n_runs:            int = N_RUNS,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Pattern : {pattern}")
    print(f"  GT size : {len(gt)}")
    print(f"{'='*60}")

    report = Report(
        query_id=name,
        template="motivational_study",
        predicate=pattern.p,
        n_expected=len(gt),
    )

    for label, motiv in [("Standard", False), ("Motivational", True)]:
        scores  = []
        n_iters = []

        for i in range(n_runs):
            tracker.reset()
            T, n_iter = scan(pattern, post_filter_conds, tracker,
                             motivational=motiv)
            values = {t.s for t in T} if subject_var else {t.o for t in T}
            score  = Metrics.compute(values, gt, mode="values")
            score.time_s = tracker.total_time_s
            score.tokens = tracker.total_tokens
            n_iters.append(n_iter)

            print(f"    {label} run {i+1}/{n_runs} → "
                  f"returned={len(values)} expected={len(gt)} | "
                  f"P={score.precision:.3f} R={score.recall:.3f} "
                  f"F1={score.f1:.3f} "
                  f"iter={n_iter} t={score.time_s:.1f}s tok={score.tokens}")
            scores.append(score)

        agg = AggregatedScores.from_runs(scores)
        import statistics
        avg_iter = round(statistics.mean(n_iters), 1)
        std_iter = round(statistics.stdev(n_iters) if n_runs > 1 else 0.0, 1)
        print(f"  → {label}: F1={agg.f1_mean:.3f}±{agg.f1_std:.3f} "
              f"tok={agg.tokens_mean:.0f} t={agg.time_mean:.1f}s "
              f"iter={avg_iter}±{std_iter}")
        report.add(label, agg)

    report.print_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_json(RESULTS_DIR / f"motivational_{name}.json")
    report.save_csv(RESULTS_DIR  / f"motivational_{name}.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_llm, exps: list[str]):
    tracker = TrackingLLM(base_llm)

    if "A" in exps:
        # Load Oscar GT from file
        gt_path = Path(__file__).resolve().parent / "gt_oscar_films.json"
        if gt_path.exists():
            gt_a = set(json.loads(gt_path.read_text(encoding="utf-8")))
        else:
            print(f"  Warning: {gt_path} not found — using empty GT")
            gt_a = set()

        # Object fixed — no post_filter needed
        pattern_a = TriplePattern("?film", "award", "Academy Award")

        run_experiment(
            name="ExpA_Oscar_Films",
            pattern=pattern_a,
            post_filter_conds=[],
            gt=gt_a,
            subject_var=True,
            tracker=tracker,
        )

    if "B" in exps:
        gt_b_path = Path(__file__).resolve().parent / "gt_baroque_composers.json"
        if gt_b_path.exists():
            gt_b = set(json.loads(gt_b_path.read_text(encoding="utf-8")))
        else:
            gt_b = GT_B
        # Pattern: ?composer style Baroque — object is fixed, no post_filter needed
        pattern_b = TriplePattern("?composer", "style", "Baroque")

        run_experiment(
            name="ExpB_Baroque_Composers",
            pattern=pattern_b,
            post_filter_conds=[],
            gt=gt_b,
            subject_var=True,
            tracker=tracker,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     default="AB", help="A, B, or AB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_llm = MockLLM() if args.dry_run else AzureOpenAIClient()
    exps     = list(args.exp.upper())
    main(base_llm, exps)