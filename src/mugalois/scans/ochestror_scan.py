# src/mugalois/scans/orchestrator_scan.py
"""
LLMScan — Orchestrator.

Order:
  1. Determine strategy (TripleScan / SeedScan / KeyScan)
  2. split_conditions with seed_example if KeyScan
  3. Run strategy with (inject_conds, post_filter_conds)
  4. Reconstruct triples if ValueScan was used
"""

from __future__ import annotations

from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import seedsOf
from mugalois.core.prompts import genConfidencePrompt, build_messages
from mugalois.core.condition_filter import split_conditions
from mugalois.llm.llm_client import BaseLLM
from mugalois.scans.key_scan import LLMKeyScan
from mugalois.scans.seed_scan import LLMSeedScan
from mugalois.scans.triple_scan import LLMTripleScan, LLMValueScan


def LLMScan(
    pattern:       TriplePattern,
    env:           Environment,
    gamma:         Environment,
    llm:           BaseLLM,
    max_iter:      int   = 5,
    tau_strategie: float = 0.7,
    tau_condition: float = 0.85,
) -> set[Triple]:
    """
    Orchestrator — selects strategy first, then splits conditions.

    1. Determine strategy
    2. split_conditions (with seed_example if KeyScan)
    3. Run strategy
    4. Reconstruct triples if ValueScan used
    """
    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    # ── singleton seeds → bound terms ────────────────────────────────────────
    if len(seeds_s) == 1:
        pattern = TriplePattern(list(seeds_s)[0], pattern.p, pattern.o)
        seeds_s = set()
    if len(seeds_o) == 1:
        pattern = TriplePattern(pattern.s, pattern.p, list(seeds_o)[0])
        seeds_o = set()

    # ── 1. determine strategy ─────────────────────────────────────────────────
    strategy, seed_example = _choose_strategy(
        pattern, env, seeds_s, seeds_o, llm, tau_strategie
    )

    # ── 2. split conditions with seed_example ─────────────────────────────────
    inject_conds, post_filter_conds = split_conditions(
        pattern, env, gamma, llm,
        tau_c=tau_condition,
        seed_example=seed_example,
    )

    # ── 3 & 4. run strategy ───────────────────────────────────────────────────
    if strategy == "triple":
        return _run_triple(
            pattern, env, llm, inject_conds, post_filter_conds, max_iter
        )

    elif strategy == "seed":
        return LLMSeedScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
        )

    else:  # key
        return LLMKeyScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
        )


# ── Strategy selection ────────────────────────────────────────────────────────

def _choose_strategy(
    pattern:       TriplePattern,
    env:           Environment,
    seeds_s:       set,
    seeds_o:       set,
    llm:           BaseLLM,
    tau_strategie: float,
) -> tuple[str, str | None]:
    """
    Returns (strategy_name, seed_example).

    strategy_name : "triple" | "seed" | "key"
    seed_example  : first element of iterated seed if "key", else None
    """
    # no seeds → TripleScan
    if not seeds_s and not seeds_o:
        return "triple", None

    # confidence check for seed vs key
    prompt = genConfidencePrompt(pattern, env)
    response = llm.chat(build_messages(prompt))
    try:
        c = float(response.text.strip())
    except ValueError:
        c = 0.0

    if c > tau_strategie:
        return "seed", None

    # KeyScan — pick seed_example from the smaller side
    if not seeds_s:
        seed_example = next(iter(seeds_o))
    elif not seeds_o:
        seed_example = next(iter(seeds_s))
    elif len(seeds_s) <= len(seeds_o):
        seed_example = next(iter(seeds_s))
    else:
        seed_example = next(iter(seeds_o))

    return "key", seed_example


# ── TripleScan or ValueScan ───────────────────────────────────────────────────

def _run_triple(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list,
    post_filter_conds: list,
    max_iter:          int,
) -> set[Triple]:
    """
    Use ValueScan when only one side is variable (T1 pattern).
    Reconstruct triples from values.
    Use TripleScan when both sides are variable (T2 pattern).
    """
    s_var = pattern.s_is_var()
    o_var = pattern.o_is_var()

    # T1 — one variable → ValueScan + reconstruct
    if s_var and not o_var:
        values = LLMValueScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
        )
        return {Triple(v, pattern.p, pattern.o) for v in values}

    elif not s_var and o_var:
        values = LLMValueScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
        )
        return {Triple(pattern.s, pattern.p, v) for v in values}

    # T2 — both variables → TripleScan
    else:
        return LLMTripleScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
        )