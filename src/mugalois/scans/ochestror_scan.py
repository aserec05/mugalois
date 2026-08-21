# src/mugalois/scans/orchestrator_scan.py
"""
LLMScan — Orchestrator with ClosurePipeline support.

mode:
  standard  : pipeline=[]
  post_only : all conditions → post_filter
  key_only  : force KeyScan
  custom    : pass pipeline= explicitly

lookahead : optional string hint passed to scan prompts (T3 path context)
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
    tau_condition: float = 0.65,
    mode:          str   = "standard",
    pipeline=None,
    lookahead:     str   = "",
    motivational:  bool  = False,
    encoding:      str   = "", 
) -> set[Triple]:
    """
    Orchestrator.

    mode="standard"   → no closures
    mode="post_only"  → all conditions in post_filter
    mode="key_only"   → force KeyScan
    pipeline          → ClosurePipeline instance (overrides mode for scan)
    lookahead         → optional hint about next hop (T3 path context)
    encoding          → optional encoding for scan prompts
     
    """
    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    # singleton seeds → bound terms
    if len(seeds_s) == 1:
        pattern = TriplePattern(list(seeds_s)[0], pattern.p, pattern.o)
        seeds_s = set()
    if len(seeds_o) == 1:
        pattern = TriplePattern(pattern.s, pattern.p, list(seeds_o)[0])
        seeds_o = set()

    # ── 1. strategy ───────────────────────────────────────────────────────────
    if mode == "key_only":
        strategy     = "key"
        seed_example = _pick_seed_example(seeds_s, seeds_o)
    else:
        strategy, seed_example = _choose_strategy(
            pattern, env, seeds_s, seeds_o, llm, tau_strategie
        )

    # ── 2. condition split ────────────────────────────────────────────────────
    if mode == "post_only":
        inject_conds      = []
        post_filter_conds = []
        pattern_vars = {t for t in (pattern.s, pattern.o)
                        if isinstance(t, str) and t.startswith("?")} 
        for var in pattern_vars:
            post_filter_conds.extend(gamma.get(var))
    else:
        inject_conds, post_filter_conds = split_conditions(
            pattern, env, gamma, llm,
            tau_c=tau_condition,
            seed_example=seed_example,
        )
    

    # ── 3. run ────────────────────────────────────────────────────────────────
    if strategy == "seed":
        return LLMSeedScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
            lookahead=lookahead,
            motivational=motivational,
            #encoding=encoding
        )

    elif strategy == "key":
        return LLMKeyScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            lookahead=lookahead,   
            motivational=motivational
        )

    else:  # triple
        return _run_triple(
            pattern, env, llm,
            inject_conds, post_filter_conds,
            max_iter, pipeline, lookahead, motivational,
        )


# ── Strategy selection ────────────────────────────────────────────────────────

SEED_FORCE_THRESHOLD = 20

def _choose_strategy(pattern, env, seeds_s, seeds_o, llm, tau):
    if not seeds_s and not seeds_o:
        return "triple", None
    if len(seeds_s) + len(seeds_o) > SEED_FORCE_THRESHOLD:
        return "seed", None
    prompt = genConfidencePrompt(pattern, env)
    resp = llm.chat(build_messages(prompt))
    try:
        c = float(resp.text.strip())
    except ValueError:
        c = 0.0
    if c > tau:
        return "seed", None
    return "key", _pick_seed_example(seeds_s, seeds_o)


def _pick_seed_example(seeds_s, seeds_o):
    if not seeds_s and not seeds_o:
        return None
    if not seeds_s:
        return next(iter(seeds_o))
    if not seeds_o:
        return next(iter(seeds_s))
    return next(iter(seeds_s if len(seeds_s) <= len(seeds_o) else seeds_o))


# ── Triple / Value scan with optional pipeline ────────────────────────────────

def _run_triple(pattern, env, llm, inject_conds, post_filter_conds,
                max_iter, pipeline, lookahead="", motivational=False, encoding=""):
    s_var = pattern.s_is_var()
    o_var = pattern.o_is_var()

    if pipeline is not None:
        T, _, _ = pipeline.run(pattern, llm)
        from mugalois.core.condition_filter import post_filter as _pf
        T = _pf(T, post_filter_conds, pattern)
        return T

    if s_var and not o_var:
        values = LLMValueScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
            lookahead=lookahead,
            motivational=motivational,
            **({"encoding": encoding} if encoding else {}),
            
        )
        return {Triple(v, pattern.p, pattern.o) for v in values}

    elif not s_var and o_var:
        values = LLMValueScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
            lookahead=lookahead,
            motivational=motivational,
            **({"encoding": encoding} if encoding else {}),
        )
        return {Triple(pattern.s, pattern.p, v) for v in values}

    else:
        return LLMTripleScan(
            pattern, env, llm,
            inject_conds=inject_conds,
            post_filter_conds=post_filter_conds,
            max_iter=max_iter,
            lookahead=lookahead,
        )