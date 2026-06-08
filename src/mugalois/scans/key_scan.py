# src/mugalois/scans/key_scan.py
"""
KeyScan — Algorithm 4.

Receives (inject_conds, post_filter_conds) already computed by the orchestrator.
Never calls LLMConfCond — condition split happened once upstream.
"""

from __future__ import annotations

from mugalois.core.types import Triple, TriplePattern, Environment, AnyCondition
from mugalois.core.helpers import updateEnv, seedsOf
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import genKeyCrankPrompt, genCheckPrompt, build_messages
from mugalois.core.condition_filter import post_filter
from mugalois.llm.llm_client import BaseLLM


def LLMKeyScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
) -> set[Triple]:
    """
    KeyScan with pre-split conditions.

    inject_conds      → injected into each per-key prompt.
    post_filter_conds → applied on the full union after all keys are processed.
    """
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    if not seeds_s and not seeds_o:
        from mugalois.scans.triple_scan import LLMTripleScan
        return LLMTripleScan(pattern, env, llm,
                             inject_conds=inject_conds,
                             post_filter_conds=post_filter_conds)

    # ── pick direction ───────────────────────────────────────────────────────
    if not seeds_s:
        direction, iter_seeds, other, other_seeds = "R->L", seeds_o, pattern.s, seeds_s
    elif not seeds_o:
        direction, iter_seeds, other, other_seeds = "L->R", seeds_s, pattern.o, seeds_o
    elif len(seeds_s) <= len(seeds_o):
        direction, iter_seeds, other, other_seeds = "L->R", seeds_s, pattern.o, seeds_o
    else:
        direction, iter_seeds, other, other_seeds = "R->L", seeds_o, pattern.s, seeds_s

    T = set()

    # ── iterate key by key ───────────────────────────────────────────────────
    for k in iter_seeds:
        if not pattern.is_variable(other) or len(other_seeds) == 1:
            # one-to-one check
            k_prime = other if not pattern.is_variable(other) else list(other_seeds)[0]
            triple  = (
                Triple(k, pattern.p, k_prime) if direction == "L->R"
                else Triple(k_prime, pattern.p, k)
            )
            prompt   = genCheckPrompt((triple.s, triple.p, triple.o),
                                      conditions=inject_conds)
            response = llm.chat(build_messages(prompt))
            if response.text.strip().lower().startswith("yes"):
                T.add(triple)
        else:
            # one-to-many crank
            prompt   = genKeyCrankPrompt(pattern, env, k, direction,
                                         conditions=inject_conds)
            response = llm.chat(build_messages(prompt))
            T = T | json_to_triples(response.text)

    T = post_filter(T, post_filter_conds, pattern)
    updateEnv(env, T, pattern.s, pattern.o)
    return T