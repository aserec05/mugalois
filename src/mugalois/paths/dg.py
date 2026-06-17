# src/mugalois/paths/dg.py
"""
Right-to-left path evaluation.

DG: LLMScan(?b, p2, t) → env(?b) → LLMScan(s, p1, ?b)
"""
from __future__ import annotations
import copy
from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


def scan_dg(
    s:            str,
    p1:           str,
    p2:           str,
    t:            str,
    llm:          BaseLLM,
    gamma:        Environment = None,
    lookahead:    bool = False,
    motivational: bool = False,
    pipeline=None,
    max_iter:     int = 5,
) -> set[str]:
    """
    Right-to-left: scan(?b, p2, t) → seeds → scan(s, p1, ?b)

    lookahead: if True, hints the next hop in the first scan prompt.
    Returns set of ?b values.
    """
    gamma = gamma or Environment()

    # Hop 1 — scan (?b, p2, t)
    pat1 = TriplePattern("?b", p2, t)
    env1 = Environment()
    la   = f"({s}, {p1}, ?b)" if lookahead else ""

    triples1 = LLMScan(
        pattern=pat1,
        env=copy.deepcopy(env1),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=max_iter,
        lookahead=la,
        motivational=motivational,
        pipeline=pipeline,
    )
    seeds_b = {t.s for t in triples1}  # values of ?b

    if not seeds_b:
        return set()

    # Hop 2 — scan (s, p1, ?b) with seeds from hop 1
    pat2 = TriplePattern(s, p1, "?b")
    env2 = Environment()
    env2.set("?b", seeds_b)

    triples2 = LLMScan(
        pattern=pat2,
        env=copy.deepcopy(env2),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=max_iter,
    )
    return {t.o for t in triples2}