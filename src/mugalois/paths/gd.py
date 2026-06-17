# src/mugalois/paths/gd.py
"""
Left-to-right path evaluation.

GD: LLMScan(s, p1, ?b) → env(?b) → LLMScan(?b, p2, t)
"""
from __future__ import annotations
import copy
from mugalois.core.types import TriplePattern, Environment
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


def scan_gd(
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
    Left-to-right: scan(s, p1, ?b) → seeds → scan(?b, p2, t)

    lookahead: if True, hints the next hop in the first scan prompt.
    Returns set of ?b values.
    """
    gamma = gamma or Environment()

    # Hop 1 — scan (s, p1, ?b)
    pat1  = TriplePattern(s, p1, "?b")
    env1  = Environment()
    la    = f"(?b, {p2}, {t})" if lookahead else ""

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
    seeds_b = {t.o for t in triples1}  # values of ?b

    if not seeds_b:
        return set()

    # Hop 2 — scan (?b, p2, t) with seeds from hop 1
    pat2 = TriplePattern("?b", p2, t)
    env2 = Environment()
    env2.set("?b", seeds_b)

    triples2 = LLMScan(
        pattern=pat2,
        env=copy.deepcopy(env2),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=max_iter,
    )
    return {t.s for t in triples2}