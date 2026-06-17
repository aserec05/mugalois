# src/mugalois/paths/join.py
"""
Join path evaluation.

Two independent scans + path_join in memory:
  T1 = LLMScan(s, p1, ?b)   — no seeds from right
  T2 = LLMScan(?b, p2, t)   — no seeds from left
  result = path_join(T1, T2) — intersect on ?b

lookahead variant: each scan hints the other side.
"""
from __future__ import annotations
import copy
from mugalois.core.types import TriplePattern, Environment
from mugalois.core.operators import path_join
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


def scan_join(
    s:         str,
    p1:        str,
    p2:        str,
    t:         str,
    llm:       BaseLLM,
    gamma:     Environment = None,
    lookahead: bool = False,
    max_iter:  int = 5,
) -> set[str]:
    """
    Independent scans + join in memory on ?b.

    T1 = LLMScan(s, p1, ?b)  — left side
    T2 = LLMScan(?b, p2, t)  — right side
    result = path_join(T1, T2)

    lookahead=True: T1 hints (?b, p2, t), T2 hints (s, p1, ?b)
    """
    gamma = gamma or Environment()

    la_t1 = f"(?b, {p2}, {t})" if lookahead else ""
    la_t2 = f"({s}, {p1}, ?b)" if lookahead else ""

    # Left scan — (s, p1, ?b) independent
    T1 = LLMScan(
        pattern=TriplePattern(s, p1, "?b"),
        env=Environment(),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=max_iter,
        lookahead=la_t1,
    )

    # Right scan — (?b, p2, t) independent
    T2 = LLMScan(
        pattern=TriplePattern("?b", p2, t),
        env=Environment(),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=max_iter,
        lookahead=la_t2,
    )

    return path_join(T1, T2)