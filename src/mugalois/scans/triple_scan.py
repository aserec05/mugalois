# src/mugalois/scans/triple_scan.py
"""
TripleScan (Algorithm 2) and ValueScan.

Receives (inject_conds, post_filter_conds) already computed by the orchestrator.
Never calls LLMConfCond — condition split happened once upstream.
"""

from __future__ import annotations

from mugalois.core.types import Triple, TriplePattern, Environment, AnyCondition
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples, json_to_values
from mugalois.core.prompts import (
    genTripleScanPrompt, genTripleScanIterativePrompt,
    genValueScanPrompt, genValueScanIterativePrompt,
    build_messages, build_value_messages,
)
from mugalois.core.condition_filter import post_filter
from mugalois.llm.llm_client import BaseLLM


def LLMTripleScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
    max_iter:          int = 5,
    encoding:          str = "pattern",
) -> set[Triple]:
    """
    TripleScan — Algorithm 2.

    Returns full triples (s, p, o).
    inject_conds      → passed to prompt generation.
    post_filter_conds → applied programmatically after iteration.
    encoding : "pattern" | "constrained" | "sparql"
    """
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    T   = set()
    ctx = []
    sys_msg = build_messages("")[0]

    for i in range(max_iter):
        prompt = (
            genTripleScanPrompt(pattern, encoding, conditions=inject_conds)
            if i == 0
            else genTripleScanIterativePrompt(T)
        )
        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        response = llm.chat(messages)
        T_new    = json_to_triples(response.text)

        if T_new.issubset(T):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        T = T | T_new

    T = post_filter(T, post_filter_conds, pattern)
    updateEnv(env, T, pattern.s, pattern.o)
    return T


def LLMValueScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
    max_iter:          int = 5,
    encoding:          str = "pattern",
) -> set[str]:
    """
    ValueScan — returns plain string values instead of full triples.

    inject_conds      → passed to prompt generation.
    post_filter_conds → applied on values after iteration.
    encoding : "pattern" | "constrained" | "sparql"
    """
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    V   = set()
    ctx = []
    sys_msg = build_value_messages("")[0]

    for i in range(max_iter):
        prompt = (
            genValueScanPrompt(pattern, encoding, conditions=inject_conds)
            if i == 0
            else genValueScanIterativePrompt(V)
        )
        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        response = llm.chat(messages)
        V_new    = json_to_values(response.text)

        if V_new.issubset(V):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        V = V | V_new

    V = _post_filter_values(V, post_filter_conds)
    return V


def _post_filter_values(
    values:     set[str],
    conditions: list[AnyCondition],
) -> set[str]:
    """Post-filter plain values against conditions."""
    from mugalois.core.operators import _apply
    from mugalois.core.types import Condition, ConditionIN

    if not conditions:
        return values

    result = set()
    for v in values:
        ok = True
        for cond in conditions:
            if isinstance(cond, ConditionIN):
                if v not in cond.values:
                    ok = False
                    break
            elif isinstance(cond, Condition):
                if not _apply(v, cond.op, cond.val):
                    ok = False
                    break
        if ok:
            result.add(v)
    return result