# src/mugalois/scans/triple_scan.py
"""
TripleScan (Algorithm 2) and ValueScan.

Receives (inject_conds, post_filter_conds) already computed by the orchestrator.
Never calls LLMConfCond — condition split happened once upstream.

verbose=True prints full trace: prompt, raw response, parsed result, post_filter.
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

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)


def _vprint(verbose: bool, *args):
    if verbose:
        print(*args)


def LLMTripleScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
    max_iter:          int  = 5,
    encoding:          str  = "pattern",
    motivational:      bool = False,
    verbose:           bool = False,
    lookahead:         str  = "",
) -> set[Triple]:
    """
    TripleScan — Algorithm 2.

    inject_conds      → injected into prompt at iter 0.
    post_filter_conds → applied programmatically after all iterations.
    motivational      → motivational prompt at iter > 0.
    verbose           → print full trace (prompt, response, parsed, filter).
    """
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    T   = set()
    ctx = []
    sys_msg = build_messages("")[0]

    _vprint(verbose, f"\n{'─'*50}")
    _vprint(verbose, f"LLMTripleScan | pattern={pattern}")
    _vprint(verbose, f"  inject_conds     : {inject_conds}")
    _vprint(verbose, f"  post_filter_conds: {post_filter_conds}")
    _vprint(verbose, f"  motivational     : {motivational}")
    _vprint(verbose, f"{'─'*50}")

    for i in range(max_iter):
        if i == 0:
            prompt = genTripleScanPrompt(pattern, encoding,
                                         conditions=inject_conds)
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
            prompt = genTripleScanIterativePrompt(T, pattern=pattern)

        _vprint(verbose, f"\n=== ITER {i} ===")
        _vprint(verbose, f"[PROMPT]\n{prompt}")

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]

        response = llm.chat(messages)

        _vprint(verbose, f"\n[RESPONSE RAW]\n{response.text}")

        T_new = json_to_triples(response.text)
        _vprint(verbose, f"\n[PARSED] {len(T_new)} triples:")
        for t in sorted(T_new):
            _vprint(verbose, f"  {t}")

        if T_new.issubset(T):
            _vprint(verbose, f"\n→ Empty / subset — breaking at iter {i}")
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        T = T | T_new

    _vprint(verbose, f"\n[BEFORE POST_FILTER] {len(T)} triples")
    T_filtered = post_filter(T, post_filter_conds, pattern)
    if verbose and post_filter_conds:
        kept    = T_filtered
        removed = T - T_filtered
        _vprint(verbose, f"[POST_FILTER] kept={len(kept)} removed={len(removed)}")
        if removed:
            _vprint(verbose, f"  Removed: {sorted(str(t) for t in removed)[:5]}")

    updateEnv(env, T_filtered, pattern.s, pattern.o)
    return T_filtered


def LLMValueScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
    max_iter:          int  = 5,
    encoding:          str  = "pattern",
    motivational:      bool = False,
    verbose:           bool = False,
    lookahead:         str  = "",
) -> set[str]:
    """
    ValueScan — returns plain string values instead of full triples.

    inject_conds      → injected into prompt at iter 0.
    post_filter_conds → applied on values after iteration.
    motivational      → motivational prompt at iter > 0.
    verbose           → print full trace.
    """
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    V   = set()
    ctx = []
    sys_msg = build_value_messages("")[0]

    _vprint(verbose, f"\n{'─'*50}")
    _vprint(verbose, f"LLMValueScan | pattern={pattern}")
    _vprint(verbose, f"  inject_conds     : {inject_conds}")
    _vprint(verbose, f"  post_filter_conds: {post_filter_conds}")
    _vprint(verbose, f"{'─'*50}")

    for i in range(max_iter):
        if i == 0:
            prompt = genValueScanPrompt(pattern, encoding,
                                        conditions=inject_conds)
        elif motivational:
            already = ", ".join(sorted(V))
            prompt = (
                f"Already retrieved:\n{already}\n\n"
                f"{MOTIVATIONAL_PROMPT}\n"
                f"List more values if any remain. "
                f"If truly none, return an empty list."
            )
        else:
            prompt = genValueScanIterativePrompt(V, pattern=pattern)

        _vprint(verbose, f"\n=== ITER {i} ===")
        _vprint(verbose, f"[PROMPT]\n{prompt}")

        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        response = llm.chat(messages)

        _vprint(verbose, f"\n[RESPONSE RAW]\n{response.text}")

        V_new = json_to_values(response.text)
        _vprint(verbose, f"\n[PARSED] {len(V_new)} values: {sorted(V_new)[:10]}")

        if V_new.issubset(V):
            _vprint(verbose, f"\n→ Empty / subset — breaking at iter {i}")
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        V = V | V_new

    _vprint(verbose, f"\n[BEFORE POST_FILTER] {len(V)} values")
    V_filtered = _post_filter_values(V, post_filter_conds)
    if verbose and post_filter_conds:
        removed = V - V_filtered
        _vprint(verbose, f"[POST_FILTER] kept={len(V_filtered)} removed={len(removed)}")
        if removed:
            _vprint(verbose, f"  Removed: {sorted(removed)[:5]}")

    return V_filtered


def _post_filter_values(values, conditions):
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
                    ok = False; break
            elif isinstance(cond, Condition):
                if not _apply(v, cond.op, cond.val):
                    ok = False; break
        if ok:
            result.add(v)
    return result