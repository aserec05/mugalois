# src/mugalois/core/condition_filter.py
"""
Condition handling for µ-Galois scans.

LLMConfCond      : one LLM call per condition.
split_conditions : called AFTER strategy selection, once per scan.
post_filter      : programmatic filtering on the final triple set.
"""

from __future__ import annotations

from mugalois.core.types import (
    AnyCondition, Condition, ConditionIN,
    Environment, TriplePattern,
)
from mugalois.core.operators import _apply
from mugalois.core.prompts import (
    build_messages,
    genConfidenceConditionTripleScanPrompt,
)
from mugalois.llm.llm_client import BaseLLM


# ── Confidence call ────────────────────────────────────────────────────────────

def LLMConfCond(
    pattern:      TriplePattern,
    env:          Environment,
    condition:    AnyCondition,
    llm:          BaseLLM,
    seed_example: str | None = None,
) -> float:
    """
    Ask the LLM how confident it is in respecting <condition>
    while scanning <pattern>.

    seed_example: when in KeyScan context, the first element of the
    iterated seed — gives the LLM a concrete reference instead of
    reasoning abstractly over the full pattern.

    Returns float in [0, 1]. Falls back to 0.0 on parse error.
    """
    prompt = genConfidenceConditionTripleScanPrompt(
        pattern, env, repr(condition), seed_example=seed_example
    )
    response = llm.chat(build_messages(prompt))
    try:
        text = response.text.strip()
        if text.startswith("{"):
            import json as _json
            data = _json.loads(text)
            text = str(data.get("confidence", data.get("score", 0.0)))
        return max(0.0, min(1.0, float(text)))
    except (ValueError, KeyError, Exception):
        return 0.0


# ── Splitting — called AFTER strategy selection ────────────────────────────────

def split_conditions(
    pattern:      TriplePattern,
    env:          Environment,
    gamma:        Environment,
    llm:          BaseLLM,
    tau_c:        float = 0.85,
    seed_example: str | None = None,
) -> tuple[list[AnyCondition], list[AnyCondition]]:
    """
    Called AFTER strategy selection — seed_example is known at this point.

    For every variable in the pattern that has conditions in gamma:
        LLMConfCond score > tau_c  → inject_conds  (passed to scan prompt)
        LLMConfCond score <= tau_c → post_filter_conds (programmatic after scan)

    seed_example: first element of the iterated seed in KeyScan,
                  None for TripleScan and SeedScan.

    Returns (inject_conds, post_filter_conds).
    """
    inject_conds:      list[AnyCondition] = []
    post_filter_conds: list[AnyCondition] = []

    pattern_vars = {
        t for t in (pattern.s, pattern.o)
        if isinstance(t, str) and t.startswith("?")
    }

    for var in pattern_vars:
        for cond in gamma.get(var):
            score = LLMConfCond(
                pattern, env, cond, llm,
                seed_example=seed_example,
            )
            if score > tau_c:
                inject_conds.append(cond)
            else:
                post_filter_conds.append(cond)

    return inject_conds, post_filter_conds


# ── Post-filtering ─────────────────────────────────────────────────────────────

def post_filter(
    triples:           set,
    post_filter_conds: list[AnyCondition],
    pattern:           TriplePattern,
) -> set:
    """
    Programmatic filtering of a triple set against post_filter_conds.
    A triple passes only if ALL conditions are satisfied.
    """
    if not post_filter_conds:
        return triples

    result = set()
    for triple in triples:
        if _triple_satisfies_all(triple, post_filter_conds, pattern):
            result.add(triple)
    return result


def _triple_satisfies_all(
    triple,
    conditions: list[AnyCondition],
    pattern:    TriplePattern,
) -> bool:
    """Return True iff triple satisfies every condition."""
    binding: dict[str, str] = {}
    if pattern.s_is_var():
        binding[pattern.s] = triple.s
    if pattern.o_is_var():
        binding[pattern.o] = triple.o

    for cond in conditions:
        value = binding.get(cond.var)
        if value is None:
            continue

        if isinstance(cond, ConditionIN):
            if value not in cond.values:
                return False
        elif isinstance(cond, Condition):
            if not _apply(value, cond.op, cond.val):
                return False

    return True