# src/mugalois/core/rewrite.py
"""
Query rewriting rules for µ-Galois.

RW1 : pre-processes FILTER conditions before scan.
"""

from __future__ import annotations

from mugalois.core.types import (
    AnyCondition, Condition, ConditionIN,
    Environment,
)


def RW1(
    conditions: frozenset[AnyCondition],
    env:        Environment,
    gamma:      Environment,
) -> tuple[Environment, Environment]:
    """
    Pre-processes FILTER conditions before scan.

    For each condition c on variable ?x:

    1. Equality / IN (= or IN)
       - no seeds yet  → env.set(?x, {val})
       - seeds exist   → env.set(?x, existing ∩ {val})   # intersect

    2. Numeric / inequality (>, <, >=, <=, !=) with seeds known
       → env.set(?x, filter(existing, c))                 # filter seeds directly

    3. Otherwise (no seeds, not equality)
       → gamma.add(?x, c)                                 # LLM handles in prompt

    Returns updated (env, gamma).
    """
    for c in conditions:

        if isinstance(c, ConditionIN):
            existing = env.get(c.var)
            if existing:
                env.set(c.var, existing & set(c.values))
            else:
                env.set(c.var, set(c.values))

        elif isinstance(c, Condition):
            existing = env.get(c.var)

            if c.is_equality():
                if existing:
                    env.set(c.var, existing & {c.val})
                else:
                    env.set(c.var, {c.val})

            elif existing:
                # numeric or != with known seeds → filter directly
                filtered = {v for v in existing if _apply(v, c.op, c.val)}
                env.set(c.var, filtered)

            else:
                # no seeds, not equality → LLM handles
                gamma.add(c.var, c)

    return env, gamma


def _apply(value: str, op: str, threshold: str) -> bool:
    """Apply a single comparison between value and threshold."""
    try:
        v = float(value.replace(",", ""))
        t = float(threshold.replace(",", ""))
        if op == ">":  return v > t
        if op == "<":  return v < t
        if op == ">=": return v >= t
        if op == "<=": return v <= t
        if op == "!=": return v != t
    except ValueError:
        if op == "!=": return value != threshold
        if op == ">":  return value > threshold
        if op == "<":  return value < threshold
    return False