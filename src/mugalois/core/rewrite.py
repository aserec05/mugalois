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
from mugalois.core.operators import _apply


def RW1(
    conditions: frozenset[AnyCondition],
    env:        Environment,
    gamma:      Environment,
) -> tuple[Environment, Environment]:
    """
    Pre-processes FILTER conditions before scan.

    For each condition c on variable ?x:

    1. Equality / IN
       - no seeds  → env.set(?x, {val})
       - seeds     → env.set(?x, existing ∩ {val})

    2. Numeric / inequality with known seeds
       → env.set(?x, filter(existing, c))

    3. Otherwise (no seeds, not equality)
       → gamma.add(?x, c)   — LLM handles in prompt

    Returns (env, gamma).
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
                filtered = {v for v in existing if _apply(v, c.op, c.val)}
                env.set(c.var, filtered)

            else:
                gamma.add(c.var, c)

    return env, gamma