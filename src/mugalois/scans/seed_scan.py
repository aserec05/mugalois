# src/mugalois/scans/seed_scan.py
from __future__ import annotations

from mugalois.core.types import Triple, TriplePattern, Environment, AnyCondition
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import (
    genSeedCrankPrompt, genIterativePrompt, build_messages, SYSTEM_PROMPT
)
from mugalois.core.condition_filter import post_filter
from mugalois.llm.llm_client import BaseLLM

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)


def LLMSeedScan(
    pattern:           TriplePattern,
    env:               Environment,
    llm:               BaseLLM,
    inject_conds:      list[AnyCondition] = None,
    post_filter_conds: list[AnyCondition] = None,
    max_iter:          int = 5,
    motivational:      bool = False,
) -> set[Triple]:
    inject_conds      = inject_conds      or []
    post_filter_conds = post_filter_conds or []

    T   = set()
    ctx = []

    for i in range(max_iter):
        if i == 0:
            prompt = genSeedCrankPrompt(pattern, env, conditions=inject_conds)
        elif motivational:
            already = ", ".join(sorted(str(t) for t in T))
            prompt = (
                f"Context: Already retrieved:\n{already}\n\n"
                f"{MOTIVATIONAL_PROMPT}\n"
                f"List more triples if any remain. "
                f"If truly none, return an empty list."
            )
        else:
            prompt = genIterativePrompt(T)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *ctx,
            {"role": "user", "content": prompt},
        ]
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