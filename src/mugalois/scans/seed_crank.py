from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import (
    genSeedCrankPrompt, genIterativePrompt, build_messages, SYSTEM_PROMPT
)
from mugalois.llm.llm_client import BaseLLM


def LLMSeedCrank(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
    max_iter: int = 5
) -> set[Triple]:
    """Algorithm 3 — SeedCrank.

    At least one seed known on either or both sides.
    Seeds are injected into the prompt to constrain the LLM search space.
    Updates the environment as a side effect.
    """
    T   = set()
    ctx = []

    for i in range(max_iter):
        prompt = genSeedCrankPrompt(pattern, env) if i == 0 else genIterativePrompt(T)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *ctx,
            {"role": "user", "content": prompt},
        ]

        response = llm.chat(messages)
        T_new = json_to_triples(response.text)

        if T_new.issubset(T):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        T = T | T_new

    updateEnv(env, T, pattern.s, pattern.o)
    return T