from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv, seedsOf
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import genKeyCrankPrompt, genCheckPrompt, build_messages, SYSTEM_PROMPT
from mugalois.llm.llm_client import BaseLLM


def LLMKeyCrank(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
) -> set[Triple]:
    """Algorithm 4 — KeyCrank.

    The LLM is not confident to list all seeds at once.
    We iterate seed by seed over the smaller side.
    """
    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    if not seeds_s and not seeds_o:
        from mugalois.scans.table_scan import LLMTableScan
        return LLMTableScan(pattern, env, llm)

    if not seeds_s:
        direction   = "R->L"
        iter_seeds  = seeds_o
        other       = pattern.s
        other_seeds = seeds_s
    elif not seeds_o:
        direction   = "L->R"
        iter_seeds  = seeds_s
        other       = pattern.o
        other_seeds = seeds_o
    elif len(seeds_s) <= len(seeds_o):
        direction   = "L->R"
        iter_seeds  = seeds_s
        other       = pattern.o
        other_seeds = seeds_o
    else:
        direction   = "R->L"
        iter_seeds  = seeds_o
        other       = pattern.s
        other_seeds = seeds_s

    T = set()

    for k in iter_seeds:
        if not pattern.is_variable(other) or len(other_seeds) == 1:
            k_prime = other if not pattern.is_variable(other) else list(other_seeds)[0]
            triple  = Triple(k, pattern.p, k_prime) if direction == "L->R" \
                      else Triple(k_prime, pattern.p, k)
            prompt   = genCheckPrompt((triple.s, triple.p, triple.o))
            response = llm.chat(build_messages(prompt))
            if response.text.strip().lower().startswith("yes"):
                T.add(triple)
        else:
            prompt   = genKeyCrankPrompt(pattern, env, k, direction)
            response = llm.chat(build_messages(prompt))
            T = T | json_to_triples(response.text)

    updateEnv(env, T, pattern.s, pattern.o)
    return T