from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import genTableScanPrompt, genIterativePrompt, build_messages
from mugalois.llm.llm_client import BaseLLM
from mugalois.core.prompts import SYSTEM_PROMPT


def LLMTableScan(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
    max_iter: int = 5,
    context: str = "",
    encoding="current"  
) -> set[Triple]:
    """Algorithm 2 — TableScan with conversational context."""
    T   = set()
    ctx = []

    for i in range(max_iter):
        prompt = genTableScanPrompt(pattern, context, encoding) if i == 0 else genIterativePrompt(T)

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