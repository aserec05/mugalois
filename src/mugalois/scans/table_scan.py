from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import genTableScanPrompt, genIterativePrompt, build_messages
from mugalois.llm.llm_client import BaseLLM


def LLMTableScan(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
    max_iter: int = 5
) -> set[Triple]:
    """Algorithm 2 — TableScan.

    No seeds available on either side. Asks the LLM to list all known
    triples for the given predicate.
    """
    T = set()

    for i in range(max_iter):
        prompt = genTableScanPrompt(pattern) if i == 0 else genIterativePrompt(T)
        #print("PROMPT:", prompt)
        messages = build_messages(prompt)
        for m in messages:
            print(f"[{m['role']}]", m['content'])
        response = llm.chat(messages)
        print("RESPONSE:", repr(response.text))
        T_new = json_to_triples(response.text)

        if T_new.issubset(T):
            break

        T = T | T_new

    updateEnv(env, T, pattern.s, pattern.o)
    return T
