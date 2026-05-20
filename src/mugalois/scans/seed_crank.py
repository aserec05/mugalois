from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples
from mugalois.core.prompts import genSeedCrankPrompt, genIterativePrompt, build_messages
from mugalois.llm.llm_client import BaseLLM


def LLMSeedCrank(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
    max_iter: int = 5
) -> set[Triple]:
    """Algorithm 2 —   Seed Crank.

    At least one seed available.  s e/o o. We put one entirely in the prompt
    """
    T = set()

    for i in range(max_iter):
        prompt = genSeedCrankPrompt(pattern, env) if i == 0 else genIterativePrompt(T)
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

    print("QUI")
    updateEnv(env, T, pattern.s, pattern.o)
    print("QUA")
    return T
