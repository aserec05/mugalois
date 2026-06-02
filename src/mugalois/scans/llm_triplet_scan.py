from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import seedsOf
from mugalois.core.prompts import genConfidencePrompt, build_messages
from mugalois.llm.llm_client import BaseLLM
from mugalois.scans.key_crank import LLMKeyCrank
from mugalois.scans.seed_crank import LLMSeedCrank
from mugalois.scans.table_scan import LLMTableScan


def LLMTripletScan(
    pattern: TriplePattern,
    env: Environment,
    llm: BaseLLM,
    max_iter: int = 5,
    tau_strategie: float = 0.7
) -> set[Triple]:
    """
    Orchestror of the algorithm. Return a new table with the triples
    which hold the triple query.
    """
    seeds_s = seedsOf(pattern.s, env)
    seeds_o = seedsOf(pattern.o, env)

    if not seeds_s and not seeds_o:
        return LLMTableScan(pattern,env,llm,max_iter)
    
    prompt = genConfidencePrompt(pattern, env)
    response = llm.chat(build_messages(prompt))

    try:
        c = float(response.text.strip())
    except ValueError: # the LLM has hallucinate, and did give a float
        c = 0.0

    if c > tau_strategie: # the LLM is confident
        return LLMSeedCrank(pattern, env, llm, max_iter)
    else:
        return LLMKeyCrank(pattern, env, llm)