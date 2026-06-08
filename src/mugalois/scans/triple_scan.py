# src/mugalois/core/triple_scan.py

from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.core.helpers import updateEnv
from mugalois.core.parser import json_to_triples, json_to_values
from mugalois.core.prompts import (
    genTripleScanPrompt, genTripleScanIterativePrompt,
    genValueScanPrompt, genValueScanIterativePrompt,
    build_messages, build_value_messages,
)
from mugalois.llm.llm_client import BaseLLM


def LLMTripleScan(
    pattern:  TriplePattern,
    env:      Environment,
    llm:      BaseLLM,
    max_iter: int = 5,
    encoding: str = "pattern",
) -> set[Triple]:
    """
    TripleScan — Algorithm 2 (ex TableScan).

    Returns full triples (s, p, o).
    The experiment projects the variable side after the call.

    encoding : "pattern" | "constrained" | "sparql"
    """
    T   = set()
    ctx = []
    sys_msg = build_messages("")[0]

    for i in range(max_iter):
        prompt = (
            genTripleScanPrompt(pattern, encoding)
            if i == 0
            else genTripleScanIterativePrompt(T)
        )
        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        response = llm.chat(messages)
        T_new    = json_to_triples(response.text)

        if T_new.issubset(T):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        T = T | T_new

    updateEnv(env, T, pattern.s, pattern.o)
    return T


def LLMValueScan(
    pattern:  TriplePattern,
    env:      Environment,
    llm:      BaseLLM,
    max_iter: int = 5,
    encoding: str = "pattern",
) -> set[str]:
    """
    ValueScan — returns plain string values instead of full triples.

    Same encodings as TripleScan but uses VALUE_SCHEMA system prompt
    and returns values directly without triple structure.

    encoding : "pattern" | "constrained" | "sparql"
    """
    V   = set()
    ctx = []
    sys_msg = build_value_messages("")[0]

    for i in range(max_iter):
        prompt = (
            genValueScanPrompt(pattern, encoding)
            if i == 0
            else genValueScanIterativePrompt(V)
        )
        messages = [sys_msg, *ctx, {"role": "user", "content": prompt}]
        response = llm.chat(messages)
        V_new    = json_to_values(response.text)

        if V_new.issubset(V):
            break

        ctx.append({"role": "user",      "content": prompt})
        ctx.append({"role": "assistant", "content": response.text})
        V = V | V_new

    return V
