# src/mugalois/rec/simple_rec.py
"""
LLMRecScan       : single LLM call attempting the full transitive closure.
LLMAtomicRecConf : confidence the LLM can resolve the closure at once.

Mirrors mugalois/paths/simple.py (T3's LLMSimpleScan / LLMSimpleConf), but
for recursive property paths (RecursivePattern, p+/p*) instead of two-hop
joins.
"""
from __future__ import annotations
from mugalois.core.types import RecursivePattern
from mugalois.core.prompts import (
    genRecScanPrompt,
    genAtomicRecConfPrompt,
    build_value_messages,
    build_messages,
)
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import BaseLLM


def LLMRecScan(pattern: RecursivePattern, llm: BaseLLM) -> set[str]:
    """
    Single-call attempt at the full transitive closure described by pattern.

    pattern.is_plus() : seed = pattern.s, resolves (seed, p+, ?o)
    pattern.is_star()  : seed = pattern.o, resolves (?s, p*, seed)
    """
    mode = "plus" if pattern.is_plus() else "star"
    s = pattern.s if not pattern.s_is_var() else None
    t = pattern.o if not pattern.o_is_var() else None
    prompt = genRecScanPrompt(s, pattern.p, t, mode=mode)
    resp   = llm.chat(build_value_messages(prompt))
    return json_to_values(resp.text)


def LLMAtomicRecConf(pattern: RecursivePattern, llm: BaseLLM) -> float:
    """Confidence the LLM can solve the full recursive closure in one call."""
    mode = "plus" if pattern.is_plus() else "star"
    s = pattern.s if not pattern.s_is_var() else None
    t = pattern.o if not pattern.o_is_var() else None
    prompt = genAtomicRecConfPrompt(s, pattern.p, t, mode=mode)
    resp   = llm.chat(build_messages(prompt))
    try:
        text = resp.text.strip()
        if text.startswith("{"):
            import json as _j
            text = str(_j.loads(text).get("confidence", 0.0))
        return max(0.0, min(1.0, float(text)))
    except Exception:
        return 0.0