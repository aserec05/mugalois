# src/mugalois/paths/simple.py
"""
LLMSimpleScan  : single LLM call for the full two-hop path.
LLMSimpleConf  : confidence the LLM can solve the path at once.
LLMDirectionConf: which direction is most efficient.
"""
from __future__ import annotations
from mugalois.core.prompts import (
    genSimpleScanPathPrompt,
    genSimpleConfPathPrompt,
    genDirectionConfPathPrompt,
    build_value_messages,
    build_messages,
)
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import BaseLLM


def LLMSimpleScan(
    s:   str,
    p1:  str,
    p2:  str,
    t:   str,
    llm: BaseLLM,
) -> set[str]:
    """Single LLM call for the full two-hop path."""
    prompt = genSimpleScanPathPrompt(s, p1, p2, t)
    resp   = llm.chat(build_value_messages(prompt))
    return json_to_values(resp.text)


def LLMSimpleConf(
    s:   str,
    p1:  str,
    p2:  str,
    t:   str,
    llm: BaseLLM,
) -> float:
    """Confidence the LLM can solve the full path at once."""
    prompt = genSimpleConfPathPrompt(s, p1, p2, t)
    resp   = llm.chat(build_messages(prompt))
    try:
        text = resp.text.strip()
        if text.startswith("{"):
            import json as _j
            text = str(_j.loads(text).get("confidence", 0.0))
        return max(0.0, min(1.0, float(text)))
    except Exception:
        return 0.0


def LLMDirectionConf(
    s:   str,
    p1:  str,
    p2:  str,
    t:   str,
    llm: BaseLLM,
) -> str:
    """Returns 'left', 'right', or 'join'."""
    prompt = genDirectionConfPathPrompt(s, p1, p2, t)
    messages = [
        {"role": "system", "content": 
         "You may reason briefly, but end your response with exactly one word on the last line: A or B."},
        {"role": "user",   "content": prompt},
    ]
    resp = llm.chat(messages)
    text = resp.text.strip().upper()
    # Extract last word A or B
    word = text.split()[-1]
    if word == "B":
        return "right"
    elif word == "A":
        return "left"
    return "join"  # fallback when uncertain
