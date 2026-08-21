# src/mugalois/rec/simple_rec.py
from __future__ import annotations
import re
from mugalois.core.types import RecursivePattern
from mugalois.core.prompts import (
    genRecScanPrompt,
    genAtomicRecConfPrompt,
    build_value_messages,
    build_messages,
)
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import BaseLLM


CALM_FOLLOW_UP_PROMPT = (
    "If, and only if, you are highly confident there are a few more "
    "you missed, add a small number now. Do not guess, do not "
    "extrapolate a pattern. If you are not sure, return an empty list — "
    "that is the expected, normal answer."
)

PROPORTIONAL_CAP_FACTOR = 1.0
MIN_CAP = 10
COVERAGE_THRESHOLD = 0.8


def _prompts(pattern: RecursivePattern):
    s = pattern.s if not pattern.s_is_var() else None
    t = pattern.o if not pattern.o_is_var() else None
    return (s if s is not None else pattern.s,
            t if t is not None else pattern.o)


def genSizeEstimatePrompt(s: str, p: str, t: str, mode: str) -> str:
    if mode == "plus":
        task = f"how many entities are reachable from {s} via '{p}' (one or more steps)"
    else:
        task = f"how many entities can reach {t} via '{p}' (zero or more steps)"
    return (
        f"Question: {task}?\n"
        f"Give your best estimate as a single integer. "
        f"Answer with the integer only, no explanation."
    )


def LLMEstimateRecSize(pattern: RecursivePattern, llm: BaseLLM) -> int:
    mode = "plus" if pattern.is_plus() else "star"
    s_prompt, t_prompt = _prompts(pattern)
    prompt = genSizeEstimatePrompt(s_prompt, pattern.p, t_prompt, mode)
    resp   = llm.chat(build_messages(prompt))
    match  = re.search(r"-?\d+", resp.text.strip())
    if not match:
        return 0
    try:
        return max(0, int(match.group()))
    except ValueError:
        return 0


def _should_skip_followup(values: set, est_size: int) -> bool:
    """True when the follow-up should be skipped.

    Skip when:
    - est_size == 0  : size unknown, do not risk hallucination
    - coverage >= COVERAGE_THRESHOLD : first call already covered enough
    """
    if est_size == 0:
        return True
    return len(values) >= COVERAGE_THRESHOLD * est_size


def LLMRecScan(
    pattern:      RecursivePattern,
    llm:          BaseLLM,
    motivational: bool = True,
    capped:       bool = True,
) -> set[str]:
    mode = "plus" if pattern.is_plus() else "star"
    s_prompt, t_prompt = _prompts(pattern)

    prompt = genRecScanPrompt(s_prompt, pattern.p, t_prompt, mode=mode)
    resp   = llm.chat(build_value_messages(prompt))
    values = json_to_values(resp.text)

    if not motivational:
        return values

    est_size = LLMEstimateRecSize(pattern, llm)

    if _should_skip_followup(values, est_size):
        return values

    already = ", ".join(sorted(values))
    follow_up = (
        f"{genRecScanPrompt(s_prompt, pattern.p, t_prompt, mode=mode)}\n\n"
        f"Already retrieved so far:\n{already}\n\n"
        f"{CALM_FOLLOW_UP_PROMPT}\n"
        f"List ONLY new values not already in the list above.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    messages = [
        build_value_messages("")[0],
        {"role": "user", "content": follow_up},
    ]
    resp_follow_up = llm.chat(messages)
    new_values = json_to_values(resp_follow_up.text) - values

    if capped:
        cap = max(MIN_CAP, int(len(values) * PROPORTIONAL_CAP_FACTOR))
        if len(new_values) > cap:
            return values

    return values | new_values


def LLMAtomicRecConf(pattern: RecursivePattern, llm: BaseLLM) -> float:
    mode = "plus" if pattern.is_plus() else "star"
    s_prompt, t_prompt = _prompts(pattern)
    prompt = genAtomicRecConfPrompt(s_prompt, pattern.p, t_prompt, mode=mode)
    resp   = llm.chat(build_messages(prompt))
    try:
        text = resp.text.strip()
        if text.startswith("{"):
            import json as _j
            text = str(_j.loads(text).get("confidence", 0.0))
        return max(0.0, min(1.0, float(text)))
    except Exception:
        return 0.0