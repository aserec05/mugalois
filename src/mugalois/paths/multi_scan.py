# src/mugalois/paths/multi_scan.py
"""
LLMMultiHopScan    — single LLM call + calm motivational follow-up.
LLMStructureDetect — "chain" or "dag" (single focused call).
LLMChainConf       — confidence float [0,1] (single focused call).
LLMEstimateSize    — cardinality estimate int (single focused call).
"""
from __future__ import annotations
import re
from mugalois.paths.path_query import PathQuery
from mugalois.core.parser import json_to_values
from mugalois.core.prompts import build_value_messages, build_messages
from mugalois.llm.llm_client import BaseLLM

CALM_FOLLOW_UP_PROMPT = (
    "If, and only if, you are highly confident there are a few more "
    "you missed, add a small number now. Do not guess, do not "
    "extrapolate a pattern. If you are not sure, return an empty list — "
    "that is the expected, normal answer."
)


def _gen_multi_hop_scan_prompt(path: PathQuery) -> str:
    var    = path.target_var
    N      = path.n_hops
    labels = getattr(path, "condition_labels", {})

    cond_str = ""
    if path.conditions:
        parts = []
        for c in path.conditions:
            v, op, val = c.var, c.op, c.val
            label   = labels.get(v, "value")
            op_word = {">": "greater than", "<": "less than",
                       ">=": "at least",   "<=": "at most",
                       "=": "equal to",    "!=": "different from"}.get(op, op)
            try:
                val_str = f"{int(val):,}"
            except ValueError:
                val_str = val
            parts.append(f"the {label} of {v} is {op_word} {val_str}")
        cond_str = ", and where " + " and ".join(parts)

    multi_hint = (
        "Be exhaustive — there are likely several results, not just one. "
        "List every single value of"
    )

    if N <= 3:
        lines = [f"  - {h.s} {h.p} {h.o}" for h in path.hops]
        body  = "\n".join(lines)
        return (
            f"Task: Find all values of {var} such that all of the "
            f"following hold{cond_str}:\n{body}\n\n"
            f"{multi_hint} {var} you know.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )
    else:
        steps = []
        for h in path.hops:
            if not h.o_is_var():
                steps.append(f"{h.s} {h.p} {h.o}")
            else:
                steps.append(f"{h.s} {h.p} {h.o} (find all such {h.o})")
        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        return (
            f"Task: Starting from {path.hops[0].s}, follow this "
            f"{N}-step chain to find all values of {var}{cond_str}:\n"
            f"{steps_str}\n\n"
            f"{multi_hint} {var} that satisfies all steps. "
            f"Include only values you are factually confident about.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )


def LLMMultiHopScan(path: PathQuery, llm: BaseLLM) -> set[str]:
    """Single LLM call + calm motivational follow-up."""
    prompt = _gen_multi_hop_scan_prompt(path)
    resp   = llm.chat(build_value_messages(prompt))
    values = json_to_values(resp.text)

    already   = ", ".join(sorted(values))
    follow_up = (
        f"{prompt}\n\n"
        f"Already found so far: {already}\n\n"
        f"{CALM_FOLLOW_UP_PROMPT}\n"
        f"List ONLY new values not already in the list above.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    new_vals = json_to_values(llm.chat(build_value_messages(follow_up)).text) - values
    return values | new_vals


def LLMStructureDetect(path: PathQuery, llm: BaseLLM) -> str:
    """
    Single focused call: linear chain or branching DAG?
    Returns "chain" or "dag".
    """
    chain  = " → ".join(f"[{h.p}]" for h in path.hops)
    prompt = (
        f"For the relation: {path.source} {chain} {path.target}\n\n"
        f"Does this typically form:\n"
        f"- a LINEAR CHAIN: each node has at most one successor, or\n"
        f"- a BRANCHING DAG: one node can have multiple successors?\n\n"
        f"Answer with a single word: 'chain' or 'dag'. No explanation."
    )
    resp = llm.chat(build_messages(prompt))
    text = resp.text.strip().lower()
    return "dag" if "dag" in text else "chain"


def LLMChainConf(path: PathQuery, llm: BaseLLM) -> float:
    """
    Single focused call: confidence [0,1] to answer holistically.
    """
    chain     = " → ".join(f"[{h.p}]" for h in path.hops)
    cond_note = " (with filter conditions)" if path.conditions else ""
    prompt    = (
        f"You need to find all {path.target_var} from "
        f"{path.source} via: {chain}, ending at {path.target}{cond_note}.\n\n"
        f"How confident are you (0 to 1) that you can list ALL results "
        f"in a single answer?\n"
        f"Answer with a single float only. No explanation."
    )
    resp = llm.chat(build_messages(prompt))
    try:
        match = re.search(r"0?\.\d+|1\.0|^[01]$", resp.text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.0
    except Exception:
        return 0.0


def LLMEstimateSize(path: PathQuery, llm: BaseLLM) -> int:
    """
    Single focused call: estimated number of results.
    Called only when cardinality decides between Fixpoint and NL precise.
    """
    chain  = " → ".join(f"[{h.p}]" for h in path.hops)
    prompt = (
        f"How many distinct values of {path.target_var} satisfy: "
        f"{path.source} {chain} {path.target}?\n\n"
        f"Single integer only. No explanation."
    )
    resp  = llm.chat(build_messages(prompt))
    match = re.search(r"\d+", resp.text.strip())
    try:
        return max(0, int(match.group())) if match else 0
    except ValueError:
        return 0