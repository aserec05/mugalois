# src/mugalois/rec/fixpoint.py
"""
LLMFixpointScan — hop-by-hop transitive closure via LLMScan.

Root cause found and fixed: predicates phrased as a passive/abstract
relation ("was directly succeeded by") triggered sequence-dump
confusion on dense historical clusters, confirmed across 6+ phrasings.
"came immediately before" — combined with encoding="minimal" (a plain
sentence instead of bracket/triple notation) — was verified correct
in isolation on every case that previously failed. The fix is the
predicate wording itself (set by the caller in queries/patterns), used
here with encoding="minimal" to match the exact phrasing verified.

Architecture unchanged: still goes through LLMScan, env, and the full
strategy-selection machinery (KeyScan/SeedScan/TripleScan).
"""
from __future__ import annotations
import copy
import re
from mugalois.core.types import TriplePattern, Environment, RecursivePattern
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


NO_SPECULATION_HINT = (
    "Do not guess or speculate. Only return values you are factually "
    "confident about. If nothing is confirmed, return an empty list."
)

JUNK_MARKERS = {
    "null", "none", "n/a", "na", "unknown", "tbd", "tba",
    "to be determined", "to be announced", "pending", "",
}
_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)

 
_LEADING_NUMBER = re.compile(r"^\s*\d+[\.\)]\s*")
MAX_PLAUSIBLE_VALUE_WORDS = 6  # generic length heuristic, not domain-specific
 
 
def _clean_value(v: str) -> str:
    """Strip a leading list-numbering prefix like '10. ' or '10) '."""
    return _LEADING_NUMBER.sub("", v).strip()
 
 
def _is_garbage(value: str) -> bool:
    v = _clean_value(value)
    if v.lower() in JUNK_MARKERS:
        return True
    if not _HAS_ALNUM.search(v):
        return True
    # A refusal/explanation sentence is much longer than a real entity
    # name — generic length cutoff, not a keyword/domain match.
    if len(v.split()) > MAX_PLAUSIBLE_VALUE_WORDS:
        return True
    return False
 
 
def _strip_junk(values: set[str]) -> set[str]:
    return {_clean_value(v) for v in values if not _is_garbage(v)}



def LLMFixpointScan(
    pattern:   RecursivePattern,
    llm:       BaseLLM,
    env:       Environment = None,
    gamma:     Environment = None,
    max_depth: int = 300,
    verbose:   bool = False,
) -> set[str]:
    gamma = gamma or Environment()
    env   = env   or Environment()
    gamma_is_empty = not gamma._bindings

    seed      = pattern.seed()
    direction = pattern.direction()

    visited  = {seed} if pattern.is_star() else set()
    frontier = {seed}
    result   = set(visited)
    seen_frontiers = set()

    for depth in range(max_depth):
        if not frontier:
            break

        frontier_key = frozenset(frontier)
        if frontier_key in seen_frontiers:
            if verbose:
                print(f"  [LLMFixpointScan] depth={depth} cycle, stopping")
            break
        seen_frontiers.add(frontier_key)

        hop_env   = Environment()
        hop_gamma = gamma if gamma_is_empty else copy.deepcopy(gamma)

        if direction == "forward":
            hop_env.set("?x", set(frontier))
            extract = lambda triples: {tr.o for tr in triples}
        else:
            hop_env.set("?y", set(frontier))
            extract = lambda triples: {tr.s for tr in triples}

        scan_pattern = TriplePattern("?x", pattern.p, "?y")
        vp = LLMScan(
            scan_pattern, hop_env, gamma=hop_gamma, llm=llm,
            lookahead=NO_SPECULATION_HINT,
            encoding="minimal",
        )
        candidates = _strip_junk(extract(vp))
        new_values = candidates - visited

        if verbose:
            print(f"  [LLMFixpointScan] depth={depth} "
                  f"frontier={sorted(frontier)} candidates={sorted(candidates)} "
                  f"new={sorted(new_values)}")

        if not candidates:
            break

        visited |= new_values
        result  |= new_values
        frontier = new_values

        if not frontier:
            break

    return result