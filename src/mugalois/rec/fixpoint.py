# src/mugalois/rec/fixpoint.py
"""
LLMFixpointScan — hop-by-hop transitive closure via repeated LLMScan calls.

Used as fallback when LLMAtomicRecConf is below tau_A: instead of asking
the LLM for the entire p+/p* closure at once, we iterate one hop at a
time and re-inject newly found nodes as seeds for the next hop, until
no new node appears (fixpoint reached) — mirrors the mu-RA fixpoint
operator: U_{i+1} = U_i ∪ [[phi]][X/U_i].
"""
from __future__ import annotations
import copy
from mugalois.core.types import TriplePattern, Environment, RecursivePattern
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


def LLMFixpointScan(
    pattern:   RecursivePattern,
    llm:       BaseLLM,
    gamma:     Environment = None,
    max_depth: int = 8,
    batch:     bool = True,
    verbose:   bool = False,
) -> set[str]:
    """
    Hop-by-hop transitive closure for `pattern` (s, p+, ?o) or (?s, p*, o).

    direction = pattern.direction():
      'forward'  : seed = pattern.s, scan (u, p, ?y) for u in frontier.
      'backward' : seed = pattern.o, scan (?y, p, u) for u in frontier.

    Stops when no new node is found in an iteration, or at max_depth.
    pattern.is_plus() excludes the seed from the result (irreflexive);
    pattern.is_star() includes it (reflexive).
    """
    gamma = gamma or Environment()
    seed  = pattern.seed()
    direction = pattern.direction()

    frontier = {seed}
    visited  = {seed} if pattern.is_star() else set()

    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        new_nodes = set()

        if batch:
            new_nodes = _scan_batch(frontier, pattern.p, llm, gamma, direction)
        else:
            for node in frontier:
                new_nodes |= _scan_single(node, pattern.p, llm, gamma, direction)

        # Keep only genuinely new nodes (avoid cycles / re-visiting)
        new_nodes -= visited
        if pattern.is_plus():
            new_nodes -= {seed}

        if verbose:
            print(f"  [Fixpoint] depth={depth} frontier={len(frontier)} "
                  f"new={len(new_nodes)}")

        if not new_nodes:
            break

        visited |= new_nodes
        frontier = new_nodes

    return visited


def _scan_single(
    node: str, p: str, llm: BaseLLM, gamma: Environment, direction: str,
) -> set[str]:
    """One-hop scan from/to a single node."""
    if direction == "forward":
        scan_pattern = TriplePattern(node, p, "?y")
    else:
        scan_pattern = TriplePattern("?y", p, node)

    triples = LLMScan(
        pattern=scan_pattern,
        env=Environment(),
        gamma=copy.deepcopy(gamma),
        llm=llm,
        max_iter=3,
    )
    return {tr.o for tr in triples} if direction == "forward" \
        else {tr.s for tr in triples}


def _scan_batch(
    frontier: set, p: str, llm: BaseLLM, gamma: Environment, direction: str,
) -> set[str]:
    """
    One-hop scan for an entire frontier at once, using the batched
    prompt (genRecHopBatchPrompt) to save tokens versus one call per node.
    """
    from mugalois.core.prompts import genRecHopBatchPrompt, build_value_messages
    from mugalois.core.parser import json_to_values

    prompt = genRecHopBatchPrompt(frontier, p, direction=direction)
    resp   = llm.chat(build_value_messages(prompt))
    return json_to_values(resp.text)