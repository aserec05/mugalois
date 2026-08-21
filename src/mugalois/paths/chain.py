# src/mugalois/paths/chain.py
"""
scan_chain — N-hop sequential path evaluation.

For T4, when target_var is not the last node of the chain, we use a
two-phase approach:
  Phase 1: scan the first hop to get candidate values of target_var
  Phase 2: ask the LLM to filter those candidates against the remaining
           path constraints (a "filter scan" — single LLM call)

This avoids the semantic problem of reversing directional predicates
(e.g. reversing "is located in" produces "Italy is located in ?"
which is nonsense for asymmetric relations).
"""
from __future__ import annotations
import copy
from mugalois.paths.path_query import PathQuery
from mugalois.core.types import TriplePattern, Environment
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM


def _scan_chain_forward(
    path:      PathQuery,
    llm:       BaseLLM,
    gamma:     Environment,
    pipeline=None,
    max_iter:  int  = 5,
    verbose:   bool = False,
) -> set[str]:
    """
    Pure left→right sequential chain.
    Correct only when target_var is the object of the LAST hop.
    """
    hops = path.hops
    N    = len(hops)
    current_seeds: set[str] = {hops[0].s} if not hops[0].s_is_var() else set()

    for i, hop in enumerate(hops):
        if verbose:
            print(f"  [chain→] hop {i+1}/{N}: {hop} | seeds={len(current_seeds)}")

        env = Environment()
        if hop.s_is_var() and current_seeds:
            env.set(hop.s, current_seeds)

        triples = LLMScan(
            pattern=hop,
            env=copy.deepcopy(env),
            gamma=copy.deepcopy(gamma),
            llm=llm,
            max_iter=max_iter,
            pipeline=pipeline if i == 0 else None,
        )

        if not triples:
            if verbose:
                print(f"  [chain→] hop {i+1} empty — stopping.")
            return set()

        if hop.o_is_var():
            current_seeds = {t.o for t in triples}
        else:
            current_seeds = {t.s for t in triples if t.o == hop.o}

    return current_seeds


def _gen_filter_prompt(candidates: set[str], remaining: PathQuery) -> str:
    """
    Ask the LLM: from this candidate list, which ones satisfy the
    remaining path constraints?

    Example for q3 remaining path (?b1 is set in ?b2, ?b2 is located in Italy):
      "From the following list:
       Hamlet, Romeo and Juliet, Othello, Macbeth, ...
       Which ones satisfy all of these:
         - [item] is set in a location ?b2
         - ?b2 is located in Italy
       Return ONLY items from the list above."
    """
    cand_str = "\n".join(f"  - {c}" for c in sorted(candidates))

    # Build constraint description from remaining hops
    constraints = []
    for h in remaining.hops:
        if h.o_is_var():
            constraints.append(f"{h.s} {h.p} some value {h.o}")
        else:
            constraints.append(f"{h.s} {h.p} {h.o}")
    constr_str = "\n".join(f"  - {c}" for c in constraints)

    target = remaining.hops[0].s  # the variable being filtered

    return (
        f"From the following list of candidates for {target}:\n"
        f"{cand_str}\n\n"
        f"Which ones satisfy ALL of the following conditions:\n"
        f"{constr_str}\n\n"
        f"Return ONLY items from the list above that satisfy all conditions. "
        f"Do not add any item not in the list.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )


def scan_chain(
    path:      PathQuery,
    llm:       BaseLLM,
    direction: str = "left",
    gamma:     Environment = None,
    pipeline=None,
    max_iter:  int  = 5,
    verbose:   bool = False,
) -> set[str]:
    """
    N-hop chain evaluation — returns values of path.target_var.

    When target_var is not the last node (hops exist after it):
      Phase 1: scan hop[0] → get candidates for target_var
      Phase 2: filter candidates via LLM against remaining constraints
      → avoids reversing directional predicates (semantic correctness)

    When target_var is the last node:
      Standard forward chain (as in T5/T3).
    """
    gamma = gamma or Environment()

    if direction == "right":
        return scan_chain(
            path=path.reversed(), llm=llm, direction="left",
            gamma=gamma, pipeline=pipeline,
            max_iter=max_iter, verbose=verbose,
        )

    target_var = path.target_var

    # Find the hop index where target_var appears as object
    target_hop_idx = next(
        (i for i, h in enumerate(path.hops) if h.o == target_var),
        None
    )

    # Standard case: target_var is last or not found as any hop's object
    if target_hop_idx is None or target_hop_idx == len(path.hops) - 1:
        return _scan_chain_forward(path, llm, gamma, pipeline, max_iter, verbose)

    # target_var has hops AFTER it → two-phase approach
    if verbose:
        print(f"  [chain] target_var={target_var!r} at hop {target_hop_idx}, "
              f"hops after it = {len(path.hops) - target_hop_idx - 1} "
              f"→ two-phase filter scan")

    # Phase 1: get candidates for target_var (forward up to target_hop_idx)
    left_path = path.left_half(target_var)
    candidates = _scan_chain_forward(
        left_path, llm, gamma, pipeline, max_iter, verbose
    )

    if not candidates:
        if verbose:
            print(f"  [chain] no candidates found — stopping.")
        return set()

    if verbose:
        print(f"  [chain] {len(candidates)} candidates for {target_var}: "
              f"{sorted(candidates)[:5]}{'...' if len(candidates) > 5 else ''}")

    # Phase 2: filter candidates against remaining path constraints
    remaining = path.right_half(target_var)
    prompt    = _gen_filter_prompt(candidates, remaining)
    resp      = llm.chat(build_value_messages(prompt))
    filtered  = json_to_values(resp.text)

    # Safety: only return values that were in the original candidate set
    return filtered & candidates


def scan_chain_from_bound(
    path:     PathQuery,
    var:      str,
    seeds:    set[str],
    llm:      BaseLLM,
    gamma:    Environment = None,
    max_iter: int  = 5,
    verbose:  bool = False,
) -> set[str]:
    """Evaluate suffix of path from an already-bound variable."""
    gamma = gamma or Environment()
    suffix = path.suffix_from(var)
    new_gamma = Environment()
    new_gamma._bindings.update(gamma._bindings)
    new_gamma.set(var, seeds)
    return scan_chain(
        path=suffix, llm=llm, direction="left",
        gamma=new_gamma, max_iter=max_iter, verbose=verbose,
    )