# src/mugalois/paths/routing.py
"""
Routing logic for MuGaloisMultiPath.

Decision tree (in order of priority):

  1. BOUND VARIABLES in gamma
     If any intermediate variable is already bound in gamma, we skip
     the hops that produce it and start directly from its known seeds.
     This is a hard constraint, not an optimization.

  2. NO BOUND VARIABLES — cardinality-based direction + split decision
     a. Estimate fanout of first hop from source (left side)
     b. Estimate fanout of last hop reversed from target (right side)
     c. If one side is small (≤ WIDE_THRESHOLD): chain in that direction
     d. If both are large: estimate all hops, split at minimum fanout
     e. Fallback: chain left (zero extra LLM calls)

We do NOT ask the LLM for routing decisions after ChainConf fails —
only for integer estimates (cardinality), which are more reliable
and interpretable than confidence scores.
"""
from __future__ import annotations
import re
from typing import Optional
from mugalois.paths.path_query import PathQuery
from mugalois.core.types import TriplePattern, Environment
from mugalois.core.prompts import build_messages
from mugalois.llm.llm_client import BaseLLM

WIDE_THRESHOLD = 5   # fanout > this = "wide" intermediate set


# ── Cardinality estimation ────────────────────────────────────────────────────

def LLMEstimateHopSize(hop: TriplePattern, llm: BaseLLM) -> int:
    """
    Estimate the number of values the free variable in `hop` can take.

    (Barack Obama, was born in, ?b1) → expect ~1
    (?b1, nationality, ?b2)          → expect large (many countries)

    Returns 0 on parse failure (triggers fallback to chain left).
    """
    if hop.s_is_var():
        question = (
            f"How many distinct values of {hop.s} satisfy "
            f"({hop.s}, {hop.p}, {hop.o}) factually? "
            f"Single integer only."
        )
    else:
        question = (
            f"How many distinct values of {hop.o} satisfy "
            f"({hop.s}, {hop.p}, {hop.o}) factually? "
            f"Single integer only."
        )
    resp  = llm.chat(build_messages(question))
    match = re.search(r"\d+", resp.text.strip())
    return max(0, int(match.group())) if match else 0


# ── Bound variable detection ──────────────────────────────────────────────────

def find_bound_var(path: PathQuery, gamma: Environment) -> Optional[str]:
    """
    Return the first intermediate variable that is already bound in gamma.
    Returns None if none are bound.
    """
    for var in path.intermediate_vars:
        if gamma.has(var):
            return var
    return None


# ── Main routing function ─────────────────────────────────────────────────────

def decide_routing(
    path:    PathQuery,
    llm:     BaseLLM,
    gamma:   Environment,
    verbose: bool = False,
) -> dict:
    """
    Decide how to evaluate path without a single LLM call.

    Returns a dict:
      {
        "strategy":   "bound" | "chain" | "split",
        "direction":  "left" | "right",   # for chain
        "split_var":  str | None,          # for split
        "bound_var":  str | None,          # for bound
      }
    """
    # ── Priority 1: bound variables in gamma ─────────────────────────
    bound_var = find_bound_var(path, gamma)
    if bound_var is not None:
        if verbose:
            print(f"  [routing] bound variable found: {bound_var} "
                  f"({len(gamma.get(bound_var))} seeds) → start from there")
        return {
            "strategy":  "bound",
            "direction": "left",
            "split_var": None,
            "bound_var": bound_var,
        }

    # ── Priority 2: cardinality-based direction ───────────────────────
    size_left  = LLMEstimateHopSize(path.hops[0], llm)
    last       = path.hops[-1]
    size_right = LLMEstimateHopSize(TriplePattern(last.o, last.p, last.s), llm)

    if verbose:
        print(f"  [routing] left_fanout={size_left}  right_fanout={size_right}")

    # If estimation failed (0) → fallback chain left
    if size_left == 0 and size_right == 0:
        if verbose:
            print(f"  [routing] estimation failed → fallback chain(left)")
        return {"strategy": "chain", "direction": "left",
                "split_var": None, "bound_var": None}

    # One side is small → pure chain in that direction
    if size_left <= WIDE_THRESHOLD or size_right <= WIDE_THRESHOLD:
        direction = "left" if size_left <= size_right else "right"
        if verbose:
            print(f"  [routing] → chain({direction})")
        return {"strategy": "chain", "direction": direction,
                "split_var": None, "bound_var": None}

    # Both sides are wide → find best split point
    inter_vars = path.intermediate_vars
    if not inter_vars:
        # No intermediate variables (1-hop path) — just chain
        return {"strategy": "chain", "direction": "left",
                "split_var": None, "bound_var": None}

    if len(inter_vars) == 1:
        # T3 case: only one choice
        if verbose:
            print(f"  [routing] → split({inter_vars[0]}) [single var]")
        return {"strategy": "split", "direction": "left",
                "split_var": inter_vars[0], "bound_var": None}

    # N>2 intermediate vars: estimate all hops and split at minimum
    hop_sizes = [LLMEstimateHopSize(h, llm) for h in path.hops]
    if verbose:
        print(f"  [routing] hop_sizes={hop_sizes}")

    # Cost of splitting at inter_vars[k] (object of hop k):
    # min of incoming hop (k) and outgoing hop (k+1) sizes
    best_var  = inter_vars[0]
    best_cost = float("inf")
    for k, var in enumerate(inter_vars):
        cost = min(
            hop_sizes[k]     if k < len(hop_sizes)     else float("inf"),
            hop_sizes[k + 1] if k + 1 < len(hop_sizes) else float("inf"),
        )
        if cost < best_cost:
            best_cost = cost
            best_var  = var

    if verbose:
        print(f"  [routing] → split({best_var}) cost={best_cost}")
    return {"strategy": "split", "direction": "left",
            "split_var": best_var, "bound_var": None}
