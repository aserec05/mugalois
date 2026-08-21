# src/mugalois/paths/split.py
"""
scan_split — binary split at a breakpoint variable, join in memory.

For s -p1-> ?b1 -p2-> ?b2 -p3-> t, split at ?b1:
  left  = scan_chain(s -p1-> ?b1)             → set of ?b1 values from source
  right = scan_chain(reversed(t -p3-> ?b2 -p2-> ?b1)) → set of ?b1 values from target
  valid = left ∩ right                         → ?b1 values consistent with both ends
  then  = scan_chain(?b1=valid -p2-> ?b2 -p3-> t)  → final ?b2 values

Generalises scan_join (T3, always split at the single intermediate var).
"""
from __future__ import annotations
import copy
from mugalois.paths.path_query import PathQuery
from mugalois.paths.chain import scan_chain
from mugalois.core.types import Environment
from mugalois.llm.llm_client import BaseLLM


def scan_split(
    path:      PathQuery,
    split_var: str,
    llm:       BaseLLM,
    gamma:     Environment = None,
    max_iter:  int  = 5,
    verbose:   bool = False,
) -> set[str]:
    """
    Binary split at split_var, join in memory, then continue.
    Returns values of path.target_var.
    """
    gamma = gamma or Environment()

    left_path  = path.left_half(split_var)
    right_path = path.right_half(split_var)

    if verbose:
        print(f"  [split] at {split_var}")
        print(f"    left : {left_path}")
        print(f"    right: {right_path}")

    # Left: values of split_var reachable from source
    left_vals = scan_chain(
        path=left_path, llm=llm, direction="left",
        gamma=copy.deepcopy(gamma), max_iter=max_iter, verbose=verbose,
    )

    # Right: values of split_var that can reach target (evaluate reversed)
    right_vals = scan_chain(
        path=right_path.reversed(), llm=llm, direction="left",
        gamma=copy.deepcopy(gamma), max_iter=max_iter, verbose=verbose,
    )

    valid_split = left_vals & right_vals
    if verbose:
        print(f"  [split] left={len(left_vals)} right={len(right_vals)} "
              f"valid={len(valid_split)}")

    if not valid_split:
        return set()

    # If split_var IS the target_var, we're done
    if split_var == path.target_var:
        return valid_split

    # Continue: use valid_split as seeds for the right half
    env = Environment()
    env.set(split_var, valid_split)
    return scan_chain(
        path=right_path, llm=llm, direction="left",
        gamma=env, max_iter=max_iter, verbose=verbose,
    )
