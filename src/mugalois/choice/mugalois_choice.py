# src/mugalois/choice/mugalois_choice.py
"""
MuGaloisChoice — µ-Galois orchestrator for choice path queries (T6).

Handles:
  forward  : source  (p1 | p2 | ... | pN)  ?b   (source is subject)
  backward : ?b  (p1 | p2 | ... | pN)  source    (?b is subject)

Each branch can be:
  - str                → 1-hop predicate (simple)
  - PathQuery (T4)     → sequential multi-hop, MuGaloisMultiPath
  - RecursivePattern   → recursive closure, MuGaloisRec
  - tuple(T4, T5)      → hybrid sequential+recursive

Decision tree:

  Partition branches:
    simple   = [b if is_1hop(b)]
    complex  = [b if not is_1hop(b)]

  Hybrid execution:
    simple branches  → Holistic OR (conf decides) or parallel LLMScan
    complex branches → parallel with best orchestrator
    result = union

  For simple branches:
    conf = LLMChoiceConf()              [1 call]
    conf > τ_high → Holistic OR, no motivational
    conf > τ_low  → Holistic OR
                    est = LLMEstimateChoiceSize()  [1 call, capped at 50]
                    len < est×COV → motivational
    conf ≤ τ_low  → parallel LLMScan per branch

  For complex branches (always parallel):
    PathQuery        → MuGaloisMultiPath (T4)
    RecursivePattern → MuGaloisRec (T5)
    tuple(T4, T5)    → sequential then recursive on seeds

Thresholds: τ_high=0.60  τ_low=0.40  COVERAGE=0.80
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

from mugalois.choice.choice_path import (
    ChoicePath, LLMChoiceConf, LLMEstimateChoiceSize,
    LLMBranchSimilarity,
)
from mugalois.core.types import Environment, RecursivePattern
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import BaseLLM

TAU_HIGH = 0.60
TAU_LOW  = 0.40
COVERAGE = 0.80

CALM_FOLLOW_UP = (
    "If, and only if, you are highly confident there are a few more "
    "you missed, add a small number now. Do not guess. "
    "If you are not sure, return an empty list — "
    "that is the expected, normal answer.\n"
    "List ONLY new values not already in the list above.\n"
    "Respond ONLY in valid JSON following the schema provided."
)


# ── Branch classification ─────────────────────────────────────────────

def _is_simple(branch) -> bool:
    return isinstance(branch, str)


def _partition(branches: list) -> tuple[list, list]:
    simple   = [b for b in branches if _is_simple(b)]
    complex_ = [b for b in branches if not _is_simple(b)]
    return simple, complex_


# ── Prompt builders ───────────────────────────────────────────────────

def _branch_triple(b, path: ChoicePath) -> str:
    """Build one explicit natural language triple for a branch."""
    if isinstance(b, str):
        if path.direction == "forward":
            return f"{path.source} {b} {path.target_var}"
        else:
            return f"{path.target_var} {b} {path.source}"
    elif hasattr(b, "hops"):
        chain = " → ".join(h.p for h in b.hops)
        return f"{path.source} {chain} {path.target_var}"
    elif hasattr(b, "p"):
        op = "+" if getattr(b, "operator", "plus") == "plus" else "*"
        return f"{path.source} {b.p}{op} {path.target_var}"
    elif isinstance(b, tuple) and len(b) == 2:
        seq, rec = b
        chain = " → ".join(h.p for h in seq.hops)
        op    = "+" if getattr(rec, "operator", "plus") == "plus" else "*"
        return f"{path.source} {chain} → {rec.p}{op} {path.target_var}"
    return str(b)


def _holistic_prompt(path: ChoicePath) -> str:
    """
    One explicit triple per branch joined by OR.
    Forward : source predicate ?b
    Backward: ?b predicate source
    """
    triples = [_branch_triple(b, path) for b in path.branches]
    or_expr = " OR ".join(triples)
    return (
        f"Find all values of {path.target_var} such that "
        f"{path.target_var} satisfies at least one of the following: "
        f"{or_expr}. "
        f"Be exhaustive — there are likely several results. "
        f"Include every single value of {path.target_var} you know. "
        f"Respond ONLY in valid JSON following the schema provided."
    )


def _motivational_prompt(path: ChoicePath, already: set[str]) -> str:
    return (
        f"{_holistic_prompt(path)}\n\n"
        f"Already found: {', '.join(sorted(already))}\n\n"
        f"{CALM_FOLLOW_UP}"
    )


# ── Branch evaluation ─────────────────────────────────────────────────

def _eval_simple_branch(
    branch:     str,
    path:       ChoicePath,
    llm:        BaseLLM,
) -> set[str]:
    """
    Evaluate one simple 1-hop branch via LLMScan.
    Respects direction: forward or backward.
    """
    from mugalois.core.types import TriplePattern
    from mugalois.scans.ochestror_scan import LLMScan

    if path.direction == "forward":
        pattern = TriplePattern(path.source, branch, path.target_var)
        triples = LLMScan(pattern=pattern, env=Environment(),
                          gamma=Environment(), llm=llm)
        return {t.o for t in triples}
    else:
        # backward: ?b [branch] source
        pattern = TriplePattern(path.target_var, branch, path.source)
        triples = LLMScan(pattern=pattern, env=Environment(),
                          gamma=Environment(), llm=llm)
        return {t.s for t in triples}


def _eval_complex_branch(
    branch,
    path:   ChoicePath,
    llm:    BaseLLM,
    gamma:  Environment,
) -> set[str]:
    """
    Evaluate one complex branch with the best orchestrator.
    """
    from mugalois.paths.path_query import PathQuery

    if isinstance(branch, PathQuery):
        from mugalois.paths.mugalois_multi import MuGaloisMultiPath
        return MuGaloisMultiPath(branch, llm, gamma=gamma)

    if isinstance(branch, RecursivePattern):
        from mugalois.rec.mugalois_rec import MuGaloisRec
        return MuGaloisRec(branch, llm, gamma=gamma)

    if isinstance(branch, tuple) and len(branch) == 2:
        seq_path, rec_pattern = branch
        from mugalois.paths.mugalois_multi import MuGaloisMultiPath
        from mugalois.rec.mugalois_rec import MuGaloisRec
        seq_results = MuGaloisMultiPath(seq_path, llm, gamma=gamma)
        result = set()
        for seed in seq_results:
            seeded = RecursivePattern(
                seed, rec_pattern.p, rec_pattern.o, rec_pattern.operator
            )
            result |= MuGaloisRec(seeded, llm, gamma=gamma)
        return result

    raise ValueError(f"Unknown complex branch type: {type(branch)}")


# ── Parallel helpers ──────────────────────────────────────────────────

def _parallel_simple(simple: list, path: ChoicePath, llm: BaseLLM) -> set[str]:
    result = set()
    with ThreadPoolExecutor(max_workers=len(simple)) as executor:
        futures = {
            executor.submit(_eval_simple_branch, b, path, llm): b
            for b in simple
        }
        for future in as_completed(futures):
            try:
                result |= future.result()
            except Exception as e:
                print(f"  [MuGaloisChoice] simple branch error: {e}")
    return result


def _parallel_complex(
    complex_: list, path: ChoicePath,
    llm: BaseLLM, gamma: Environment,
) -> set[str]:
    result = set()
    with ThreadPoolExecutor(max_workers=len(complex_)) as executor:
        futures = {
            executor.submit(_eval_complex_branch, b, path, llm, gamma): b
            for b in complex_
        }
        for future in as_completed(futures):
            try:
                result |= future.result()
            except Exception as e:
                print(f"  [MuGaloisChoice] complex branch error: {e}")
    return result


# ── Simple branch strategy ────────────────────────────────────────────

def _eval_simple_branches(
    simple:       list,
    path:         ChoicePath,
    llm:          BaseLLM,
    tau_high:     float,
    tau_low:      float,
    coverage:     float,
    verbose:      bool,
    motivational: bool = True,
) -> set[str]:
    """
    Handle simple branches with full decision model:

    Signal 1 — Confidence    : chooses holistic vs parallel
    Signal 2 — Similarity    : modulates holistic vs parallel for conf > τ_low
    Signal 3 — Cardinality   : triggers motivational after holistic

    similar  + conf > τ_low  → parallel LLMScan (holistic too broad)
    distinct + conf > τ_low  → Holistic OR + cardinality check
    any      + conf ≤ τ_low  → parallel LLMScan
    """
    simple_path = ChoicePath(
        source=path.source, target_var=path.target_var,
        branches=simple, direction=path.direction,
    )

    # Signal 1 — Confidence
    conf = LLMChoiceConf(simple_path, llm)
    if verbose:
        print(f"  [simple] conf={conf:.2f} τ_high={tau_high} τ_low={tau_low}")

    if conf <= tau_low:
        if verbose: print(f"  [simple] → parallel LLMScan (conf ≤ τ_low)")
        return _parallel_simple(simple, path, llm)

    # Signal 2 — Branch similarity
    similarity = LLMBranchSimilarity(simple_path, llm)
    if verbose: print(f"  [simple] similarity={similarity}")

    if similarity == "similar":
        # Similar branches → parallel safer (holistic too broad)
        if verbose: print(f"  [simple] → parallel LLMScan (similar branches)")
        return _parallel_simple(simple, path, llm)

    # Distinct branches → Holistic OR
    if verbose: print(f"  [simple] → Holistic OR (distinct branches)")
    prompt = _holistic_prompt(simple_path)
    result = json_to_values(llm.chat(build_value_messages(prompt)).text)

    # Signal 3 — Cardinality check (only if motivational enabled)
    est = LLMEstimateChoiceSize(simple_path, llm)
    if verbose: print(f"  [simple] est={est} result={len(result)}")
    if motivational and est > 0 and len(result) < coverage * est:
        if verbose: print(f"  [simple] → motivational (below coverage)")
        follow   = _motivational_prompt(simple_path, result)
        new_vals = json_to_values(
            llm.chat(build_value_messages(follow)).text
        ) - result
        if verbose: print(f"  [simple] motivational added {len(new_vals)} values")
        result |= new_vals
    else:
        if verbose: print(f"  [simple] → no motivational (at/above coverage)")
    return result


# ── Main orchestrator ─────────────────────────────────────────────────

def MuGaloisChoice(
    path:         ChoicePath,
    llm:          BaseLLM,
    gamma:        Environment = None,
    tau_high:     float = TAU_HIGH,
    tau_low:      float = TAU_LOW,
    coverage:     float = COVERAGE,
    verbose:      bool  = False,
    motivational: bool  = True,    # ← nouveau paramètre
) -> set[str]:
    """
    µ-Galois orchestrator for choice paths (T6).
    Supports forward and backward directions.
    Hybrid execution: simple → holistic/parallel, complex → parallel.
    """
    gamma = gamma or Environment()
    simple, complex_ = _partition(path.branches)

    if verbose:
        print(f"[MuGaloisChoice] {len(path.branches)} branches "
              f"({path.direction}): {len(simple)} simple, {len(complex_)} complex")

    result = set()

    if simple:
        result |= _eval_simple_branches(
            simple, path, llm, tau_high, tau_low, coverage, verbose,
            motivational=motivational,
        )

    if complex_:
        if verbose: print(f"  [complex] → parallel {len(complex_)} branches")
        result |= _parallel_complex(complex_, path, llm, gamma)

    return result