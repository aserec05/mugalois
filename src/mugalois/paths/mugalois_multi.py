# src/mugalois/paths/mugalois_multi.py
"""
MuGaloisMultiPath — µ-Galois orchestrator for N-hop path queries (T4).

Three signals, distinct roles:
  conf    = LLMChainConf()      → chooses STRATEGY        [1 call]
  est     = LLMEstimateSize()   → triggers MOTIVATIONAL   [1 call, only when needed]
  routing = decide_routing()    → chooses GD/DG direction [1 call, T4b only]

Decision tree:

  numeric conditions → two-step filter

  T4a (final variable) — Chain always bad:
    conf > τ_high  → Holistic atom, no motivational
    τ_low < conf   → N>3: LLMGenerateNLPrompt; N≤3: structured prompt
                     est → motivational if len(result) < est×COV
    conf ≤ τ_low   → N>3: LLMGenerateNLPrompt; N≤3: structured prompt
                     + motivational always

  T4b (intermediate variable):
    conf > τ_high  → routing holistic halves, no motivational
    τ_low < conf   → routing holistic halves
                     est → post-motivational if len(result) < est×COV
    conf ≤ τ_low   → Chain standard (GD/DG/Split, LLMScan per hop)
                     no motivational

Thresholds: τ_high=0.50  τ_low=0.35  COVERAGE=0.80
"""
from __future__ import annotations

from mugalois.paths.path_query import PathQuery
from mugalois.paths.multi_scan import (
    LLMMultiHopScan, LLMChainConf, LLMEstimateSize,
    _gen_multi_hop_scan_prompt, CALM_FOLLOW_UP_PROMPT,
)
from mugalois.paths.chain import scan_chain, scan_chain_from_bound
from mugalois.paths.split import scan_split
from mugalois.paths.routing import decide_routing
from mugalois.core.types import Environment
from mugalois.core.prompts import build_value_messages, build_messages
from mugalois.core.parser import json_to_values
from mugalois.llm.llm_client import BaseLLM

TAU_HIGH = 0.50
TAU_LOW  = 0.35
COVERAGE = 0.80


# ── Helpers ───────────────────────────────────────────────────────────

def _is_numeric_condition(c) -> bool:
    try:
        float(c.val)
        return True
    except (ValueError, AttributeError):
        return False


def _is_intermediate_variable(path: PathQuery) -> bool:
    idx = next(
        (i for i, h in enumerate(path.hops) if h.o == path.target_var), None
    )
    return idx is not None and idx < len(path.hops) - 1


def _should_motivate(result: set, est_size: int) -> bool:
    """True if result is below COVERAGE × estimated size."""
    if est_size <= 0:
        return False
    return len(result) < COVERAGE * est_size


def LLMGenerateNLPrompt(path: PathQuery, llm: BaseLLM) -> str:
    """
    Ask the LLM to convert the structural path into a natural language
    question. More effective than the abstract structural prompt on long
    chains (N>3) — general, works on any dataset.
    """
    chain  = " → ".join(f"[{h.p}]" for h in path.hops)
    prompt = (
        f"Convert this path query into a single natural language question:\n"
        f"Starting from '{path.source}', follow: {chain}, ending at '{path.target}'.\n"
        f"The question asks for all values of {path.target_var}.\n\n"
        f"Write the question only. No explanation. No preamble."
    )
    resp = llm.chat(build_messages(prompt))
    return resp.text.strip()


def _holistic_atom(path: PathQuery, llm: BaseLLM) -> set[str]:
    """Single structural call, no motivational."""
    prompt = _gen_multi_hop_scan_prompt(path)
    return json_to_values(llm.chat(build_value_messages(prompt)).text)


def _holistic_nl(path: PathQuery, llm: BaseLLM) -> set[str]:
    """
    NL-generated prompt call — used for T4a N>3.
    The LLM generates the question itself from the path structure.
    """
    nl_question = LLMGenerateNLPrompt(path, llm)
    full_prompt  = (
        f"{nl_question}\n\n"
        f"Be exhaustive — there are likely several results, not just one. "
        f"List every single one you know.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    return json_to_values(llm.chat(build_value_messages(full_prompt)).text)


def _holistic_for_t4a(path: PathQuery, llm: BaseLLM) -> set[str]:
    """Choose prompt type based on N."""
    if path.n_hops > 3:
        return _holistic_nl(path, llm)
    return _holistic_atom(path, llm)


def _motivational_followup(path: PathQuery, result: set[str], llm: BaseLLM) -> set[str]:
    """Calm follow-up — general, works for any T4 path."""
    if not result:
        return result
    already   = ", ".join(sorted(result))
    follow_up = (
        f"{_gen_multi_hop_scan_prompt(path)}\n\n"
        f"Already found: {already}\n\n"
        f"{CALM_FOLLOW_UP_PROMPT}\n"
        f"List ONLY new values not already in the list above.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    new_vals = json_to_values(llm.chat(build_value_messages(follow_up)).text) - result
    return result | new_vals


def _gen_numeric_filter_prompt(candidates, conditions, labels):
    cand_str = "\n".join(f"  - {c}" for c in sorted(candidates))
    op_word  = {">": "greater than", "<": "less than",
                ">=": "at least",   "<=": "at most",
                "=": "equal to",    "!=": "different from"}
    filters = []
    for c in conditions:
        label = labels.get(c.var, "value")
        op    = op_word.get(c.op, c.op)
        try:
            val_str = f"{int(c.val):,}"
        except ValueError:
            val_str = c.val
        filters.append(f"{label} {op} {val_str}")
    return (
        f"From the following list:\n{cand_str}\n\n"
        f"Which ones have {' AND '.join(filters)}?\n"
        f"Return ONLY items from the list above. "
        f"Do not add any item not in the list.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )


def _scan_with_numeric_filter(path, llm, gamma, labels, max_iter, verbose):
    path_no_cond = PathQuery(
        hops=path.hops, target_var=path.target_var, conditions=[],
    )
    if hasattr(path, "condition_labels"):
        path_no_cond.condition_labels = path.condition_labels  # type: ignore
    candidates = LLMMultiHopScan(path_no_cond, llm)
    if not candidates:
        return set()
    prompt   = _gen_numeric_filter_prompt(candidates, path.conditions, labels)
    filtered = json_to_values(llm.chat(build_value_messages(prompt)).text)
    return filtered & candidates


def _half_scan_atom(half_path: PathQuery, llm: BaseLLM) -> set[str]:
    """
    Scan one routing half.
    1-hop  → LLMScan direct (optimal)
    2+ hop → MuGaloisMultiPath récursif (sous-arbre complet)
    """
    if half_path.n_hops == 1:
        import copy
        from mugalois.scans.ochestror_scan import LLMScan
        hop     = half_path.hops[0]
        triples = LLMScan(pattern=hop, env=Environment(),
                          gamma=Environment(), llm=llm)
        return {t.o for t in triples} if hop.o_is_var() else {t.s for t in triples}
    
    return MuGaloisMultiPath(half_path, llm)


def _routing_holistic(path, llm, gamma, max_iter, verbose) -> set[str]:
    """Routing where multi-hop halves are solved holistically."""
    routing = decide_routing(path, llm, gamma, verbose=verbose)
    if routing["strategy"] == "bound":
        return scan_chain_from_bound(
            path=path, var=routing["bound_var"],
            seeds=gamma.get(routing["bound_var"]),
            llm=llm, gamma=gamma, max_iter=max_iter, verbose=verbose,
        )
    elif routing["strategy"] == "chain":
        if _is_intermediate_variable(path):
            left  = path.left_half(path.target_var)
            right = path.right_half(path.target_var)
            lv    = _half_scan_atom(left, llm)
            rv    = _half_scan_atom(right.reversed(), llm)
            return lv & rv if rv else lv
        return scan_chain(
            path=path, llm=llm, direction=routing["direction"],
            gamma=gamma, max_iter=max_iter, verbose=verbose,
        )
    else:
        return scan_split(
            path=path, split_var=routing["split_var"], llm=llm,
            gamma=gamma, max_iter=max_iter, verbose=verbose,
        )


def _routing_chain(path, llm, gamma, max_iter, verbose) -> set[str]:
    """Standard Chain routing (GD/DG/Split with LLMScan per hop)."""
    routing = decide_routing(path, llm, gamma, verbose=verbose)
    if routing["strategy"] == "bound":
        return scan_chain_from_bound(
            path=path, var=routing["bound_var"],
            seeds=gamma.get(routing["bound_var"]),
            llm=llm, gamma=gamma, max_iter=max_iter, verbose=verbose,
        )
    elif routing["strategy"] == "chain":
        return scan_chain(
            path=path, llm=llm, direction=routing["direction"],
            gamma=gamma, max_iter=max_iter, verbose=verbose,
        )
    else:
        return scan_split(
            path=path, split_var=routing["split_var"], llm=llm,
            gamma=gamma, max_iter=max_iter, verbose=verbose,
        )


# ── Main orchestrator ─────────────────────────────────────────────────

def MuGaloisMultiPath(
    path:      PathQuery,
    llm:       BaseLLM,
    gamma:     Environment = None,
    tau_high:  float = TAU_HIGH,
    tau_low:   float = TAU_LOW,
    coverage:  float = COVERAGE,
    max_iter:  int   = 5,
    verbose:   bool  = False,
) -> set[str]:
    gamma  = gamma or Environment()
    labels = getattr(path, "condition_labels", {})

    if verbose:
        print(f"[MuGaloisMultiPath] {path.n_hops}-hop: {path}")

    # ── 1. Numeric conditions ─────────────────────────────────────────
    if any(_is_numeric_condition(c) for c in path.conditions):
        if verbose: print(f"  → two-step numeric filter")
        return _scan_with_numeric_filter(path, llm, gamma, labels, max_iter, verbose)

    # ── 2. Single-hop ─────────────────────────────────────────────────
    if path.n_hops == 1:
        import copy
        from mugalois.scans.ochestror_scan import LLMScan
        hop     = path.hops[0]
        triples = LLMScan(pattern=hop, env=Environment(),
                          gamma=copy.deepcopy(gamma), llm=llm, max_iter=max_iter)
        return {t.o for t in triples} if hop.o_is_var() else {t.s for t in triples}

    # ── 3. Confidence ─────────────────────────────────────────────────
    # Short-circuit: if tau_high=0.0 → always holistic, no routing/conf calls
    if tau_high == 0.0:
        # Always use holistic atom regardless of T4a/T4b — avoids all LLM routing calls
        return _holistic_for_t4a(path, llm)

    conf   = LLMChainConf(path, llm)
    is_t4b = _is_intermediate_variable(path)

    if verbose:
        print(f"  conf={conf:.2f} N={path.n_hops} T4{'b' if is_t4b else 'a'} "
              f"τ_high={tau_high} τ_low={tau_low}")

    # ━━━ T4a — Chain always bad ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not is_t4b:
        if conf > tau_high:
            if verbose: print(f"  → Holistic atom, no motivational (T4a, conf > τ_high)")
            return _holistic_for_t4a(path, llm)

        # Get initial result (NL prompt if N>3, structured otherwise)
        result = _holistic_for_t4a(path, llm)

        if conf > tau_low:
            # Moderate confidence — cardinality decides motivational
            est = LLMEstimateSize(path, llm)
            if verbose: print(f"  T4a moderate: est={est} result={len(result)}")
            if _should_motivate(result, est):
                if verbose: print(f"  → motivational (below coverage)")
                result = _motivational_followup(path, result, llm)
            else:
                if verbose: print(f"  → no motivational (at/above coverage)")
        else:
            # Low confidence — always motivational
            if verbose: print(f"  → motivational always (T4a, conf ≤ τ_low)")
            result = _motivational_followup(path, result, llm)

        return result

    # ━━━ T4b — routing modulated by conf ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if conf > tau_high:
        if verbose: print(f"  → routing holistic, no motivational (T4b, conf > τ_high)")
        result = _routing_holistic(path, llm, gamma, max_iter, verbose)

    elif conf > tau_low:
        if verbose: print(f"  → routing holistic + cardinality check (T4b, moderate)")
        result = _routing_holistic(path, llm, gamma, max_iter, verbose)
        est    = LLMEstimateSize(path, llm)
        if verbose: print(f"  T4b moderate: est={est} result={len(result)}")
        if _should_motivate(result, est):
            if verbose: print(f"  → post-motivational (below coverage)")
            result = _motivational_followup(path, result, llm)

    else:
        if verbose: print(f"  → Chain standard (T4b, conf ≤ τ_low)")
        result = _routing_chain(path, llm, gamma, max_iter, verbose)

    if not result:
        if verbose: print(f"  → fallback chain(left)")
        result = scan_chain(path=path, llm=llm, direction="left",
                            gamma=gamma, max_iter=max_iter, verbose=verbose)
    return result


def MuGaloisPath(
    s: str, p1: str, p2: str, t: str,
    llm: BaseLLM, gamma: Environment = None,
    tau_high: float = TAU_HIGH, tau_low: float = TAU_LOW,
    coverage: float = COVERAGE, max_iter: int = 5, verbose: bool = False,
) -> set[str]:
    """Backward-compatible T3 two-hop wrapper."""
    path = PathQuery.two_hop(s, p1, p2, t)
    return MuGaloisMultiPath(
        path=path, llm=llm, gamma=gamma,
        tau_high=tau_high, tau_low=tau_low, coverage=coverage,
        max_iter=max_iter, verbose=verbose,
    )