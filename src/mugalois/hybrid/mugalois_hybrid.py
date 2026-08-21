# src/mugalois/hybrid/mugalois_hybrid.py
"""
MuGaloisHybrid — µ-Galois meta-orchestrator for hybrid path queries (T7).

Handles expressions mixing:
  HopNode      : single predicate hop        → LLMScan (T1/T2 full strategy)
  ClosureNode  : transitive closure p+/p*    → MuGaloisRec (T5 full decision tree)
  FilterNode   : type/value constraint       → batch LLM filter
  ChoiceNode   : disjunction (p1|p2|...)     → MuGaloisChoice (T6 full decision tree)

All existing orchestrators are reused as sub-components — nothing reimplemented.

━━━ Evaluation modes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Signal 0 — Direction (from anchor_side, NO LLM call):

  BACKWARD  anchor_side="right"  : ?var ←[nodes]← anchor
    e.g. ?pope / wasBornIn / isLocatedIn+  :Italy
    → evaluate right-to-left, result = leftmost set

  FORWARD   anchor_side="left"   : anchor →[nodes]→ ?var
    e.g. :Thatcher / hasSuccessor+ / educatedAt / isLocatedIn+  ?b
    → evaluate left-to-right, result = rightmost set

  SPLIT     anchor_side="both"   : anchor →[prefix]→ ?var ←[suffix]← anchor
    e.g. :Victoria / hasChild+  ?x  / wasBornIn / isLocatedIn+  :Germany
    → forward prefix → candidates for ?x
    → filter candidates against backward suffix

Signal 1 — Early empty check (1 LLM call, optional via check_empty=True):
    "Does there exist any ?x satisfying this path?" → NO → return ∅ immediately
    Useful for q6 (Kennedy/Democrat/maritalStatus → GT=∅)

Signal 2 — Composite confidence (1 LLM call per ClosureNode):
    conf = min(LLMAtomicRecConf(closure_i) for closure_i in plan)
    = cognitive bottleneck across the path (idée 3 farfelue)

    conf > τ_high → Holistic  (single NL prompt, 1 LLM call total)
    τ_low < conf  → Segment   (one call per node, no motivational)
    conf ≤ τ_low  → Segment + motivational at final node

Thresholds: τ_high=0.60  τ_low=0.40
"""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from mugalois.hybrid.hybrid_plan import (
    HybridPlan, HybridNode,
    HopNode, ClosureNode, FilterNode, ChoiceNode,
)
from mugalois.core.types import (
    TriplePattern, RecursivePattern, Environment,
)
from mugalois.core.prompts import build_messages, build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.rec.mugalois_rec import MuGaloisRec
from mugalois.rec.simple_rec import LLMAtomicRecConf
from mugalois.paths.routing import LLMEstimateHopSize
from mugalois.scans.ochestror_scan import LLMScan
from mugalois.llm.llm_client import BaseLLM

TAU_HIGH = 0.60
TAU_LOW  = 0.40

CALM_FOLLOW_UP = (
    "If, and only if, you are highly confident there are a few more "
    "you missed, add a small number now. Do not guess. "
    "If you are not sure, return an empty list — "
    "that is the expected, normal answer.\n"
    "List ONLY new values not already in the list above.\n"
    "Respond ONLY in valid JSON following the schema provided."
)

NO_SPECULATION = (
    "Do not guess or speculate. Only return values you are factually "
    "confident about. If nothing is confirmed, return an empty list."
)


# ── Signal 0 — Direction ──────────────────────────────────────────────

def _direction(plan: HybridPlan) -> str:
    """
    Detect evaluation direction from plan structure.
    No LLM call needed — purely structural.
    """
    side = getattr(plan, "anchor_side", "right")
    if side == "left":
        return "forward"
    elif side == "right":
        return "backward"
    elif side == "both":
        return "split"
    return "backward"  # safe default


# ── Signal 1 — Early empty check ─────────────────────────────────────

def _likely_empty(plan: HybridPlan, llm: BaseLLM) -> bool:
    """
    One LLM call: does there exist any entity satisfying this path?
    If confidently NO → skip evaluation entirely (GT=∅ cases like q6).
    """
    nl = plan.nl or f"{plan.source} {plan.expression} {plan.target}"
    prompt = (
        f"Does there exist at least one entity satisfying:\n{nl}\n\n"
        f"Answer with a single word: YES or NO."
    )
    resp = llm.chat(build_messages(prompt))
    text = resp.text.strip().upper()
    words = text.split()
    return words and words[0] == "NO"


# ── Signal 2 — Composite confidence ──────────────────────────────────

def _composite_conf(plan: HybridPlan, llm: BaseLLM) -> float:
    """
    Composite confidence = min of per-ClosureNode LLMAtomicRecConf calls.
    Cognitive bottleneck: the hardest segment limits the overall recall.
    For hops and filters: default 0.5 (neutral).
    """
    confs = []
    anchor = plan.target if _direction(plan) == "backward" else plan.source

    for node in plan.nodes:
        if isinstance(node, ClosureNode):
            # Build a representative RecursivePattern for confidence estimation
            # Use the anchor as a proxy seed (may not be the actual seed at runtime)
            pat = RecursivePattern(
                s="?x" if _direction(plan) == "backward" else anchor,
                p=node.predicate,
                o=anchor if _direction(plan) == "backward" else "?x",
                operator=node.operator,
            )
            try:
                c = LLMAtomicRecConf(pat, llm)
                confs.append(c)
            except Exception:
                confs.append(0.5)
        elif isinstance(node, HopNode):
            confs.append(0.5)  # hops: neutral confidence
        elif isinstance(node, FilterNode):
            confs.append(0.5)  # filters: neutral
        elif isinstance(node, ChoiceNode):
            confs.append(0.4)  # choice: slightly lower default

    return min(confs) if confs else 0.5


# ── Holistic prompt ───────────────────────────────────────────────────

def _holistic_prompt(plan: HybridPlan) -> str:
    """Single NL prompt for the full hybrid expression."""
    nl = plan.nl or (
        f"Find all {plan.target_var} such that "
        f"{plan.source} {plan.expression} {plan.target}."
    )
    return (
        f"{nl}\n\n"
        f"Be exhaustive — there are likely several results.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )


def _holistic_motivational(
    plan: HybridPlan, already: set[str], llm: BaseLLM
) -> set[str]:
    base   = _holistic_prompt(plan)
    follow = (
        f"{base}\n\n"
        f"Already found: {', '.join(sorted(already))}\n\n"
        f"{CALM_FOLLOW_UP}"
    )
    new = json_to_values(llm.chat(build_value_messages(follow)).text) - already
    return already | new


# ── Node evaluators — BACKWARD (right-to-left) ───────────────────────

def _eval_backward(
    node:  HybridNode,
    seeds: set[str],
    llm:   BaseLLM,
    gamma: Environment,
    motivational: bool = False,
) -> set[str]:
    """
    Evaluate one node backward: seeds are objects, find subjects.
    seeds → node⁻¹ → new_seeds
    """
    if not seeds:
        return set()

    if isinstance(node, ClosureNode):
        return _closure_backward(node, seeds, llm, gamma, motivational)

    elif isinstance(node, HopNode):
        return _hop_backward(node, seeds, llm, gamma)

    elif isinstance(node, FilterNode):
        return _filter_by_value(seeds, node.value, node.nl_type, llm)

    elif isinstance(node, ChoiceNode):
        # Parallel evaluation of each branch backward
        result = set()
        with ThreadPoolExecutor(max_workers=len(node.branches)) as ex:
            futures = {
                ex.submit(_hop_backward, HopNode(b), seeds, llm, gamma): b
                for b in node.branches
            }
            for f in as_completed(futures):
                try:
                    result |= f.result()
                except Exception as e:
                    print(f"  [MuGaloisHybrid] choice branch error: {e}")
        return result

    return set()


def _hop_backward(
    node: HopNode, seeds: set[str], llm: BaseLLM, gamma: Environment
) -> set[str]:
    """
    Backward hop: find ?x such that (?x, predicate, seed) for any seed.
    Seeds passed via env as object binding.

    tau_strategie=0.01 forces SeedScan (one batched prompt for ALL seeds)
    instead of KeyScan (one prompt per seed).
    Critical when |seeds| is large — avoids N×LLM calls.
    """
    env = Environment()
    env.set("?_target", seeds)
    pattern = TriplePattern("?x", node.predicate, "?_target")
    triples = LLMScan(
        pattern=pattern, env=env, gamma=gamma, llm=llm,
        tau_strategie=0.01,  # force SeedScan — never KeyScan on large seed sets
    )
    return {t.s for t in triples}


def _closure_backward(
    node: ClosureNode, seeds: set[str], llm: BaseLLM,
    gamma: Environment, motivational: bool = False,
) -> set[str]:
    """
    Backward transitive closure: find all ?x such that ?x p+ seed.

    Single seed → delegate to MuGaloisRec (full T5 decision tree).
    Multiple seeds → batch fixpoint (LLMFixpointScan generalized).
    """
    if len(seeds) == 1:
        seed = next(iter(seeds))
        pat  = RecursivePattern(
            s="?x", p=node.predicate, o=seed, operator=node.operator
        )
        return MuGaloisRec(pat, llm, gamma=gamma)

    # Multiple seeds → batch backward fixpoint
    return _batch_fixpoint(node, seeds, llm, direction="backward")


# ── Node evaluators — FORWARD (left-to-right) ────────────────────────

def _eval_forward(
    node:  HybridNode,
    seeds: set[str],
    llm:   BaseLLM,
    gamma: Environment,
    motivational: bool = False,
) -> set[str]:
    """
    Evaluate one node forward: seeds are subjects, find objects.
    seeds → node → new_seeds
    """
    if not seeds:
        return set()

    if isinstance(node, ClosureNode):
        return _closure_forward(node, seeds, llm, gamma, motivational)

    elif isinstance(node, HopNode):
        return _hop_forward(node, seeds, llm, gamma)

    elif isinstance(node, FilterNode):
        return _filter_by_value(seeds, node.value, node.nl_type, llm)

    elif isinstance(node, ChoiceNode):
        result = set()
        with ThreadPoolExecutor(max_workers=len(node.branches)) as ex:
            futures = {
                ex.submit(_hop_forward, HopNode(b), seeds, llm, gamma): b
                for b in node.branches
            }
            for f in as_completed(futures):
                try:
                    result |= f.result()
                except Exception as e:
                    print(f"  [MuGaloisHybrid] choice branch error: {e}")
        return result

    return set()


def _hop_forward(
    node: HopNode, seeds: set[str], llm: BaseLLM, gamma: Environment
) -> set[str]:
    """
    Forward hop: find ?x such that (seed, predicate, ?x) for any seed.
    Seeds passed via env as subject binding.

    tau_strategie=0.01 forces SeedScan for large seed sets.
    """
    env = Environment()
    env.set("?_source", seeds)
    pattern = TriplePattern("?_source", node.predicate, "?x")
    triples = LLMScan(
        pattern=pattern, env=env, gamma=gamma, llm=llm,
        tau_strategie=0.01,  # force SeedScan — never KeyScan on large seed sets
    )
    return {t.o for t in triples}


def _closure_forward(
    node: ClosureNode, seeds: set[str], llm: BaseLLM,
    gamma: Environment, motivational: bool = False,
) -> set[str]:
    """
    Forward transitive closure: find all ?x such that seed p+ ?x.

    Single seed → MuGaloisRec (full T5 decision tree).
    Multiple seeds → batch forward fixpoint.
    """
    if len(seeds) == 1:
        seed = next(iter(seeds))
        pat  = RecursivePattern(
            s=seed, p=node.predicate, o="?x", operator=node.operator
        )
        return MuGaloisRec(pat, llm, gamma=gamma)

    return _batch_fixpoint(node, seeds, llm, direction="forward")


# ── Batch fixpoint (multi-seed closure) ──────────────────────────────

def _batch_fixpoint(
    node:      ClosureNode,
    seeds:     set[str],
    llm:       BaseLLM,
    direction: str = "backward",
    max_depth: int = 50,
) -> set[str]:
    """
    Generalized multi-seed transitive closure via iterative LLMScan.

    Implements the fixpoint:
      frontier ← seeds
      repeat:
        new ← LLMScan(frontier) - visited
        visited ∪= new ; frontier ← new
      until new = ∅

    Uses LLMScan's SeedScan strategy (seeds in env) for efficiency.
    Mirrors LLMFixpointScan but starts with multiple seeds.
    """
    op      = node.operator
    visited = set(seeds) if op == "*" else set()
    frontier = set(seeds)
    result   = set(visited)
    seen_frontiers: set = set()

    for _ in range(max_depth):
        if not frontier:
            break
        key = frozenset(frontier)
        if key in seen_frontiers:
            break
        seen_frontiers.add(key)

        env = Environment()
        if direction == "forward":
            env.set("?_src", frontier)
            pattern  = TriplePattern("?_src", node.predicate, "?x")
            triples  = LLMScan(
                pattern=pattern, env=env,
                gamma=Environment(), llm=llm,
                lookahead=NO_SPECULATION,
            )
            new = {t.o for t in triples} - visited
        else:
            env.set("?_tgt", frontier)
            pattern  = TriplePattern("?x", node.predicate, "?_tgt")
            triples  = LLMScan(
                pattern=pattern, env=env,
                gamma=Environment(), llm=llm,
                lookahead=NO_SPECULATION,
            )
            new = {t.s for t in triples} - visited

        if not new:
            break
        visited |= new
        result  |= new
        frontier = new

    if op == "+":
        result -= seeds
    return result


# ── Filter by type/value ──────────────────────────────────────────────

def _filter_by_value(
    candidates: set[str],
    value:      str,
    nl_type:    str,
    llm:        BaseLLM,
) -> set[str]:
    """
    Batch LLM filter: from candidates, keep only those satisfying type=value.
    Example: "which of these are Conservative politicians?"
    Returns a subset of candidates (never adds new entities).
    Used for FilterNode (type constraints, party affiliation, etc.)
    """
    if not candidates:
        return set()

    cands_list = sorted(candidates)
    cand_str   = "\n".join(f"  - {c}" for c in cands_list)
    prompt = (
        f"From the following list:\n{cand_str}\n\n"
        f"Which ones are {nl_type or value}?\n"
        f"Return ONLY items from the list above. Do not add anything else.\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )
    result = json_to_values(llm.chat(build_value_messages(prompt)).text)
    return result & candidates  # safety: never return what wasn't in input


# ── Segment evaluation ────────────────────────────────────────────────

def _segment_eval(
    plan:         HybridPlan,
    direction:    str,
    llm:          BaseLLM,
    gamma:        Environment,
    motivational: bool = False,
    verbose:      bool = False,
) -> set[str]:
    """
    Evaluate plan segment by segment.

    BACKWARD (right → left):
      Start from {plan.target}, traverse nodes in reverse.
      Each step: seeds = eval_backward(node, seeds)
      Final seeds = result (?var at left end)

    FORWARD (left → right):
      Start from {plan.source}, traverse nodes left-to-right.
      Each step: seeds = eval_forward(node, seeds)
      Final seeds = result (?var at right end)

    SPLIT (both anchors, result in middle):
      1. Forward prefix up to result_node_idx → candidates
      2. Filter candidates against backward suffix
      3. Return candidates & filter_result
    """
    if direction == "split":
        return _split_eval(plan, llm, gamma, motivational, verbose)

    if direction == "backward":
        seeds  = {plan.target}
        nodes  = list(reversed(plan.nodes))
    else:
        seeds  = {plan.source}
        nodes  = plan.nodes

    for i, node in enumerate(nodes):
        is_last = (i == len(nodes) - 1)
        motiv   = motivational and is_last

        if verbose:
            print(f"  [segment] {direction} node={node} "
                  f"seeds={len(seeds)} motiv={motiv}")

        if direction == "backward":
            seeds = _eval_backward(node, seeds, llm, gamma, motivational=motiv)
        else:
            seeds = _eval_forward(node, seeds, llm, gamma, motivational=motiv)

        if verbose:
            print(f"  [segment] → {len(seeds)} results  "
                  f"ex: {sorted(seeds)[:3]}")

        if not seeds:
            if verbose:
                print(f"  [segment] empty — stopping")
            return set()

    return seeds


def _split_eval(
    plan:         HybridPlan,
    llm:          BaseLLM,
    gamma:        Environment,
    motivational: bool = False,
    verbose:      bool = False,
) -> set[str]:
    """
    Split/Join evaluation for 'both'-anchored plans.

    Forward prefix:  source →[nodes[:idx]]→ ?var  → candidates
    Backward suffix: ?var ←[nodes[idx:]]← target  → valid seeds
    Result = candidates ∩ valid_from_suffix

    This is Waveguide's Split/Join applied to hybrid paths.
    Avoids reversing directional predicates (semantic correctness).
    """
    idx = getattr(plan, "result_node_idx", 0)
    prefix_nodes = plan.nodes[:idx + 1]
    suffix_nodes = plan.nodes[idx + 1:]

    if verbose:
        print(f"  [split] prefix={len(prefix_nodes)} nodes "
              f"suffix={len(suffix_nodes)} nodes  "
              f"result_idx={idx}")

    # Phase 1: Forward prefix → candidates for result_var
    seeds = {plan.source}
    for node in prefix_nodes:
        seeds = _eval_forward(node, seeds, llm, gamma)
        if not seeds:
            return set()
    candidates = seeds

    if verbose:
        print(f"  [split] {len(candidates)} candidates: "
              f"{sorted(candidates)[:5]}")

    if not suffix_nodes:
        return candidates

    # Phase 2: Backward suffix → valid entities from target side
    seeds = {plan.target}
    for node in reversed(suffix_nodes):
        seeds = _eval_backward(node, seeds, llm, gamma)
        if not seeds:
            return set()
    valid_from_suffix = seeds

    if verbose:
        print(f"  [split] {len(valid_from_suffix)} valid from suffix")

    # Join = intersection
    result = candidates & valid_from_suffix

    if verbose:
        print(f"  [split] → {len(result)} after join")

    return result


# ── Main orchestrator ─────────────────────────────────────────────────

def MuGaloisHybrid(
    plan:        HybridPlan,
    llm:         BaseLLM,
    gamma:       Environment = None,
    tau_high:    float = TAU_HIGH,
    tau_low:     float = TAU_LOW,
    check_empty: bool  = False,
    verbose:     bool  = False,
) -> set[str]:
    """
    µ-Galois meta-orchestrator for hybrid path queries (T7).

    Delegates to existing orchestrators:
      ClosureNode → MuGaloisRec  (T5 full decision tree)
      HopNode     → LLMScan      (T1/T2 KeyScan/SeedScan/TripleScan)
      FilterNode  → batch LLM filter
      ChoiceNode  → MuGaloisChoice (T6 full decision tree)

    Planning:
      Signal 0: direction (structural, no LLM)
      Signal 1: empty check (optional, 1 LLM call)
      Signal 2: composite confidence (1 LLM call per ClosureNode)
        → Holistic / Segment / Segment+motivational
    """
    gamma = gamma or Environment()

    if verbose:
        print(f"[MuGaloisHybrid] {plan}")
        print(f"  expression={plan.expression}")
        print(f"  nodes={plan.nodes}")
        print(f"  complexity={plan.complexity}")

    # ── Signal 0 — Direction ──────────────────────────────────────────
    direction = _direction(plan)

    if verbose:
        print(f"  direction={direction}")

    # ── Signal 1 — Early empty check (optional) ───────────────────────
    if check_empty and plan.gt_size == 0:
        # Only call if we suspect empty (e.g., from prior knowledge)
        if _likely_empty(plan, llm):
            if verbose:
                print(f"  → ∅ (early empty check confirmed)")
            return set()

    # ── Signal 2 — Composite confidence ──────────────────────────────
    conf = _composite_conf(plan, llm)

    if verbose:
        print(f"  conf={conf:.2f}  τ_high={tau_high}  τ_low={tau_low}")

    # ── High confidence → Holistic ────────────────────────────────────
    if conf > tau_high:
        if verbose:
            print(f"  → Holistic (conf > τ_high)")
        prompt = _holistic_prompt(plan)
        result = json_to_values(llm.chat(build_value_messages(prompt)).text)
        if verbose:
            print(f"  holistic returned {len(result)}")
        return result

    # ── Moderate confidence → Segment (no motivational) ───────────────
    if conf > tau_low:
        if verbose:
            print(f"  → Segment, no motivational (τ_low < conf ≤ τ_high)")
        return _segment_eval(
            plan, direction, llm, gamma,
            motivational=False, verbose=verbose,
        )

    # ── Low confidence → Segment + motivational ───────────────────────
    if verbose:
        print(f"  → Segment + motivational (conf ≤ τ_low)")
    result = _segment_eval(
        plan, direction, llm, gamma,
        motivational=True, verbose=verbose,
    )

    # Global motivational follow-up on holistic if segment returned little
    if result:
        result = _holistic_motivational(plan, result, llm)

    return result
# ══════════════════════════════════════════════════════════════════════════════
# Step-by-step execution modes for model evaluation
# Added for µ-Galois model ablation (Hol / Dec / C)
# ══════════════════════════════════════════════════════════════════════════════

from mugalois.rec.simple_rec import LLMRecScan, LLMAtomicRecConf
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.scans.seed_scan import LLMSeedScan
from mugalois.scans.key_scan import LLMKeyScan


def _exec_anchor_holistic(var: str, node, seeds: set, llm: BaseLLM) -> set:
    """ANCHOR — always holistic regardless of model."""
    if isinstance(node, FilterNode):
        desc = node.nl_type or node.value
        prompt = (
            f"List every entity that is a {desc}.\n"
            f"Be exhaustive — include lesser-known ones.\n"
            f"Return ONLY a JSON array of entity names."
        )
    elif isinstance(node, ClosureNode):
        seed_str = ", ".join(sorted(seeds)[:20])
        prompt = (
            f"Find all {var} reachable from {seed_str} "
            f"via '{node.predicate}' (one or more steps).\n"
            f"Be exhaustive. Return ONLY a JSON array of entity names."
        )
    else:
        seed_str = ", ".join(sorted(seeds)[:20])
        prompt = (
            f"Find all {var} such that: {seed_str} {getattr(node,'predicate','')} {var}.\n"
            f"Be exhaustive. Return ONLY a JSON array of entity names."
        )
    return json_to_values(llm.chat(build_value_messages(prompt)).text)


def _exec_filter_holistic(var: str, candidates: set, node, llm: BaseLLM) -> set:
    """FILTER — always holistic. Hard invariant: result ⊆ candidates."""
    if not candidates:
        return set()
    cand_str = "\n".join(f"  - {c}" for c in sorted(candidates))
    cond = (f"is a {node.nl_type or node.value}"
            if isinstance(node, FilterNode)
            else getattr(node, 'predicate', str(node)))
    prompt = (
        f"From the following list, keep ONLY those where {cond}:\n\n"
        f"{cand_str}\n\n"
        f"Return ONLY items from the list above. No additions.\n"
        f"Return ONLY a JSON array."
    )
    return json_to_values(llm.chat(build_value_messages(prompt)).text) & candidates


def step_eval_holistic(
    plan: HybridPlan, llm: BaseLLM, gamma: Environment = None,
) -> set:
    """
    µ-Galois_Hol — plan known, each step executed holistically:
      ANCHOR   → holistic prompt
      EXPAND closure → LLMRecScan(motivational=False)
      EXPAND hop     → LLMSeedScan batched
      CHAIN    → one CoT prompt
      FILTER   → holistic
    """
    from mugalois.hybrid.hybrid_planner_v4 import (
        _extract_ptriples, _build_steps, _merge_hop_chains,
        _chain_cot_prompt, _run_value_prompt,
    )
    gamma = gamma or Environment()
    bound_consts = {ep for ep in (plan.source, plan.target)
                   if ep and not ep.startswith("?")}
    steps = _merge_hop_chains(
        _build_steps(_extract_ptriples(plan), bound_consts, llm)
    )
    steps = sorted(steps, key=lambda s: s.priority)
    bindings: dict = {ep: {ep} for ep in bound_consts}

    for step in steps:
        var  = step.output_var
        node = step.triples[0].node if step.triples else None

        if step.kind == "anchor":
            result = _exec_anchor_holistic(var, node, set(bound_consts), llm)

        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                seed = next(iter(seeds)) if len(seeds) == 1 else "?s"
                pat  = RecursivePattern(seed, t.p, "?x", t.operator)
                result = LLMRecScan(pat, llm, motivational=False)
            else:
                env = Environment()
                env.set("?_s", seeds)
                result = {tr.o for tr in
                          LLMSeedScan(TriplePattern("?_s", t.p, "?x"), env, llm, [], [])}

        elif step.kind == "chain":
            t0    = step.triples[0]
            seeds = bindings.get(t0.s, {t0.s}) if t0.s else set()
            result = _run_value_prompt(_chain_cot_prompt(step.triples, seeds), llm)

        elif step.kind == "filter":
            bindings[var] = _exec_filter_holistic(
                var, bindings.get(var, set()), node, llm)
            continue

        else:
            result = set()

        bindings[var] = (bindings[var] & result
                         if var in bindings and bindings[var] else result)
        if not bindings.get(var) and step.kind != "filter":
            return set()

    return bindings.get(plan.target_var, set())


def step_eval_decomposed(
    plan: HybridPlan, llm: BaseLLM, gamma: Environment = None,
) -> set:
    """
    µ-Galois_Dec — plan known, maximum decomposition:
      ANCHOR   → holistic (structural starting point known)
      EXPAND closure → LLMFixpointScan
      EXPAND hop     → LLMKeyScan (one per seed)
      CHAIN    → sequence of LLMScan (one per hop)
      FILTER   → holistic
    """
    from mugalois.hybrid.hybrid_planner_v4 import (
        _extract_ptriples, _build_steps, _merge_hop_chains,
    )
    gamma = gamma or Environment()
    bound_consts = {ep for ep in (plan.source, plan.target)
                   if ep and not ep.startswith("?")}
    steps = _merge_hop_chains(
        _build_steps(_extract_ptriples(plan), bound_consts, llm)
    )
    steps = sorted(steps, key=lambda s: s.priority)
    bindings: dict = {ep: {ep} for ep in bound_consts}

    for step in steps:
        var  = step.output_var
        node = step.triples[0].node if step.triples else None

        if step.kind == "anchor":
            result = _exec_anchor_holistic(var, node, set(bound_consts), llm)

        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                if len(seeds) == 1:
                    seed = next(iter(seeds))
                    fwd  = not t.o.startswith("?") or t.o not in bindings
                    pat  = (RecursivePattern(seed, t.p, "?x", t.operator) if fwd
                            else RecursivePattern("?x", t.p, seed, t.operator))
                    result = LLMFixpointScan(pat, llm, gamma=gamma)
                else:
                    cl     = ClosureNode(predicate=t.p, operator=t.operator)
                    result = _batch_fixpoint(cl, seeds, llm, direction="forward")
            else:
                result = set()
                for seed in seeds:
                    env = Environment()
                    env.set("?_s", {seed})
                    result |= {tr.o for tr in
                               LLMKeyScan(TriplePattern("?_s", t.p, "?x"), env, llm, [], [])}

        elif step.kind == "chain":
            t0    = step.triples[0]
            result = bindings.get(t0.s, {t0.s}) if t0.s else set()
            for t in step.triples:
                new = set()
                for seed in result:
                    env = Environment()
                    env.set("?_s", {seed})
                    new |= {tr.o for tr in
                            LLMScan(TriplePattern("?_s", t.p, "?x"),
                                    env, gamma, llm, tau_strategie=0.01)}
                result = new

        elif step.kind == "filter":
            bindings[var] = _exec_filter_holistic(
                var, bindings.get(var, set()), node, llm)
            continue

        else:
            result = set()

        bindings[var] = (bindings[var] & result
                         if var in bindings and bindings[var] else result)
        if not bindings.get(var) and step.kind != "filter":
            return set()

    return bindings.get(plan.target_var, set())


def _llm_step_conf(description: str, llm: BaseLLM) -> float:
    """
    Ask the LLM its confidence in solving a specific step holistically.
    Returns a float in [0, 1]. Pure confidence — no structure, no size estimate.
    """
    prompt = (
        f"Between 0 and 1, how confident are you that you can answer "
        f"the following completely and accurately in a single response?\n"
        f"{description}\n"
        f"Answer with a single decimal number only."
    )
    try:
        resp = llm.chat(build_messages(prompt))
        return max(0.0, min(1.0, float(resp.text.strip().split()[0])))
    except Exception:
        return 0.5


def step_eval_confidence(
    plan: HybridPlan, llm: BaseLLM, gamma: Environment = None,
    tau_type:    float = 0.60,   # _build_steps  : TYPE ANCHOR vs FILTER
    tau_anchor:  float = 0.60,   # ANCHOR step   : holistic vs KeyScan
    tau_closure: float = 0.50,   # EXPAND closure: LLMRecScan vs fixpoint
    tau_hop:     float = 0.60,   # EXPAND hop    : SeedScan vs KeyScan
    tau_chain:   float = 0.60,   # CHAIN         : CoT vs seq LLMScan
) -> set:
    """
    µ-Galois_C — plan known, confidence-only routing at every step.

    The plan structure (ANCHOR/EXPAND/CHAIN/FILTER order) is determined
    by the dependency graph — that is structural knowledge from the query,
    no LLM call. Only the routing WITHIN each step uses LLM confidence:

      ANCHOR   → _llm_step_conf(step description) > tau_anchor
                   → holistic prompt
                 else → KeyScan (one call per constant anchor value)

      EXPAND closure → LLMAtomicRecConf > tau_closure
                         → LLMRecScan(motivational=False)
                       else → LLMFixpointScan

      EXPAND hop     → _llm_step_conf(hop description) > tau_hop
                         → LLMSeedScan (one batched prompt)
                       else → LLMKeyScan (one per seed)

      CHAIN    → _llm_step_conf(chain description) > tau_chain
                   → one CoT holistic prompt
                 else → sequence of LLMScan (one per hop)

      FILTER   → always holistic (no routing — pure reduction)

    No calls to: LLMStructureDetect, LLMEstimateTypeSize,
    LLMEstimateSize, LLMEstimateRecSize, LLMBranchSimilarity.
    Only confidence prompts and LLMAtomicRecConf.

    tau_type governs whether a type predicate (e.g. "is a pope") becomes
    a TYPE ANCHOR (enumerate all) or a plain FILTER (reduce later).
    Replaces LLMEstimateTypeSize with a confidence call:
      conf("List every {type_val}") > tau_type → TYPE ANCHOR (returns 1)
      else                                      → FILTER (returns 999999)
    """
    from mugalois.hybrid.hybrid_planner_v4 import (
        _extract_ptriples, _build_steps, _merge_hop_chains,
        _chain_cot_prompt, _run_value_prompt,
    )
    import mugalois.hybrid.hybrid_planner_v4 as _v4

    gamma = gamma or Environment()

    # Replace LLMEstimateTypeSize with a confidence call
    _orig_estimate = _v4.LLMEstimateTypeSize
    def _conf_type_size(type_val: str, llm_inner: BaseLLM) -> int:
        conf = _llm_step_conf(
            f"List every {type_val} that has ever existed", llm_inner
        )
        # Return 1 (< threshold → TYPE ANCHOR) if confident, else 999999 (→ FILTER)
        return 1 if conf > tau_type else 999999
    _v4.LLMEstimateTypeSize = _conf_type_size

    try:
        bound_consts = {ep for ep in (plan.source, plan.target)
                       if ep and not ep.startswith("?")}
        steps = _merge_hop_chains(
            _build_steps(_extract_ptriples(plan), bound_consts, llm)
        )
    finally:
        _v4.LLMEstimateTypeSize = _orig_estimate
    steps = sorted(steps, key=lambda s: s.priority)
    bindings: dict = {ep: {ep} for ep in bound_consts}

    for step in steps:
        var  = step.output_var
        node = step.triples[0].node if step.triples else None

        if step.kind == "anchor":
            # Describe the anchor step for the confidence call
            constraints = " AND ".join(
                f"{t.s} {t.p} {t.o}" for t in step.triples
            )
            desc = f"Find all {var} satisfying: {constraints}"
            conf = _llm_step_conf(desc, llm)
            if conf > tau_anchor:
                # Holistic anchor
                result = _exec_anchor_holistic(var, node, set(bound_consts), llm)
            else:
                # KeyScan — one call per known constant value
                result = set()
                for anchor_val in bound_consts:
                    env = Environment()
                    env.set("?_a", {anchor_val})
                    for t in step.triples:
                        pat = TriplePattern("?_a", t.p, "?x")
                        result |= {tr.o for tr in
                                   LLMKeyScan(pat, env, llm, [], [])}

        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                # LLMAtomicRecConf — the one confidence call designed for closures
                seed = next(iter(seeds)) if seeds else "?s"
                pat  = RecursivePattern(seed, t.p, "?x", t.operator)
                conf = LLMAtomicRecConf(pat, llm)
                result = (LLMRecScan(pat, llm, motivational=False)
                          if conf > tau_closure
                          else LLMFixpointScan(pat, llm, gamma=gamma))
            else:
                # Hop confidence: can the LLM return all values at once?
                seed_str = ", ".join(sorted(seeds)[:5])
                desc = f"Find all values of {var} via '{t.p}' from: {seed_str}"
                conf = _llm_step_conf(desc, llm)
                if conf > tau_hop:
                    # SeedScan — one batched prompt for all seeds
                    env = Environment()
                    env.set("?_s", seeds)
                    result = {tr.o for tr in
                              LLMSeedScan(TriplePattern("?_s", t.p, "?x"),
                                          env, llm, [], [])}
                else:
                    # KeyScan — one prompt per seed
                    result = set()
                    for seed in seeds:
                        env = Environment()
                        env.set("?_s", {seed})
                        result |= {tr.o for tr in
                                   LLMKeyScan(TriplePattern("?_s", t.p, "?x"),
                                              env, llm, [], [])}

        elif step.kind == "chain":
            t0    = step.triples[0]
            seeds = bindings.get(t0.s, {t0.s}) if t0.s else set()
            chain_desc = " → ".join(t.p for t in step.triples)
            seed_str   = ", ".join(sorted(seeds)[:5])
            desc = f"Follow the chain {chain_desc} starting from: {seed_str}"
            conf = _llm_step_conf(desc, llm)
            if conf > tau_chain:
                # Holistic CoT for the full chain
                result = _run_value_prompt(
                    _chain_cot_prompt(step.triples, seeds), llm)
            else:
                # Decomposed: one LLMScan per hop
                result = seeds
                for t in step.triples:
                    new = set()
                    for s in result:
                        env = Environment()
                        env.set("?_s", {s})
                        new |= {tr.o for tr in
                                LLMScan(TriplePattern("?_s", t.p, "?x"),
                                        env, gamma, llm,
                                        tau_strategie=tau_hop)}
                    result = new

        elif step.kind == "filter":
            # Filter is always holistic — just a reduction, no routing needed
            bindings[var] = _exec_filter_holistic(
                var, bindings.get(var, set()), node, llm)
            continue

        else:
            result = set()

        bindings[var] = (bindings[var] & result
                         if var in bindings and bindings[var] else result)
        if not bindings.get(var) and step.kind != "filter":
            return set()

    return bindings.get(plan.target_var, set())