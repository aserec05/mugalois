"""
HybridPlannerV4 — Dependency-graph planner for T7 hybrid path queries.

Philosophy
----------
Inspired by Waveguide (Yakovets et al., SIGMOD 2016): the evaluation order
of a hybrid path expression is NOT fixed — it is determined by the structure
of available bindings (seeds) and the cognitive cost of each segment.

The planner builds a variable dependency graph from the plan's triples
(path nodes + where_triples), then schedules execution in topological order:

  ANCHOR  : resolve a variable from a constant anchor + ALL its co-constraints
            in a single holistic CoT prompt.
            Co-constraints include both path triples AND where_triples for that var.
            e.g. "?pm successorOf Thatcher AND ?pm memberOf Conservative Party"
                 → one prompt, return all Conservative successors of Thatcher

  EXPAND  : produce a new variable binding from an already-bound variable.
            Dispatches to:
              - MuGaloisRec     (ClosureNode: full T5 decision tree)
              - LLMSeedScan     (HopNode: batched seed prompt)

  CHAIN   : sequence of consecutive simple hops → single CoT prompt.
            Avoids error accumulation across N sequential EXPAND calls.
            e.g. ?film → directedBy → wasBornIn  (2 hops, 1 prompt)

  FILTER  : reduce an already-bound variable using residual constraints.
            Hard invariant: result ⊆ candidates (never adds entities).
            Applied AFTER ANCHOR (safety) and AFTER EXPAND (where_triples).

Execution order: ANCHOR (0) → EXPAND/CHAIN (1) → FILTER (2) → JOIN (3)


"""
from __future__ import annotations
import re

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from mugalois.hybrid.hybrid_plan import (
    HybridPlan, HopNode, ClosureNode, FilterNode, ChoiceNode,
)
from mugalois.core.types import TriplePattern, RecursivePattern, Environment
from mugalois.core.prompts import build_value_messages
from mugalois.core.parser import json_to_values
from mugalois.rec.mugalois_rec import MuGaloisRec
from mugalois.scans.seed_scan import LLMSeedScan
from mugalois.llm.llm_client import BaseLLM

SEED_BATCH_SIZE = 30
MAX_VALS_IN_PROMPT = 20   # max entity names shown in a single prompt


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PTriple:
    """
    A triple (s, predicate, o) extracted from a HybridPlan node or where_triple.
    Retains closure metadata and source tag for scheduling decisions.

    level_hint: optional instruction on the granularity expected for the output
    variable. E.g. "city" for wasBornIn when followed by isAdministrativelyLocatedIn*.
    Injected into the EXPAND hop prompt to steer the LLM toward the right level.
    """
    s:          str
    p:          str
    o:          str
    is_closure: bool = False
    operator:   str  = "+"
    source:     str  = "path"   # "path" | "where"


    @property
    def variables(self) -> Set[str]:
        return {v for v in (self.s, self.o) if _var(v)}

    def __repr__(self):
        op = self.operator if self.is_closure else ""
        return f"({self.s}, {self.p}{op}, {self.o})[{self.source}]"


@dataclass
class ExecStep:
    """One scheduled execution step."""
    kind:       str            # "anchor" | "expand" | "chain" | "filter" | "join"
    output_var: str
    triples:    List[PTriple]
    priority:   int = 0        # 0=ANCHOR 1=EXPAND/CHAIN 2=FILTER 3=JOIN

    def describe(self) -> str:
        parts = " ∧ ".join(f"({t.s},{t.p},{t.o})" for t in self.triples)
        return f"{self.kind.upper()}[{parts}] → {self.output_var}"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _var(v: str) -> bool:
    return isinstance(v, str) and v.startswith("?")


def _humanize(v: str) -> str:
    return v.replace("_", " ") if not _var(v) else v


def _fmt_vals(vals: Set[str], max_n: int = MAX_VALS_IN_PROMPT) -> str:
    lst = sorted(vals)[:max_n]
    suffix = f" (+{len(vals)-max_n} more)" if len(vals) > max_n else ""
    return ", ".join(_humanize(v) for v in lst) + suffix


def _run_value_prompt(prompt: str, llm: BaseLLM) -> Set[str]:
    resp = llm.chat(build_value_messages(prompt))
    return json_to_values(resp.text)


# Maximum number of instances for a type to be considered "selective enough"
# to use as a TYPE ANCHOR seed. Above this threshold, enumerating all instances
# is worse than starting from a more specific path anchor.
# Calibrated against MuGaloisRec's CARD_MAX=15 for recursive paths.
TYPE_SELECTIVITY_THRESHOLD = 500


# Session-level cache for type size estimates.
# Ensures the same type always produces the same scheduling decision
# within a single evaluation run — avoids plan instability across N_RUNS.
_type_size_cache: Dict[str, int] = {}


def LLMEstimateTypeSize(type_value: str, llm: BaseLLM) -> int:
    """
    Ask the LLM to estimate the total number of entities of a given type.
    Returns an integer estimate (or a large number on failure/uncertainty).

    Cached per session: same type → same estimate → same plan across all
    N_RUNS repetitions. Without caching, stochastic LLM responses cause
    different scheduling decisions between runs (TYPE ANCHOR vs FILTER),
    making F1 variance meaningless.

    Used to decide whether a where_triple "?x is a [type]" is selective
    enough to serve as a TYPE ANCHOR (enumerate all, then filter) vs.
    a plain FILTER (reduce an already-bound set).

    Examples:
      "pope"     → ~266   → ≤ 500 → selective → TYPE ANCHOR
      "Film"     → 999999 → > 500 → generic   → FILTER only
      "president"→ ~50    → ≤ 500 → selective → TYPE ANCHOR
    """
    global _type_size_cache
    if type_value in _type_size_cache:
        return _type_size_cache[type_value]

    # Reformulate to count INSTANCES, not the entity type itself.
    # "How many Conservative Party members have there been?" (not "how many Conservative Partys")
    # Detects membership phrases to reformulate appropriately.
    type_lower = type_value.lower()
    if any(w in type_lower for w in ["party", "group", "organisation", "organization"]):
        question = (
            f"Approximately how many individual members of the {type_value} "
            f"have there been in total (throughout history)?"
        )
    else:
        question = (
            f"Approximately how many {type_value}s have existed in total "
            f"(worldwide, throughout all of history)?"
        )
    prompt = (
        f"{question}\n"
        f"Answer with a single integer only. No explanation.\n"
        f"If you are unsure or the number is very large, answer 999999."
    )
    from mugalois.core.prompts import build_messages
    resp = llm.chat(build_messages(prompt))
    try:
        m = re.search(r"\d+", resp.text.strip().replace(",", "").replace(".", ""))
        est = int(m.group()) if m else 999999
    except Exception:
        est = 999999

    _type_size_cache[type_value] = est
    return est


# ══════════════════════════════════════════════════════════════════════════════
# Triple extraction from HybridPlan
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ptriples(plan: HybridPlan) -> List[PTriple]:
    """
    Convert plan.nodes + plan.where_triples into a flat list of PTriple.
    Falls back to sequential variable names (?_v0, ?_v1, ...) when node.s/o
    are absent — mirrors the convention in hybrid_plan.py from_query_json.
    """
    triples: List[PTriple] = []
    n = len(plan.nodes)
    fallback_s = [plan.source] + [f"?_v{i}" for i in range(n)]
    fallback_o = [f"?_v{i}" for i in range(n)] + [plan.target]

    for i, node in enumerate(plan.nodes):
        s = getattr(node, "s", "") or fallback_s[i]
        o = getattr(node, "o", "") or fallback_o[i]
        p = getattr(node, "predicate", "")

        if isinstance(node, ClosureNode):
            triples.append(PTriple(s=s, p=p, o=o,
                                   is_closure=True, operator=node.operator,
                                   source="path"))
        elif isinstance(node, HopNode):
            triples.append(PTriple(s=s, p=p, o=o,
                                   is_closure=False, source="path"))
        elif isinstance(node, FilterNode):
            # FilterNode becomes a WHERE-style constraint on its variable
            triples.append(PTriple(s=s, p=node.predicate, o=node.value,
                                   is_closure=False, source="where"))
        elif isinstance(node, ChoiceNode):
            # Treat as a compound-label hop (T6 routing handled by MuGaloisRec
            # if needed; for T7 planning purposes, we treat as simple hop)
            choice_p = " | ".join(node.branches)
            triples.append(PTriple(s=s, p=choice_p, o=o,
                                   is_closure=False, source="path"))

    for wt in getattr(plan, "where_triples", []) or []:
        triples.append(PTriple(s=wt.s, p=wt.p, o=wt.o,
                               is_closure=False, source="where"))

    return triples


# ══════════════════════════════════════════════════════════════════════════════
# Scheduling: variable dependency graph → ordered ExecSteps
# ══════════════════════════════════════════════════════════════════════════════

# Type/role predicates that trigger the "enumerate all [type], then filter"
# CoT strategy in ANCHOR prompts.
TYPE_PREDS = {
    "is a", "rdf:type", "type", "isA",
    "wasPresidentOf", "is a member of", "memberOf", "hasRole",
}



def _build_steps(ptriples: List[PTriple],
                 bound_consts: Set[str],
                 llm: BaseLLM) -> List[ExecStep]:
    """
    Build an ordered list of ExecStep by iteratively resolving variables
    whose dependencies are satisfied.

    A variable V is schedulable when at least one path triple involving V
    has its OTHER side already bound (constant or previously scheduled var).

    Classification:
      - constant on other side              → ANCHOR (priority 0)
      - previously scheduled var on other side → EXPAND  (priority 1)
      - no satisfied dependency yet        → deferred (JOIN, priority 3)

    WHERE triples for V are always bundled into V's ANCHOR step so that
    the CoT prompt knows all constraints upfront. A safety FILTER step
    is also added afterward for hard constraints the ANCHOR might miss.
    """
    path_triples  = [t for t in ptriples if t.source == "path"]
    where_triples = [t for t in ptriples if t.source == "where"]

    # Index: variable → triples that involve it
    by_var: Dict[str, List[PTriple]] = {}
    for t in ptriples:
        for v in t.variables:
            by_var.setdefault(v, []).append(t)

    all_vars      = {v for t in ptriples for v in t.variables}
    scheduled     = set(bound_consts)   # grows as steps are planned
    scheduled_vars: Set[str] = set()
    steps: List[ExecStep] = []
    anchored_vars: Set[str] = set()     # vars resolved via ANCHOR

    # Iterative topological scheduling
    for _ in range(len(all_vars) + 5):
        unscheduled = all_vars - scheduled_vars
        if not unscheduled:
            break

        progress = False
        for var in sorted(unscheduled):
            var_path   = [t for t in by_var.get(var, []) if t.source == "path"]
            var_where  = [t for t in by_var.get(var, []) if t.source == "where"]

            # A where_triple with a type predicate (is a, rdf:type, ...) and a
            # constant value is a TYPE ANCHOR — the most selective constraint.
            # Priority -1: higher than path-based anchors (prio=0).
            # e.g. "?pope is a pope" → enumerate all popes, then filter by
            # wasBornIn/isLocatedIn — not: find all Italian places, find who
            # was born there, then filter for popes.
            #
            # Selectivity check: ask the LLM how many instances of this type
            # exist. If too many (> TYPE_SELECTIVITY_THRESHOLD), the type is
            # too generic to enumerate — use as FILTER only, not ANCHOR.
            # This replaces any hardcoded list of "non-selective" types.
            type_anchor_triples = []
            for t in var_where:
                if not (t.p in TYPE_PREDS):
                    continue
                if t.s == var and not _var(t.o):
                    type_val = t.o
                elif t.o == var and not _var(t.s):
                    type_val = t.s
                else:
                    continue
                est = LLMEstimateTypeSize(type_val, llm)
                if est <= TYPE_SELECTIVITY_THRESHOLD:
                    type_anchor_triples.append(t)
                # if est > threshold: skip — will be handled as FILTER

            if type_anchor_triples:
                # TYPE ANCHOR: highest priority (prio=-1).
                # When a type anchor exists, it becomes the sole starting point.
                # Path triples for this var are NOT bundled here — they will be
                # evaluated as EXPAND steps once this var is bound, preserving
                # the correct execution order:
                #   TYPE ANCHOR ?pope → EXPAND ?place (wasBornIn) → FILTER Italy
                # rather than creating two competing ANCHORs.
                kind, prio = "anchor", -1
                anchored_vars.add(var)
                step_triples = type_anchor_triples  # type constraint only
                steps.append(ExecStep(kind=kind, output_var=var,
                                      triples=step_triples, priority=prio))
                scheduled_vars.add(var)
                scheduled.add(var)
                progress = True
                continue

            # Find path triples where the other side is already bound
            satisfying = [
                t for t in var_path
                if (t.s == var and (not _var(t.o) or t.o in scheduled))
                or (t.o == var and (not _var(t.s) or t.s in scheduled))
            ]

            if not satisfying:
                continue  # dependencies not yet met

            # Is there a direct constant anchor in path triples?
            has_const = any(
                (t.s == var and not _var(t.o))
                or (t.o == var and not _var(t.s))
                for t in satisfying
            )

            if has_const:
                kind, prio = "anchor", 0
                anchored_vars.add(var)
                # Bundle ALL constraints for this var (path + where)
                step_triples = satisfying + var_where
            else:
                kind, prio = "expand", 1
                step_triples = satisfying   # where_triples handled by FILTER

            steps.append(ExecStep(kind=kind, output_var=var,
                                  triples=step_triples, priority=prio))
            scheduled_vars.add(var)
            scheduled.add(var)
            progress = True

        if not progress:
            # Remaining vars: schedule as JOIN (unresolved dependency)
            for var in sorted(all_vars - scheduled_vars):
                var_triples = by_var.get(var, [])
                steps.append(ExecStep(kind="join", output_var=var,
                                      triples=var_triples, priority=3))
                scheduled_vars.add(var)
                scheduled.add(var)
            break

    # When a TYPE ANCHOR exists, remove competing path-based ANCHORs for
    # variables that can be reached via EXPAND from the TYPE ANCHOR var.
    # e.g. Q1: ?pope TYPE ANCHOR → ?place should be EXPAND(wasBornIn) not ANCHOR(Italy)
    # because Italy anchor produces Italian places without filtering for popes.
    type_anchor_vars = {s.output_var for s in steps if s.kind == "anchor" and s.priority == -1}
    if type_anchor_vars:
        # Find path-based ANCHORs (prio=0) that are reachable from a type anchor var
        # via a single EXPAND hop — convert them to FILTER instead
        cleaned_steps = []
        for s in steps:
            if s.kind == "anchor" and s.priority == 0:
                # Check if this var is the object/subject of a hop from a type anchor var
                reachable = any(
                    (t.s in type_anchor_vars and t.o == s.output_var)
                    or (t.o in type_anchor_vars and t.s == s.output_var)
                    for t in path_triples
                    if not t.is_closure
                )
                if reachable:
                    # Convert to FILTER — will be applied after EXPAND produces the var
                    cleaned_steps.append(ExecStep(
                        kind="filter", output_var=s.output_var,
                        triples=s.triples, priority=2
                    ))
                    continue
            cleaned_steps.append(s)
        steps = cleaned_steps

    # Safety FILTER pass
    filter_added: Set[str] = set()
    for var in scheduled_vars:
        var_where = [t for t in where_triples if var in t.variables]
        if var_where and var not in filter_added:
            steps.append(ExecStep(kind="filter", output_var=var,
                                  triples=var_where, priority=2))
            filter_added.add(var)

    return steps


# ══════════════════════════════════════════════════════════════════════════════
# Chain merging: adjacent EXPAND hops → single CHAIN step
# ══════════════════════════════════════════════════════════════════════════════

def _merge_hop_chains(steps: List[ExecStep]) -> List[ExecStep]:
    """
    Merge consecutive EXPAND steps on simple (non-closure) hops into a
    single CHAIN step evaluated by one CoT prompt.

    Merging conditions (ALL must hold):
      1. steps[i] is EXPAND (single hop, not closure)
      2. steps[i+1] is EXPAND (single hop, not closure)
      3. steps[i].output_var == steps[i+1].triples[0].s  (sequential dep.)
      4. No intermediate variable in the chain is used as seed by a
         SUBSEQUENT closure EXPAND — if it is, merging would collapse
         diversity and starve the closure of seeds.

    Example of why condition 4 matters (q8):
      EXPAND ?director = directedBy from ?film       (hop)
      EXPAND ?place    = wasBornIn  from ?director   (hop)
      EXPAND ?country  = isAdministrativelyLocatedIn+ from ?place  (closure!)
      → ?place seeds the closure → do NOT merge directedBy+wasBornIn into CHAIN
        (CHAIN returns only 1 collapsed ?place; closure then finds little)
      → keep as 2 separate EXPANDs so ?place has full cardinality for closure

    Example where merge IS correct (no downstream closure):
      EXPAND ?city    = wasBornIn from ?person   (hop)
      EXPAND ?country = isIn      from ?city     (hop, no closure after)
      → merge into CHAIN ✓
    """
    # Pre-compute: which variables are seeds of a closure EXPAND?
    closure_seeds: Set[str] = set()
    for step in steps:
        if (step.kind == "expand"
                and len(step.triples) == 1
                and step.triples[0].is_closure):
            t = step.triples[0]
            # The seed is the bound side — whichever is a variable with prior binding
            # In forward direction: t.s is the seed variable
            # In backward direction: t.o is the seed variable
            # At merge time we don't have bindings, so mark both variables
            for v in t.variables:
                closure_seeds.add(v)

    merged: List[ExecStep] = []
    skip: Set[int] = set()

    for i, step in enumerate(steps):
        if i in skip:
            continue

        if (step.kind == "expand"
                and len(step.triples) == 1
                and not step.triples[0].is_closure):

            chain_triples = [step.triples[0]]
            chain_out = step.output_var
            j = i + 1

            while j < len(steps) and j not in skip:
                nxt = steps[j]
                next_out = nxt.output_var  # what chain_out would become
                if (nxt.kind == "expand"
                        and len(nxt.triples) == 1
                        and not nxt.triples[0].is_closure
                        and nxt.triples[0].s == chain_out
                        # Condition 4: don't absorb if the NEW output would be
                        # a seed for a downstream closure (preserves cardinality)
                        and next_out not in closure_seeds):
                    chain_triples.append(nxt.triples[0])
                    chain_out = next_out
                    skip.add(j)
                    j += 1
                else:
                    break

            if len(chain_triples) >= 2:
                merged.append(ExecStep(kind="chain", output_var=chain_out,
                                       triples=chain_triples,
                                       priority=step.priority))
                continue

        merged.append(step)

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def _render_constraint(t: PTriple, var: str,
                        bindings: Dict[str, Set[str]]) -> str:
    """
    Render one PTriple as a natural-language constraint on `var`,
    substituting already-bound variable values where available.
    """
    s, p, o = t.s, t.p, t.o

    if _var(s) and s != var and s in bindings:
        s_str = f"one of: {_fmt_vals(bindings[s])}"
    else:
        s_str = _humanize(s)

    if _var(o) and o != var and o in bindings:
        o_str = f"one of: {_fmt_vals(bindings[o])}"
    else:
        o_str = _humanize(o)

    if t.s == var:
        return f"{var} {p} {o_str}"
    else:
        return f"{s_str} {p} {var}"


def _anchor_prompt(var: str,
                   triples: List[PTriple],
                   bindings: Dict[str, Set[str]]) -> str:
    """
    Holistic CoT prompt for ANCHOR.

    When a type/role predicate is present (e.g. "is a pope",
    "wasPresidentOf France"), uses a two-step CoT:
      Step 1: Enumerate ALL entities of that type you know.
      Step 2: Keep ONLY those satisfying all other constraints.

    Otherwise, lists all constraints and asks for exhaustive retrieval.
    Both variants include anti-saliency bias instructions
    ("include lesser-known ones too").
    """
    type_triples = [
        t for t in triples
        if t.p in TYPE_PREDS and not _var(t.o if t.s == var else t.s)
    ]
    other_triples = [t for t in triples if t not in type_triples]

    # Only include constraints where the other variable is already bound —
    # an unbound variable in a constraint is vacuous ("?pope wasBornIn ?place"
    # when ?place is unknown gives no information to the LLM).
    def _other_side_bound(t: PTriple) -> bool:
        other = t.o if t.s == var else t.s
        return not _var(other) or other in bindings

    actionable_constraints = [t for t in other_triples if _other_side_bound(t)]

    if type_triples:
        def _build_type_desc(t: PTriple, var: str) -> str:
            """
            Build a natural-language description of what entities to enumerate,
            combining predicate semantics with the constant value — so that
            "Step 1: enumerate all [type_desc]s" reads naturally.

            "is a pope"                        → "pope"
            "is a member of Conservative Party"→ "member of the Conservative Party"
            "wasPresidentOf France"            → "President of France"
            "memberOf Labour Party"            → "member of the Labour Party"
            """
            val  = _humanize(t.o if t.s == var else t.s)
            pred = t.p.strip()
            pred_l = pred.lower()

            if pred_l in ("is a", "rdf:type", "type", "isa", "hasrole", "role"):
                # Direct type: value IS the type label
                return val

            elif "member" in pred_l:
                # Membership predicate: "member of the Conservative Party"
                return f"member of the {val}"

            elif pred_l.startswith("was") and "of" in pred_l:
                # Role predicate: "wasPresidentOf France" → "President of France"
                # Extract role name between "was" and "Of"
                role_raw = re.sub(r"(?i)^was(.+?)of$", r"\1", pred).strip()
                # CamelCase → "President"
                role = re.sub(r"([A-Z])", r" \1", role_raw).strip().title()
                return f"{role} of {val}"

            else:
                # Generic: use the full predicate + value as-is
                return f"{pred} {val}"

        type_desc = " and ".join(
            _build_type_desc(t, var) for t in type_triples
        )
        if actionable_constraints:
            constraint_lines = "\n".join(
                f"  - {_render_constraint(t, var, bindings)}"
                for t in actionable_constraints
            )
            step2 = (
                f"Step 2: From that complete list, keep ONLY those satisfying "
                f"ALL of the following:\n"
                f"{constraint_lines}\n\n"
            )
        else:
            # No bound constraints yet — just enumerate exhaustively
            step2 = (
                f"Step 2: Return all of them — every entity that is a {type_desc}.\n\n"
            )

        return (
            f"Think step by step:\n"
            f"Step 1: In your mind, enumerate every entity that is a {type_desc} — "
            f"including the lesser-known ones, not just the famous ones.\n"
            f"{step2}"
            f"Be exhaustive. Do not stop at the obvious ones.\n"
            f"Return ONLY a JSON array of entity names."
        )
    else:
        constraint_lines = "\n".join(
            f"  - {_render_constraint(t, var, bindings)}"
            for t in triples
        )
        return (
            f"Find ALL entities for {var} satisfying every condition below:\n"
            f"{constraint_lines}\n\n"
            f"Think step by step. Include all you are confident about, "
            f"even the less well-known ones. Do not guess.\n"
            f"Return ONLY a JSON array of entity names."
        )


def _chain_cot_prompt(chain: List[PTriple],
                       seeds: Set[str]) -> str:
    """
    Chain-of-Thought prompt for a sequence of simple hops.
    The LLM follows each relation in order and returns ONLY the final result.

    e.g. for [directedBy, wasBornIn] with seeds = {Quantum_of_Solace, Skyfall}:
      Step 1: From each film, find the directedBy value.
      Step 2: From each director found in step 1, find the wasBornIn value.
      Return ONLY step 2 results.
    """
    seed_str = _fmt_vals(seeds)
    steps_desc = []
    cur_label = f"each of: {seed_str}"

    for i, t in enumerate(chain):
        steps_desc.append(
            f"  Step {i+1}: From {cur_label}, find the '{_humanize(t.p)}' value(s)."
        )
        cur_label = f"the results of step {i+1}"

    steps_text = "\n".join(steps_desc)
    final_p    = _humanize(chain[-1].p)

    return (
        f"Follow this chain of relations step by step, keeping "
        f"only intermediate results in your working memory:\n"
        f"{steps_text}\n\n"
        f"Return ONLY the results of the LAST step ('{final_p}' values). "
        f"Do not include values from earlier steps.\n"
        f"Be exhaustive and factual. Do not guess.\n"
        f"Return ONLY a JSON array of entity names."
    )


def _filter_prompt(var: str,
                   candidates: Set[str],
                   triples: List[PTriple],
                   bindings: Dict[str, Set[str]],
                   expand_triples: List[PTriple] = None) -> str:
    """
    FILTER prompt: reduce candidates to those satisfying ALL constraints.
    Hard invariant enforced post-call: result ⊆ candidates.

    expand_triples: the EXPAND/CHAIN triples that produced `var`.
    Injected as extra conditions so the LLM understands provenance.
    e.g. for ?sequel produced by hasSequel+ from {TDK, Inception}:
      "?sequel must be a direct sequel of one of: The Dark Knight, Inception"
    Without this, the LLM only sees where_triples (is a Film, directedBy Nolan)
    and may keep entities that satisfy those but are not actual sequels.
    """
    cand_str = "\n".join(f"  - {_humanize(c)}" for c in sorted(candidates))

    all_conditions: List[str] = []

    # 1. Provenance constraints from the EXPAND step (most important)
    for t in (expand_triples or []):
        other_var = t.s if t.o == var else t.o
        other_vals = bindings.get(other_var, set()) if _var(other_var) else {other_var}
        if other_vals:
            vals_str = _fmt_vals(other_vals)
            op_str   = t.operator if t.is_closure else ""
            if t.o == var:   # source → predicate → var
                src_str = _humanize(other_var) if not _var(other_var) else f"one of: {vals_str}"
                all_conditions.append(
                    f"  - {var} must be reachable via '{t.p}{op_str}' "
                    f"from {src_str}"
                )
            else:             # var → predicate → target
                tgt_str = _humanize(other_var) if not _var(other_var) else f"one of: {vals_str}"
                all_conditions.append(
                    f"  - {var} {t.p}{op_str} {tgt_str}"
                )

    # 2. WHERE-triple constraints
    for t in triples:
        all_conditions.append(f"  - {_render_constraint(t, var, bindings)}")

    conditions_str = "\n".join(all_conditions)

    return (
        f"From the following candidates, keep ONLY those satisfying "
        f"ALL conditions:\n\n"
        f"Conditions:\n{conditions_str}\n\n"
        f"Candidates:\n{cand_str}\n\n"
        f"Return ONLY items from the candidate list. "
        f"Do not add or rename items. No explanations.\n"
        f"Return ONLY a JSON array."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step executors
# ══════════════════════════════════════════════════════════════════════════════

def _exec_anchor(step: ExecStep,
                 bindings: Dict[str, Set[str]],
                 llm: BaseLLM,
                 verbose: bool) -> Set[str]:
    prompt = _anchor_prompt(step.output_var, step.triples, bindings)
    if verbose:
        print(f"\n  [ANCHOR] {step.output_var}")
        print(f"  {prompt[:500]}")
    result = _run_value_prompt(prompt, llm)
    if verbose:
        print(f"  → {len(result)}: {sorted(result)[:5]}")
    return result



def _holistic_closure(t: PTriple, seeds: Set[str], llm: BaseLLM,
                      verbose: bool) -> Set[str]:
    """
    Single holistic prompt for a multi-seed transitive closure.

    Generic CoT strategy — no domain hardcoding:
      Ask the LLM to write, for each seed, the explicit chain of entities
      reachable via the predicate, then collect all names in those chains.

    The chain format (A → B → C → ...) is domain-agnostic:
      - isAdministrativelyLocatedIn: city → county → state → country
      - hasSuccessor:                PM_1 → PM_2 → PM_3
      - hasChild:                    parent → child → grandchild

    The LLM determines what the chain looks like for the given predicate —
    no geographic or domain assumptions are hardcoded here.

    operator="*" (reflexive): seeds included in result.
    operator="+" (irreflexive): seeds excluded from result.
    """
    seed_list = sorted(seeds)
    seeds_fmt = "\n".join(f"  - {_humanize(s)}" for s in seed_list)
    op        = t.operator
    pred      = _humanize(t.p)

    include_seeds = (
        "Include each starting entity itself in the final list."
        if op == "*" else
        "Do NOT include the starting entities themselves in the final list."
    )

    prompt = (
        f"For each of the following entities, write the complete chain of "
        f"entities reachable via the relation '{pred}', "
        f"step by step until no further entity exists:\n"
        f"{seeds_fmt}\n\n"
        f"Format each chain as:\n"
        f"  [start] → [next] → [next] → ... → [end]\n\n"
        f"Rules:\n"
        f"  - Follow the relation '{pred}' faithfully and precisely.\n"
        f"  - Do not skip any intermediate step in the chain.\n"
        f"  - Do not add entities that are not reachable via '{pred}'.\n"
        f"  - Stop when no further '{pred}' relation applies.\n"
        f"  - {include_seeds}\n\n"
        f"After writing all chains, return a flat JSON array containing "
        f"every entity name that appears in any chain "
        f"(no duplicates, no chain notation).\n"
        f"Return ONLY the JSON array."
    )

    if verbose:
        print(f"  [holistic_closure] pred={pred!r} seeds={len(seeds)}")
        print(f"  {prompt[:400]}")

    result = _run_value_prompt(prompt, llm)

    if op == "+":
        result -= seeds

    return result


def _exec_expand(step: ExecStep,
                 bindings: Dict[str, Set[str]],
                 llm: BaseLLM,
                 gamma: Environment,
                 verbose: bool,
                 rec_fn=None,
                 hop_fn=None) -> Set[str]:
    """
    EXPAND: produce output_var from a bound source.

    Routing:
      ClosureNode → rec_fn (default: MuGaloisRec, T5 decision tree)
      HopNode     → hop_fn (default: LLMSeedScan, batched SeedCrank)

    rec_fn(pattern, llm, gamma) -> Set[str]
    hop_fn(triple, seeds, forward, llm) -> Set[str]
    Injectable for confidence-only or structure-only variants.
    """
    t = step.triples[0]

    # Determine direction and seeds
    if not _var(t.s):
        seeds, forward = {t.s}, True
    elif t.s in bindings and bindings[t.s]:
        seeds, forward = bindings[t.s], True
    elif not _var(t.o):
        seeds, forward = {t.o}, False
    elif t.o in bindings and bindings[t.o]:
        seeds, forward = bindings[t.o], False
    else:
        if verbose:
            print(f"  [EXPAND] {t} — no seeds, skipping")
        return set()

    if verbose:
        direction_str = "fwd" if forward else "bwd"
        print(f"\n  [EXPAND] ({t.s},{t.p},{t.o}) {direction_str} "
              f"seeds={len(seeds)}: {sorted(seeds)[:3]}")

    # Closure dispatch
    if t.is_closure:
        direction_label = "forward" if forward else "backward"
        closure_node = ClosureNode(
            predicate=t.p, operator=t.operator,
            inverse=(not forward), s=t.s, o=t.o,
        )

        if len(seeds) == 1:
            # Single seed → full MuGaloisRec decision tree (T5)
            seed = next(iter(seeds))
            pat  = (RecursivePattern(seed, t.p, "?x", t.operator) if forward
                    else RecursivePattern("?x", t.p, seed, t.operator))
            _rec = rec_fn if rec_fn is not None else MuGaloisRec
            try:
                result = _rec(pat, llm, gamma=gamma, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  [EXPAND closure] rec_fn error: {e}")
                result = set()
        else:
            # Multi-seed → bounded holistic closure prompt.
            #
            # _batch_fixpoint was tried but has two failure modes:
            #   1. KeyScan per entity causes runaway DAG traversal
            #      (Germany→Europe→EU→Brussels→Belgium→Flanders→...)
            #   2. Depth=50 never stops on open-ended admin hierarchies
            #
            # Instead: one holistic LLM call that asks for the
            # complete administrative hierarchy of each seed up to
            # the sovereign country level — bounded by construction.
            result = _holistic_closure(t, seeds, llm, verbose)

        if verbose:
            strategy = "MuGaloisRec" if len(seeds) == 1 else "holistic_closure"
            print(f"  [EXPAND closure] {t.p}{t.operator} ({strategy}) "
                  f"seeds={len(seeds)} -> {len(result)}: {sorted(result)[:4]}")
        return result

    # Simple hop → LLMSeedScan (batched SeedCrank)
    # Parse level hint from predicate syntax: wasBornIn(city) → hint="city"
    # This keeps the hint in the query expression, not in metadata fields.
    _hint_match = re.match(r"^(.+?)\((.+?)\)$", t.p)
    clean_pred  = _hint_match.group(1) if _hint_match else t.p
    level_hint  = _hint_match.group(2) if _hint_match else ""
    lookahead = (
        f"Return the {level_hint} level only — not a higher-level region or country."
        if level_hint else ""
    )

    seed_list = list(seeds)
    result = set()
    if hop_fn is not None:
        # Custom hop executor (e.g. confidence-only _C variant)
        result = hop_fn(t, seeds, forward, llm, clean_pred, lookahead)
    else:
        for i in range(0, len(seed_list), SEED_BATCH_SIZE):
            batch = set(seed_list[i:i + SEED_BATCH_SIZE])
            env = Environment()
            if forward:
                var_name = t.s if _var(t.s) else "?_s"
                env.set(var_name, batch)
                triples_out = LLMSeedScan(
                    TriplePattern(var_name, clean_pred, "?x"), env, llm,
                    inject_conds=[], post_filter_conds=[],
                    lookahead=lookahead,
                )
                result |= {x.o for x in triples_out}
            else:
                var_name = t.o if _var(t.o) else "?_o"
                env.set(var_name, batch)
                triples_out = LLMSeedScan(
                    TriplePattern("?x", clean_pred, var_name), env, llm,
                    inject_conds=[], post_filter_conds=[],
                    lookahead=lookahead,
                )
                result |= {x.s for x in triples_out}

    if verbose:
        hint_str = f"({level_hint})" if level_hint else ""
        print(f"  [EXPAND hop] {clean_pred}{hint_str} {'fwd' if forward else 'bwd'} "
              f"seeds={len(seeds)} → {len(result)}: {sorted(result)[:4]}")
    return result


def _exec_chain(step: ExecStep,
                bindings: Dict[str, Set[str]],
                llm: BaseLLM,
                verbose: bool) -> Set[str]:
    """
    CHAIN: execute a sequence of simple hops in one CoT prompt.
    Seeds come from the first triple's source variable/constant.
    """
    t0 = step.triples[0]
    if not _var(t0.s):
        seeds = {t0.s}
    elif t0.s in bindings:
        seeds = bindings[t0.s]
    else:
        if verbose:
            print(f"  [CHAIN] no seeds for {t0.s} — skipping")
        return set()

    if not seeds:
        return set()

    if verbose:
        print(f"\n  [CHAIN] {len(step.triples)} hops, "
              f"seeds={len(seeds)}: {sorted(seeds)[:3]}")

    prompt = _chain_cot_prompt(step.triples, seeds)
    result = _run_value_prompt(prompt, llm)

    if verbose:
        print(f"  [CHAIN] → {len(result)}: {sorted(result)[:4]}")
    return result


def _exec_filter(step: ExecStep,
                 bindings: Dict[str, Set[str]],
                 llm: BaseLLM,
                 verbose: bool,
                 expand_triples: List[PTriple] = None) -> Set[str]:
    """
    FILTER: reduce candidates to those satisfying all constraints.
    Hard invariant: result ⊆ candidates (enforced with & after LLM call).

    expand_triples: triples from the EXPAND/CHAIN step that produced
    step.output_var, passed through to _filter_prompt for provenance context.
    """
    var = step.output_var
    candidates = bindings.get(var, set())
    if not candidates:
        if verbose:
            print(f"  [FILTER] {var}: no candidates — skip")
        return set()

    if verbose:
        print(f"\n  [FILTER] {var} ({len(candidates)} candidates)")

    prompt = _filter_prompt(var, candidates, step.triples, bindings,
                            expand_triples=expand_triples)
    result = _run_value_prompt(prompt, llm) & candidates  # hard invariant

    if verbose:
        print(f"  [FILTER] {var}: {len(candidates)} → {len(result)}: "
              f"{sorted(result)[:4]}")
    return result


def _exec_join(step: ExecStep,
               bindings: Dict[str, Set[str]],
               llm: BaseLLM,
               gamma: Environment,
               verbose: bool) -> Set[str]:
    """
    JOIN: Waveguide-style split-join for variables with no direct constant anchor.
    Fallback: treat as ANCHOR (holistic CoT without a type anchor).
    This is rare in T7 queries — most have at least one constant endpoint.
    """
    if verbose:
        print(f"  [JOIN] {step.output_var} — no constant anchor, "
              f"falling back to holistic ANCHOR prompt")
    return _exec_anchor(step, bindings, llm, verbose)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════


def _v4_split_eval(
    plan:    HybridPlan,
    llm:     BaseLLM,
    gamma:   Environment,
    verbose: bool = False,
) -> Set[str]:
    """
    Waveguide-style Split/Join for anchor_side="both" plans.

    Implemented locally to avoid circular import with mugalois_hybrid.
    Uses _exec_expand (forward) and _v4_eval_backward (backward).

    Forward prefix:  source →[nodes[:idx+1]]→ ?result_var → candidates
    Backward suffix: target ←[nodes[idx+1:]]← ?result_var → valid set
    Result = candidates ∩ valid
    """
    from mugalois.hybrid.hybrid_plan import HopNode, ClosureNode, FilterNode
    from mugalois.scans.ochestror_scan import LLMScan

    idx          = getattr(plan, "result_node_idx", 0)
    prefix_nodes = plan.nodes[:idx + 1]
    suffix_nodes = plan.nodes[idx + 1:]

    if verbose:
        print(f"  [split] prefix={len(prefix_nodes)} nodes "
              f"suffix={len(suffix_nodes)} nodes  result_idx={idx}")

    # ── Phase 1: Forward prefix ───────────────────────────────────────
    seeds = {plan.source}
    for node in prefix_nodes:
        if not seeds:
            return set()
        if isinstance(node, ClosureNode):
            t = PTriple(s=node.s or plan.source, p=node.predicate,
                        o=node.o or "?x",
                        is_closure=True, operator=node.operator)
            step = ExecStep("expand", "?_fwd", [t], 1)
            seeds = _exec_expand(step, {plan.source: {plan.source}},
                                 llm, gamma, verbose)
        elif isinstance(node, HopNode):
            t = PTriple(s=node.s or "?_s", p=node.predicate, o=node.o or "?x")
            step = ExecStep("expand", "?_fwd", [t], 1)
            src_var = node.s if node.s and node.s.startswith("?") else "?_s"
            seeds = _exec_expand(step, {src_var: seeds, node.s: seeds},
                                 llm, gamma, verbose)
    candidates = seeds

    if verbose:
        print(f"  [split] {len(candidates)} candidates: "
              f"{sorted(candidates)[:5]}")

    if not suffix_nodes:
        return candidates

    # ── Phase 2: Backward suffix ──────────────────────────────────────
    seeds = {plan.target}
    for node in reversed(suffix_nodes):
        if not seeds:
            return set()
        if isinstance(node, ClosureNode):
            pat = RecursivePattern("?x", node.predicate,
                                   next(iter(seeds)), node.operator)
            result = set()
            for seed in list(seeds):
                pat_s = RecursivePattern("?x", node.predicate, seed, node.operator)
                try:
                    result |= MuGaloisRec(pat_s, llm, gamma=gamma)
                except Exception:
                    pass
            seeds = result
        elif isinstance(node, HopNode):
            env = Environment()
            env.set("?_tgt", seeds)
            triples = LLMScan(
                pattern=TriplePattern("?x", node.predicate, "?_tgt"),
                env=env, gamma=gamma, llm=llm,
                tau_strategie=0.01,
            )
            seeds = {t.s for t in triples}

    valid_from_suffix = seeds

    if verbose:
        print(f"  [split] {len(valid_from_suffix)} valid from suffix")

    result = candidates & valid_from_suffix
    if verbose:
        print(f"  [split] → {len(result)} after join")
    return result


def plan_and_execute_v4(
    plan:    HybridPlan,
    llm:     BaseLLM,
    gamma:   Optional[Environment] = None,
    verbose: bool = False,
    rec_fn=None,
    hop_fn=None,
) -> Set[str]:
    """
    HybridPlannerV4 — main entry point.

    Steps:
      1. Extract PTriples from plan nodes + where_triples.
      2. Initialise bindings from constant endpoints.
      3. Schedule ExecSteps via dependency graph (ANCHOR→EXPAND→FILTER→JOIN).
      4. Merge adjacent hop EXPANDs into CHAIN steps.
      5. Execute in priority order, updating bindings after each step.
      6. Return bindings[plan.target_var].
    """
    gamma = gamma or Environment()

    # ── 0. SPLIT plans: anchor_side="both" ───────────────────────────
    # Two fixed anchors (source + target), result variable in the middle.
    # Waveguide Split/Join: forward prefix → candidates,
    # backward suffix → valid from target side, intersection = result.
    if getattr(plan, "anchor_side", "right") == "both":
        if verbose:
            print(f"[PlannerV4] anchor_side=both → split/join")
        return _v4_split_eval(plan, llm, gamma, verbose)

    # ── 1. Extract triples ────────────────────────────────────────────
    ptriples = _extract_ptriples(plan)

    if verbose:
        print(f"\n[PlannerV4] {plan.expression}")
        print(f"  source={plan.source}  target={plan.target}  "
              f"target_var={plan.target_var}")
        print(f"  {len(ptriples)} triples:")
        for t in ptriples:
            print(f"    {t}")

    # ── 2. Initial constant bindings ──────────────────────────────────
    bindings: Dict[str, Set[str]] = {}
    bound_consts: Set[str] = set()

    for endpoint in (plan.source, plan.target):
        if endpoint and not _var(endpoint):
            bound_consts.add(endpoint)
            bindings[endpoint] = {endpoint}

    # ── 3. Schedule steps ─────────────────────────────────────────────
    steps = _build_steps(ptriples, bound_consts, llm)

    # ── 4. Merge hop chains ───────────────────────────────────────────
    steps = _merge_hop_chains(steps)

    if verbose:
        print(f"\n[PlannerV4] {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"  {i}: [{s.priority}] {s.describe()}")

    # ── 5. Execute in priority order ──────────────────────────────────
    steps_sorted = sorted(steps, key=lambda s: s.priority)

    # Track which EXPAND/CHAIN triples produced each variable,
    # so FILTER steps can include provenance constraints.
    expand_triples_by_var: Dict[str, List[PTriple]] = {}

    for step in steps_sorted:
        if verbose:
            print(f"\n[PlannerV4] ▶ {step.kind.upper()} {step.output_var}")

        if step.kind == "anchor":
            result = _exec_anchor(step, bindings, llm, verbose)
        elif step.kind == "chain":
            result = _exec_chain(step, bindings, llm, verbose)
            expand_triples_by_var.setdefault(step.output_var, []).extend(step.triples)
        elif step.kind == "expand":
            result = _exec_expand(step, bindings, llm, gamma, verbose,
                                  rec_fn=rec_fn, hop_fn=hop_fn)
            expand_triples_by_var.setdefault(step.output_var, []).extend(step.triples)
        elif step.kind == "filter":
            # Pass provenance triples from the EXPAND/CHAIN that produced this var
            provenance = expand_triples_by_var.get(step.output_var)
            result = _exec_filter(step, bindings, llm, verbose,
                                  expand_triples=provenance)
        elif step.kind == "join":
            result = _exec_join(step, bindings, llm, gamma, verbose)
        else:
            result = set()

        var = step.output_var

        if step.kind == "filter":
            # FILTER always reduces — hard invariant
            bindings[var] = bindings.get(var, set()) & result
        elif var in bindings and bindings[var]:
            # Second EXPAND on same var (e.g. split-join convergence) → intersect
            bindings[var] &= result
        else:
            bindings[var] = result

        if verbose:
            n      = len(bindings.get(var, set()))
            sample = sorted(bindings.get(var, set()))[:4]
            print(f"  bindings[{var}] = {n}: {sample}")

        # Early stop: non-filter step returned empty → whole path fails
        if step.kind not in ("filter",) and not bindings.get(var):
            if verbose:
                print(f"  [PlannerV4] ∅ binding at {var} — early stop")
            return set()

    # ── 6. Return target ──────────────────────────────────────────────
    target = bindings.get(plan.target_var, set())
    if verbose:
        print(f"\n[PlannerV4] ✓ {plan.target_var}: {len(target)} results")
    return target

# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_C variant — confidence-only executors
# ══════════════════════════════════════════════════════════════════════════════

def plan_and_execute_v4_c(
    plan:    "HybridPlan",
    llm:     "BaseLLM",
    gamma:   "Optional[Environment]" = None,
    verbose: bool = False,
    tau:     float = 0.50,
) -> "Set[str]":
    """
    Confidence-only variant of plan_and_execute_v4 for µ-Galois_C on T7.

    Same ANCHOR/EXPAND/FILTER/CHAIN planner structure, but:
      - Closure EXPAND  → µ-Galois_C T5  (confidence-only recursive routing)
      - Hop     EXPAND  → µ-Galois_C T1  (confidence-only scan routing)

    α (tau) is propagated uniformly to all internal decisions, consistent
    with the global trust parameter defined in run_ablation_tau.py.
    """
    from mugalois.core.types import RecursivePattern, TriplePattern, Environment

    def rec_fn_c(pattern, llm, gamma=None, verbose=False):
        """Confidence-only recursive scan (T5)."""
        from mugalois.rec.mugalois_rec import MuGaloisRecConf
        try:
            return MuGaloisRecConf(pattern, llm,
                                   tau_high=tau, tau_low=max(0.0, tau - 0.20),
                                   gamma=gamma or Environment())
        except Exception:
            # Fallback to holistic if MuGaloisRecConf unavailable
            from mugalois.rec.simple_rec import LLMRecScan
            return LLMRecScan(pattern, llm, motivational=False)

    def hop_fn_c(t, seeds, forward, llm, clean_pred, lookahead):
        """Confidence-only hop scan (T1/T2 style)."""
        from mugalois.scans.ochestror_scan import LLMScan
        from mugalois.core.types import Environment as Env
        result = set()
        seed_list = list(seeds)
        for i in range(0, len(seed_list), SEED_BATCH_SIZE):
            batch = set(seed_list[i:i + SEED_BATCH_SIZE])
            env   = Env()
            if forward:
                var_name = "?_s"
                env.set(var_name, batch)
                triples_out = LLMScan(
                    TriplePattern(var_name, clean_pred, "?x"),
                    env, Env(), llm,
                    tau_strategie=tau,
                )
                result |= {x.o for x in triples_out}
            else:
                var_name = "?_o"
                env.set(var_name, batch)
                triples_out = LLMScan(
                    TriplePattern("?x", clean_pred, var_name),
                    env, Env(), llm,
                    tau_strategie=tau,
                )
                result |= {x.s for x in triples_out}
        return result

    return plan_and_execute_v4(
        plan, llm, gamma=gamma, verbose=verbose,
        rec_fn=rec_fn_c,
        hop_fn=hop_fn_c,
    )


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_S variant — structure & cardinality executors only
# ══════════════════════════════════════════════════════════════════════════════

def plan_and_execute_v4_s(
    plan:    "HybridPlan",
    llm:     "BaseLLM",
    gamma:   "Optional[Environment]" = None,
    verbose: bool = False,
) -> "Set[str]":
    """
    Structure & cardinality-only variant of plan_and_execute_v4 (µ-Galois_S).

    Same ANCHOR/EXPAND/FILTER/CHAIN structure, but:
      - Closure EXPAND → µ-Galois_S T5 (LLMStructureDetect + LLMEstimateRecSize,
                                         no LLMAtomicRecConf)
      - Hop     EXPAND → LLMScan standard (no confidence routing)

    No confidence call is made at any level.
    """
    from mugalois.core.types import RecursivePattern, TriplePattern, Environment

    def rec_fn_s(pattern, llm, gamma=None, verbose=False):
        """Structure+cardinality recursive scan — no confidence."""
        from mugalois.rec.mugalois_rec import LLMStructureDetect
        from mugalois.rec.simple_rec import LLMEstimateRecSize, LLMRecScan
        from mugalois.rec.fixpoint import LLMFixpointScan
        g = gamma or Environment()
        structure = LLMStructureDetect(pattern, llm)
        if structure == "dag":
            return LLMRecScan(pattern, llm, motivational=False)
        size = LLMEstimateRecSize(pattern, llm)
        CARD_MAX = 15
        if 0 < size <= CARD_MAX:
            return LLMFixpointScan(pattern, llm, gamma=g)
        return LLMRecScan(pattern, llm, motivational=False)

    def hop_fn_s(t, seeds, forward, llm, clean_pred, lookahead):
        """Standard hop scan — no confidence routing."""
        result = set()
        seed_list = list(seeds)
        for i in range(0, len(seed_list), SEED_BATCH_SIZE):
            batch = set(seed_list[i:i + SEED_BATCH_SIZE])
            env   = Environment()
            if forward:
                var_name = "?_s"
                env.set(var_name, batch)
                triples_out = LLMSeedScan(
                    TriplePattern(var_name, clean_pred, "?x"), env, llm,
                    inject_conds=[], post_filter_conds=[],
                    lookahead=lookahead,
                )
                result |= {x.o for x in triples_out}
            else:
                var_name = "?_o"
                env.set(var_name, batch)
                triples_out = LLMSeedScan(
                    TriplePattern("?x", clean_pred, var_name), env, llm,
                    inject_conds=[], post_filter_conds=[],
                    lookahead=lookahead,
                )
                result |= {x.s for x in triples_out}
        return result

    return plan_and_execute_v4(
        plan, llm, gamma=gamma, verbose=verbose,
        rec_fn=rec_fn_s,
        hop_fn=hop_fn_s,
    )