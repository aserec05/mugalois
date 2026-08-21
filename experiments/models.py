"""
src/mugalois/evaluation/models.py
==================================
The five µ-Galois evaluation models, template-agnostic.

All thresholds are explicit named parameters on each model function.
No magic numbers hidden inside the implementation — every routing
decision is visible and modifiable from the call site.

Models
------
  MuGaloisHol  : Holistic forced — one holistic LLM call per plan step
  MuGaloisDec  : Decomposed forced — planner detects anchor side,
                 then forces maximum decomposition (KeyScan, fixpoint, parallel)
  MuGaloisC    : Confidence-only routing — structure and cardinality disabled
  MuGaloisFM   : Full routing WITHOUT motivational prompting
  MuGaloisF    : Full system — all signals + motivational (the complete µ-Galois)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Set

from mugalois.llm.llm_client import BaseLLM


# ── QueryContext ───────────────────────────────────────────────────────────────

@dataclass
class QueryContext:
    """Template-agnostic query descriptor built by each template adapter."""
    template:    str          # "T1" | "T2" | "T3" | "T4" | "T5" | "T6" | "T7"

    # Baselines
    nl_naive:    str = ""
    nl_precise:  str = ""
    sparql:      str = ""

    # T1-T2
    triple:      Any = None   # TriplePattern
    env:         Any = None   # Environment
    gamma:       Any = None   # Environment (conditions)

    # T3
    s:           str = ""
    p1:          str = ""
    p2:          str = ""
    t:           str = ""

    # T4
    path:        Any = None   # PathQuery

    # T5
    pattern:     Any = None   # RecursivePattern

    # T6
    choice_path: Any = None   # ChoicePath

    # T7
    plan:        Any = None   # HybridPlan

    extra:       dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _apply_numeric_filter(values: Set[str], ctx) -> Set[str]:
    """
    Numeric post-filter conditions (>, <, !=) cannot be applied on string
    values without a database. The LLM already understood the numeric
    constraint from the NL context — return values as-is.
    This function is a documented no-op placeholder for transparency.
    """
    return values


def _triples_to_values(triples, pattern, ctx=None) -> Set[str]:
    """Extract string values from a set[Triple].
    Uses ctx.extra['return_var'] when available to pick the right side.
    Falls back to: if only o is var → extract o, else extract s.
    """
    return_var = ctx.extra.get("return_var") if ctx else None
    if return_var:
        if return_var == pattern.o:
            return {tr.o for tr in triples}
        else:
            return {tr.s for tr in triples}
    # Fallback: extract the variable side
    if pattern.s_is_var() and not pattern.o_is_var():
        return {tr.s for tr in triples}
    return {tr.o for tr in triples}

def _env(ctx):
    import copy
    from mugalois.core.types import Environment
    return copy.deepcopy(ctx.env) if ctx.env is not None else Environment()

def _gamma(ctx):
    import copy
    from mugalois.core.types import Environment
    return copy.deepcopy(ctx.gamma) if ctx.gamma is not None else Environment()

def _choice_path(ctx):
    """Retrieve the ChoicePath from ctx.extra (set by _adapter_t6)."""
    cp = ctx.extra.get("choice_path")
    if cp is None:
        raise ValueError(
            "T6 QueryContext has no 'choice_path' in extra. "
            "Check that _adapter_t6 builds and stores it correctly."
        )
    return cp


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_Hol
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisHol(
    ctx: QueryContext,
    llm: BaseLLM,
) -> Set[str]:
    """
    µ-Galois_Hol — holistic forced everywhere.

    T1-T2 : LLMScan(tau_strategie=1.0) → always SeedScan, never KeyScan
    T3    : LLMSimpleScan — one call for the full 2-hop path
    T4    : LLMMultiHopScan — one holistic call for the full N-hop path
    T5    : LLMRecScan(motivational=False) — one holistic call, no fixpoint
    T6    : MuGaloisChoice(tau_hol=1.1, tau_par=1.1) → always holistic OR
    T7    : HybridPlannerV4 builds steps; each step executed with its
            real holistic prompt (_anchor_prompt, _holistic_closure,
            _chain_cot_prompt, _filter_prompt). Multiple calls, none decomposed.
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.seed_scan import LLMSeedScan
        from mugalois.core.types import Environment
        triples = LLMSeedScan(
            ctx.triple,
            ctx.env or Environment(),
            llm,
            inject_conds=[],
            post_filter_conds=ctx.extra.get("post_filter_conds", []),
            motivational=False,
        )
        return _triples_to_values(triples, ctx.triple, ctx)

    if t == "T3":
        from mugalois.paths.simple import LLMSimpleScan
        return LLMSimpleScan(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        from mugalois.paths.multi_scan import LLMMultiHopScan
        return LLMMultiHopScan(ctx.path, llm)

    if t == "T5":
        from mugalois.rec.simple_rec import LLMRecScan
        return LLMRecScan(ctx.pattern, llm, motivational=False)

    if t == "T6":
        # Hol: force holistic OR — one single prompt, no routing, no motivational.
        # We call _holistic_prompt directly so similarity/confidence are bypassed.
        from mugalois.choice.mugalois_choice import _holistic_prompt, _partition, _parallel_complex
        from mugalois.core.prompts import build_value_messages
        from mugalois.core.parser import json_to_values
        from mugalois.core.types import Environment
        from mugalois.choice.choice_path import ChoicePath
        cp = _choice_path(ctx)
        simple, complex_ = _partition(cp.branches)
        result = set()
        if simple:
            simple_cp = ChoicePath(source=cp.source, target_var=cp.target_var,
                                   branches=simple, direction=cp.direction)
            prompt = _holistic_prompt(simple_cp)
            result = json_to_values(llm.chat(build_value_messages(prompt)).text)
        if complex_:
            result |= _parallel_complex(complex_, cp, llm, Environment())
        return result

    if t == "T7":
        from mugalois.core.types import Environment
        return _t7_hol(ctx.plan, llm, ctx.gamma or Environment())

    return set()



# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_Dec
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisDec(
    ctx: QueryContext,
    llm: BaseLLM,
    *,
    max_depth_t5:   int   = 300,   # fixpoint max depth for T5
    max_iter_t4:    int   = 5,     # chain scan max iterations for T4
) -> Set[str]:
    """
    µ-Galois_Dec — decomposed forced everywhere.

    The planner still detects the optimal anchor side (left/right) from
    the query structure — this is structural knowledge, not LLM routing.
    Execution is then forced to maximum decomposition:

    T1-T2 : LLMScan(mode='key_only') → KeyScan, one call per seed
    T3    : scan_gd / scan_dg / scan_join — direction from query structure
            (which side has a constant anchor), no LLM confidence call
    T4    : scan_chain — forced chain decomposition, direction from anchors
    T5    : LLMFixpointScan — iterative fixpoint, no holistic call
    T6    : AllParallel — one LLMScan per branch, results unioned
    T7    : plan_and_execute_v4 with TAU_REC_HIGH=0, TAU_REC_LOW=0
            → MuGaloisRec always falls to fixpoint (low confidence branch)
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.ochestror_scan import LLMScan
        triples = LLMScan(
            pattern=ctx.triple, env=_env(ctx), gamma=_gamma(ctx), llm=llm,
            mode="key_only",
        )
        return _triples_to_values(triples, ctx.triple, ctx)
        # post_filter_conds handled by LLMScan via gamma

    if t == "T3":
        # Direction from query structure: which side has a constant?
        direction = ctx.extra.get("direction", "left")
        if direction == "left":
            from mugalois.paths.gd import scan_gd
            return scan_gd(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        elif direction == "right":
            from mugalois.paths.dg import scan_dg
            return scan_dg(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        else:
            from mugalois.paths.join import scan_join
            return scan_join(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        from mugalois.paths.chain import scan_chain
        from mugalois.core.types import Environment
        # Direction detected from anchors in the PathQuery (no LLM call)
        direction = ctx.extra.get("direction", "left")
        return scan_chain(
            path=ctx.path, llm=llm,
            direction=direction,
            gamma=ctx.gamma or Environment(),
            max_iter=max_iter_t4,
        )

    if t == "T5":
        from mugalois.rec.fixpoint import LLMFixpointScan
        return LLMFixpointScan(
            ctx.pattern, llm,
            max_depth=max_depth_t5,
        )

    if t == "T6":
        # Dec: AllParallel — one call per branch (simple=LLMScan, complex=best orchestrator)
        from mugalois.choice.mugalois_choice import _parallel_simple, _parallel_complex, _partition
        cp = _choice_path(ctx)
        simple, complex_ = _partition(cp.branches)
        from mugalois.core.types import Environment
        result = set()
        if simple:
            result |= _parallel_simple(simple, cp, llm)
        if complex_:
            result |= _parallel_complex(complex_, cp, llm, Environment())
        return result

    if t == "T7":
        from mugalois.core.types import Environment
        return _t7_dec(ctx.plan, llm, ctx.gamma or Environment())

    return set()


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_C
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisC(
    ctx: QueryContext,
    llm: BaseLLM,
    *,
    tau_scan:    float = 0.70,  # T1-T2 : SeedScan vs KeyScan threshold
    tau_simple:  float = 0.80,  # T3    : SimpleScan vs GD/DG threshold
    tau_chain:   float = 0.50,  # T4    : holistic vs chain threshold
    tau_rec:     float = 0.50,  # T5    : holistic vs fixpoint threshold
    tau_choice:  float = 0.50,  # T6    : holistic OR vs parallel threshold
    tau_hybrid:  float = 0.50,  # T7    : holistic vs decomposed threshold
) -> Set[str]:
    """
    µ-Galois_C — confidence signal ONLY; structure and cardinality disabled.

    Each template has its own named tau parameter.
    No structural detection (LLMStructureDetect, LLMBranchSimilarity,
    LLMEstimateTypeSize, LLMEstimateSize) is called — confidence alone
    decides holistic vs decomposed at every level.

    T1-T2 : LLMScan with tau_strategie=tau_scan (conf → SeedScan vs KeyScan)
    T3    : LLMSimpleConf > tau_simple → SimpleScan, else GD (default direction)
    T4    : LLMChainConf > tau_chain → LLMMultiHopScan, else scan_chain(left)
    T5    : LLMAtomicRecConf > tau_rec → LLMRecScan, else LLMFixpointScan
            (LLMStructureDetect bypassed)
    T6    : LLMChoiceConf > tau_choice → holistic OR, else AllParallel
            (LLMBranchSimilarity bypassed)
    T7    : LLMAtomicConf on plan > tau_hybrid → _t7_holistic_steps,
            else plan_and_execute_v4 with LLMStructureDetect patched to 'chain'
            and TYPE_SELECTIVITY_THRESHOLD=0 (no TYPE ANCHOR)
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.ochestror_scan import LLMScan
        triples = LLMScan(
            pattern=ctx.triple, env=_env(ctx), gamma=_gamma(ctx), llm=llm,
            tau_strategie=tau_scan,
        )
        return _triples_to_values(triples, ctx.triple, ctx)

    if t == "T3":
        from mugalois.paths.simple import LLMSimpleConf, LLMSimpleScan
        from mugalois.paths.gd import scan_gd
        conf = LLMSimpleConf(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        if conf > tau_simple:
            return LLMSimpleScan(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        return scan_gd(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        from mugalois.paths.multi_scan import LLMChainConf, LLMMultiHopScan
        from mugalois.paths.chain import scan_chain
        from mugalois.core.types import Environment
        conf = LLMChainConf(ctx.path, llm)
        if conf > tau_chain:
            return LLMMultiHopScan(ctx.path, llm)
        return scan_chain(
            path=ctx.path, llm=llm,
            direction="left",
            gamma=ctx.gamma or Environment(),
        )

    if t == "T5":
        # Bypass LLMStructureDetect — call LLMAtomicRecConf directly
        from mugalois.rec.simple_rec import LLMAtomicRecConf, LLMRecScan
        from mugalois.rec.fixpoint import LLMFixpointScan
        conf = LLMAtomicRecConf(ctx.pattern, llm)
        if conf > tau_rec:
            return LLMRecScan(ctx.pattern, llm, motivational=False)
        return LLMFixpointScan(ctx.pattern, llm)

    if t == "T6":
        # Bypass LLMBranchSimilarity — call LLMChoiceConf directly
        from mugalois.choice.choice_path import LLMChoiceConf
        from mugalois.choice.mugalois_choice import (
            _holistic_prompt, _parallel_simple, _partition,
            _parallel_complex,
        )
        from mugalois.core.prompts import build_value_messages
        from mugalois.core.parser import json_to_values
        from mugalois.core.types import Environment

        cp = _choice_path(ctx)
        simple, complex_ = _partition(cp.branches)
        conf = LLMChoiceConf(cp, llm)
        if conf > tau_choice:
            prompt = _holistic_prompt(cp)
            result = json_to_values(llm.chat(build_value_messages(prompt)).text)
        else:
            result = _parallel_simple(simple, cp, llm)
        if complex_:
            result |= _parallel_complex(complex_, cp, llm, Environment())
        return result

    if t == "T7":
        from mugalois.core.types import Environment
        return _t7_conf(ctx.plan, llm, ctx.gamma or Environment(), tau=tau_hybrid)

    return set()


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_S  — Structure & Cardinality signals ONLY (no confidence)
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisS(
    ctx: QueryContext,
    llm: BaseLLM,
    *,
    tau_rec_high:    float = 0.50,   # T5 : size threshold (DAG conf proxy)
    tau_rec_low:     float = 0.30,
    tau_choice_high: float = 0.60,   # T6 : similarity-based routing
    tau_choice_low:  float = 0.40,
    coverage:        float = 1.1,    # no motivational
) -> Set[str]:
    """
    µ-Galois_S — Structure and Cardinality signals ONLY.

    No LLM confidence call is made for routing decisions.
    Structural signals (DAG/chain detection, cardinality estimation,
    branch similarity) drive all routing choices.
    Motivational prompting disabled (coverage=1.1).

    T1-T2 : LLMScan standard (no routing change — scan operators unchanged)
    T3    : LLMDirectionConf → direction (cardinality-based), no LLMSimpleConf
    T4    : decide_routing (cardinality) directly, skip LLMChainConf
            → holistic if single-hop, chain otherwise
    T5    : LLMStructureDetect + LLMEstimateRecSize decide fixpoint vs holistic
            LLMAtomicRecConf bypassed
    T6    : LLMBranchSimilarity + LLMEstimateChoiceSize decide parallel vs holistic
            LLMChoiceConf bypassed
    T7    : plan_and_execute_v4 full planner (already structure-driven)
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.ochestror_scan import LLMScan
        triples = LLMScan(
            pattern=ctx.triple, env=_env(ctx), gamma=_gamma(ctx), llm=llm,
            motivational=False,
        )
        return _triples_to_values(triples, ctx.triple, ctx)

    if t == "T3":
        # Direction driven by cardinality (LLMDirectionConf), no holistic check
        from mugalois.paths.simple import LLMDirectionConf
        from mugalois.paths.gd import scan_gd
        from mugalois.paths.dg import scan_dg
        from mugalois.paths.join import scan_join
        direction = LLMDirectionConf(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        if direction == "left":
            return scan_gd(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        elif direction == "right":
            return scan_dg(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        else:
            return scan_join(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        # Routing decided by cardinality estimation only — no LLMChainConf
        from mugalois.paths.routing import decide_routing
        from mugalois.paths.chain import scan_chain
        from mugalois.paths.split import scan_split
        from mugalois.paths.chain import scan_chain_from_bound
        from mugalois.paths.multi_scan import LLMMultiHopScan
        from mugalois.core.types import Environment
        gamma  = ctx.gamma or Environment()
        routing = decide_routing(ctx.path, llm, gamma)
        if routing["strategy"] == "bound":
            return scan_chain_from_bound(
                path=ctx.path, var=routing["bound_var"],
                seeds=gamma.get(routing["bound_var"]),
                llm=llm, gamma=gamma,
            )
        elif routing["strategy"] == "chain":
            return scan_chain(
                path=ctx.path, llm=llm,
                direction=routing["direction"], gamma=gamma,
            )
        else:
            return scan_split(
                path=ctx.path, split_var=routing["split_var"],
                llm=llm, gamma=gamma,
            )

    if t == "T5":
        # Structure + size decide — no confidence call
        from mugalois.rec.mugalois_rec import LLMStructureDetect
        from mugalois.rec.simple_rec import LLMEstimateRecSize, LLMRecScan
        from mugalois.rec.fixpoint import LLMFixpointScan
        from mugalois.core.types import Environment
        structure = LLMStructureDetect(ctx.pattern, llm)
        size      = LLMEstimateRecSize(ctx.pattern, llm)
        if structure == "dag":
            # DAG: always holistic (fixpoint diverges on DAGs)
            return LLMRecScan(ctx.pattern, llm, motivational=False)
        else:
            # Chain: size decides fixpoint vs holistic
            CARD_MAX = 15
            if 0 < size <= CARD_MAX:
                return LLMFixpointScan(
                    ctx.pattern, llm,
                    gamma=ctx.gamma or Environment(),
                )
            else:
                return LLMRecScan(ctx.pattern, llm, motivational=False)

    if t == "T6":
        # Similarity + size decide — no confidence call
        from mugalois.choice.choice_path import (
            LLMBranchSimilarity, LLMEstimateChoiceSize, ChoicePath,
        )
        from mugalois.choice.mugalois_choice import (
            _holistic_prompt, _parallel_simple, _partition, _parallel_complex,
        )
        from mugalois.core.prompts import build_value_messages
        from mugalois.core.parser import json_to_values
        from mugalois.core.types import Environment
        cp = _choice_path(ctx)
        simple, complex_ = _partition(cp.branches)
        result = set()
        if simple:
            simple_cp = ChoicePath(
                source=cp.source, target_var=cp.target_var,
                branches=simple, direction=cp.direction,
            )
            similarity = LLMBranchSimilarity(simple_cp, llm)
            if similarity == "distinct":
                # Distinct branches → holistic OR safe
                est = LLMEstimateChoiceSize(simple_cp, llm)
                prompt = _holistic_prompt(simple_cp)
                result = json_to_values(llm.chat(build_value_messages(prompt)).text)
            else:
                # Similar branches → parallel safer (proactive interference)
                result = _parallel_simple(simple, cp, llm)
        if complex_:
            result |= _parallel_complex(complex_, cp, llm, Environment())
        return result

    if t == "T7":
        # _S on T7: uses plan_and_execute_v4_s which injects structure-only
        # executors for both closure EXPANDs (LLMStructureDetect+LLMEstimateRecSize)
        # and hop EXPANDs (LLMSeedScan, no confidence routing).
        from mugalois.hybrid.hybrid_planner_v4 import plan_and_execute_v4_s
        from mugalois.core.types import Environment
        return plan_and_execute_v4_s(ctx.plan, llm,
                                     gamma=ctx.gamma or Environment())

    return set()


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_F-M
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisFM(
    ctx: QueryContext,
    llm: BaseLLM,
    *,
    tau_choice_high: float = 0.60,   # T6 MuGaloisChoice high threshold
    tau_choice_low:  float = 0.40,   # T6 MuGaloisChoice low threshold
    coverage_t4:     float = 1.1,    # T4 coverage — disables motivational
) -> Set[str]:
    """
    µ-Galois_F-M — full routing WITHOUT motivational prompting.

    All routing signals active (conf + structure + cardinality).
    Motivational prompting disabled via parameter tricks — no new code:

    T1-T2 : LLMScan(motivational=False)
    T3    : MuGaloisPath(use_closure=False) — MotivationalClosure disabled
    T4    : MuGaloisMultiPath(coverage=coverage_t4=1.1) — cardinality
            check never triggers motivational (len < 1.1×est always false)
    T5    : MuGaloisRec(tau_high=tau_rec_hol=99) — conf always > tau_high
            → LLMRecScan(motivational=False) always chosen
    T6    : MuGaloisChoice(coverage=coverage_t6=1.1) — motivational never
            triggered
    T7    : plan_and_execute_v4 with TAU_REC_HOL=99 patched into module
            → MuGaloisRec always uses LLMRecScan(motivational=False)
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.ochestror_scan import LLMScan
        triples = LLMScan(
            pattern=ctx.triple, env=_env(ctx), gamma=_gamma(ctx), llm=llm,
            motivational=False,
        )
        return _triples_to_values(triples, ctx.triple, ctx)

    if t == "T3":
        # F-M: full routing (conf + direction from ctx), no motivational closure
        from mugalois.paths.simple import LLMSimpleConf, LLMSimpleScan
        from mugalois.paths.gd import scan_gd
        from mugalois.paths.dg import scan_dg
        from mugalois.paths.join import scan_join
        conf = LLMSimpleConf(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        if conf > 0.8:
            return LLMSimpleScan(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        direction = ctx.extra.get("direction", "left")
        if direction == "left":
            return scan_gd(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        elif direction == "right":
            return scan_dg(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        else:
            return scan_join(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        from mugalois.paths.mugalois_multi import MuGaloisMultiPath
        # F-M: tau_high=0.0 → always holistic atom, no routing/estimation
        # coverage=1.1 → motivational never triggered
        return MuGaloisMultiPath(
            ctx.path, llm,
            tau_high=0.0,
            coverage=1.1,
        )

    if t == "T5":
        from mugalois.rec.mugalois_rec import MuGaloisRec
        return MuGaloisRec(
            ctx.pattern, llm,
            motivational=False,
        )

    if t == "T6":
        from mugalois.choice.mugalois_choice import MuGaloisChoice
        return MuGaloisChoice(
            _choice_path(ctx), llm,
            tau_high=tau_choice_high,
            tau_low=tau_choice_low,
            motivational=False,
        )

    if t == "T7":
        # Wrap MuGaloisRec in the planner module to force motivational=False
        # without touching thresholds or plan logic
        import mugalois.hybrid.hybrid_planner_v4 as _v4
        import mugalois.rec.mugalois_rec as _rec_mod
        _orig_rec = _v4.MuGaloisRec
        _v4.MuGaloisRec = lambda pat, llm_inner, **kw: _rec_mod.MuGaloisRec(
            pat, llm_inner, motivational=False, **kw
        )
        try:
            from mugalois.hybrid.hybrid_planner_v4 import plan_and_execute_v4
            from mugalois.core.types import Environment
            result = plan_and_execute_v4(
                ctx.plan, llm, ctx.gamma or Environment()
            )
        finally:
            _v4.MuGaloisRec = _orig_rec
        return result

    return set()


# ══════════════════════════════════════════════════════════════════════════════
# µ-Galois_F
# ══════════════════════════════════════════════════════════════════════════════

def MuGaloisF(
    ctx: QueryContext,
    llm: BaseLLM,
    *,
    tau_rec_high:    float = 0.50,   # T5 MuGaloisRec high threshold
    tau_rec_low:     float = 0.35,   # T5 MuGaloisRec low threshold
    tau_choice_high: float = 0.60,   # T6 MuGaloisChoice high threshold
    tau_choice_low:  float = 0.40,   # T6 MuGaloisChoice low threshold
    coverage_t4:     float = 0.80,   # T4 motivational coverage threshold
    coverage_t6:     float = 0.80,   # T6 motivational coverage threshold
) -> Set[str]:
    """
    µ-Galois_F — full system, all signals active.

    T1-T2 : LLMScan standard (conf → SeedScan vs KeyScan, RW1 conditions)
    T3    : MuGaloisPath (SimpleConf + DirectionConf + MotivationalClosure)
    T4    : MuGaloisMultiPath (ChainConf + EstimateSize + routing)
    T5    : MuGaloisRec (StructureDetect + AtomicRecConf + EstimateRecSize)
    T6    : MuGaloisChoice (ChoiceConf + BranchSimilarity + EstimateChoiceSize)
    T7    : plan_and_execute_v4 (full planner + MuGaloisRec + LLMSeedScan)
    """
    t = ctx.template

    if t in ("T1", "T2"):
        from mugalois.scans.ochestror_scan import LLMScan
        triples = LLMScan(
            pattern=ctx.triple, env=_env(ctx), gamma=_gamma(ctx), llm=llm,
        )
        return _triples_to_values(triples, ctx.triple, ctx)

    if t == "T3":
        # F: full routing (conf + direction from ctx) WITH motivational closure
        from mugalois.paths.simple import LLMSimpleConf, LLMSimpleScan
        from mugalois.paths.gd import scan_gd
        from mugalois.paths.dg import scan_dg
        from mugalois.paths.join import scan_join
        conf = LLMSimpleConf(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        if conf > 0.8:
            return LLMSimpleScan(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)
        direction = ctx.extra.get("direction", "left")
        from mugalois.closures.motivational import MotivationalClosure
        from mugalois.closures.pipeline import ClosurePipeline
        pipeline = ClosurePipeline([MotivationalClosure()], n_break=2, max_iter=8)
        if direction == "left":
            return scan_gd(ctx.s, ctx.p1, ctx.p2, ctx.t, llm, pipeline=pipeline)
        elif direction == "right":
            return scan_dg(ctx.s, ctx.p1, ctx.p2, ctx.t, llm, pipeline=pipeline)
        else:
            return scan_join(ctx.s, ctx.p1, ctx.p2, ctx.t, llm)

    if t == "T4":
        from mugalois.paths.mugalois_multi import MuGaloisMultiPath
        # F: tau_high=0.0 → holistic atom, coverage decides motivational
        return MuGaloisMultiPath(
            ctx.path, llm,
            tau_high=0.0,
            coverage=coverage_t4,
        )

    if t == "T5":
        from mugalois.rec.mugalois_rec import MuGaloisRec
        return MuGaloisRec(
            ctx.pattern, llm,
            tau_high=tau_rec_high,
            tau_low=tau_rec_low,
        )

    if t == "T6":
        from mugalois.choice.mugalois_choice import MuGaloisChoice
        return MuGaloisChoice(
            _choice_path(ctx), llm,
            tau_high=tau_choice_high,
            tau_low=tau_choice_low,
            coverage=coverage_t6,
        )

    if t == "T7":
        from mugalois.hybrid.hybrid_planner_v4 import plan_and_execute_v4
        from mugalois.core.types import Environment
        return plan_and_execute_v4(
            ctx.plan, llm, ctx.gamma or Environment()
        )

    return set()



# ── T7 step-by-step helpers (inline, no mugalois_hybrid dependency) ───────────

def _t7_steps(plan, llm, gamma):
    """Build and sort plan steps from hybrid_planner_v4."""
    from mugalois.hybrid.hybrid_planner_v4 import (
        _extract_ptriples, _build_steps, _merge_hop_chains,
    )
    from mugalois.core.types import Environment
    gamma = gamma or Environment()
    bound_consts = {ep for ep in (plan.source, plan.target)
                   if ep and not ep.startswith("?")}
    steps = _merge_hop_chains(
        _build_steps(_extract_ptriples(plan), bound_consts, llm)
    )
    return sorted(steps, key=lambda s: s.priority), bound_consts, gamma


def _t7_anchor_prompt(var, triples, bindings) -> str:
    """Holistic anchor prompt from plan triples."""
    from mugalois.hybrid.hybrid_planner_v4 import _anchor_prompt
    return _anchor_prompt(var, triples, bindings)


def _t7_run(plan, llm, gamma, exec_fn) -> Set[str]:
    """
    Execute a T7 plan step by step.
    exec_fn(step, bindings, llm, gamma) → set[str]
    """
    from mugalois.hybrid.hybrid_planner_v4 import (
        _run_value_prompt, _filter_prompt, _anchor_prompt,
    )
    from mugalois.core.types import Environment

    steps, bound_consts, gamma = _t7_steps(plan, llm, gamma)
    bindings: dict = {ep: {ep} for ep in bound_consts}
    expand_by_var: dict = {}

    for step in steps:
        var = step.output_var

        if step.kind == "filter":
            candidates = bindings.get(var, set())
            if not candidates:
                continue
            provenance = expand_by_var.get(var)
            prompt = _filter_prompt(var, candidates, step.triples,
                                    bindings, expand_triples=provenance)
            bindings[var] = _run_value_prompt(prompt, llm) & candidates
            continue

        result = exec_fn(step, bindings, llm, gamma)

        if step.kind in ("expand", "chain"):
            expand_by_var.setdefault(var, []).extend(step.triples)

        if var in bindings and bindings[var]:
            bindings[var] &= result
        else:
            bindings[var] = result

        if not bindings.get(var):
            return set()

    return bindings.get(plan.target_var, set())


def _t7_hol(plan, llm, gamma) -> Set[str]:
    """µ-Galois_Hol T7: plan steps executed holistically."""
    from mugalois.hybrid.hybrid_planner_v4 import (
        _anchor_prompt, _chain_cot_prompt, _run_value_prompt,
    )
    from mugalois.rec.simple_rec import LLMRecScan
    from mugalois.scans.seed_scan import LLMSeedScan
    from mugalois.core.types import TriplePattern, RecursivePattern, Environment

    def exec_hol(step, bindings, llm, gamma):
        var = step.output_var
        if step.kind == "anchor":
            prompt = _anchor_prompt(var, step.triples, bindings)
            return _run_value_prompt(prompt, llm)
        elif step.kind == "chain":
            t0    = step.triples[0]
            seeds = bindings.get(t0.s, {t0.s}) if t0.s else set()
            prompt = _chain_cot_prompt(step.triples, seeds)
            return _run_value_prompt(prompt, llm)
        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                if len(seeds) == 1:
                    pat = RecursivePattern(next(iter(seeds)), t.p, "?x", t.operator)
                    return LLMRecScan(pat, llm, motivational=False)
                else:
                    # Holistic closure prompt for multiple seeds
                    seed_str = ", ".join(sorted(seeds)[:20])
                    pred = t.p
                    prompt = (
                        "Find all " + var + " reachable from " + seed_str +
                        " via '" + pred + "' (one or more steps).\n"
                        "Be exhaustive. Return ONLY a JSON array."
                    )
                    return _run_value_prompt(prompt, llm)
            else:
                env = Environment()
                env.set("?_s", seeds)
                from mugalois.scans.seed_scan import LLMSeedScan
                return {tr.o for tr in
                        LLMSeedScan(TriplePattern("?_s", t.p, "?x"), env, llm, [], [])}
        return set()

    return _t7_run(plan, llm, gamma, exec_hol)


def _t7_dec(plan, llm, gamma) -> Set[str]:
    """µ-Galois_Dec T7: plan steps with maximum decomposition."""
    from mugalois.hybrid.hybrid_planner_v4 import (
        _anchor_prompt, _run_value_prompt,
    )
    from mugalois.rec.fixpoint import LLMFixpointScan
    from mugalois.scans.key_scan import LLMKeyScan
    from mugalois.scans.ochestror_scan import LLMScan
    from mugalois.core.types import TriplePattern, RecursivePattern, Environment

    def exec_dec(step, bindings, llm, gamma):
        var = step.output_var
        if step.kind == "anchor":
            # Anchor is always holistic — structural starting point
            prompt = _anchor_prompt(var, step.triples, bindings)
            return _run_value_prompt(prompt, llm)
        elif step.kind == "chain":
            # Chain: one LLMScan per hop
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
            return result
        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                fwd = not t.o.startswith("?") or t.o not in bindings
                if len(seeds) == 1:
                    seed = next(iter(seeds))
                    pat  = (RecursivePattern(seed, t.p, "?x", t.operator)
                            if fwd else RecursivePattern("?x", t.p, seed, t.operator))
                    return LLMFixpointScan(pat, llm, gamma=gamma)
                else:
                    from mugalois.hybrid.mugalois_hybrid import _batch_fixpoint
                    from mugalois.hybrid.hybrid_plan import ClosureNode
                    cl = ClosureNode(predicate=t.p, operator=t.operator)
                    return _batch_fixpoint(cl, seeds, llm,
                                           direction="forward" if fwd else "backward")
            else:
                result = set()
                for seed in seeds:
                    env = Environment()
                    env.set("?_s", {seed})
                    result |= {tr.o for tr in
                               LLMKeyScan(TriplePattern("?_s", t.p, "?x"),
                                          env, llm, [], [])}
                return result
        return set()

    return _t7_run(plan, llm, gamma, exec_dec)


def _t7_conf(plan, llm, gamma, tau=0.60) -> Set[str]:
    """µ-Galois_C T7: confidence-only routing at each step."""
    from mugalois.hybrid.hybrid_planner_v4 import (
        _anchor_prompt, _chain_cot_prompt, _run_value_prompt,
        _extract_ptriples, _build_steps, _merge_hop_chains,
    )
    from mugalois.rec.simple_rec import LLMAtomicRecConf, LLMRecScan
    from mugalois.rec.fixpoint import LLMFixpointScan
    from mugalois.scans.seed_scan import LLMSeedScan
    from mugalois.scans.key_scan import LLMKeyScan
    from mugalois.scans.ochestror_scan import LLMScan
    from mugalois.core.types import TriplePattern, RecursivePattern, Environment
    from mugalois.core.prompts import build_messages

    def _conf(desc):
        prompt = (
            f"Between 0 and 1, how confident are you that you can answer "
            f"the following completely and accurately in a single response?\n"
            f"{desc}\n"
            f"Answer with a single decimal number only."
        )
        try:
            resp = llm.chat(build_messages(prompt))
            return max(0.0, min(1.0, float(resp.text.strip().split()[0])))
        except Exception:
            return 0.5

    # Replace LLMEstimateTypeSize with confidence call for plan building
    import mugalois.hybrid.hybrid_planner_v4 as _v4
    _orig = _v4.LLMEstimateTypeSize
    _v4.LLMEstimateTypeSize = lambda tv, l: (1 if _conf(f"List every {tv}") > tau else 999999)

    try:
        steps, bound_consts, gamma2 = _t7_steps(plan, llm, gamma)
    finally:
        _v4.LLMEstimateTypeSize = _orig

    bindings: dict = {ep: {ep} for ep in bound_consts}
    from mugalois.hybrid.hybrid_planner_v4 import _filter_prompt
    expand_by_var: dict = {}

    for step in steps:
        var = step.output_var

        if step.kind == "filter":
            candidates = bindings.get(var, set())
            if not candidates:
                continue
            prompt = _filter_prompt(var, candidates, step.triples,
                                    bindings, expand_triples=expand_by_var.get(var))
            bindings[var] = _run_value_prompt(prompt, llm) & candidates
            continue

        if step.kind == "anchor":
            desc = " AND ".join(f"{t.s} {t.p} {t.o}" for t in step.triples)
            c = _conf(f"Find all {var} satisfying: {desc}")
            if c > tau:
                prompt = _anchor_prompt(var, step.triples, bindings)
                result = _run_value_prompt(prompt, llm)
            else:
                result = set()
                for t in step.triples:
                    for av in bound_consts:
                        env = Environment()
                        env.set("?_a", {av})
                        result |= {tr.o for tr in
                                   LLMKeyScan(TriplePattern("?_a", t.p, "?x"),
                                              env, llm, [], [])}

        elif step.kind == "chain":
            t0    = step.triples[0]
            seeds = bindings.get(t0.s, {t0.s}) if t0.s else set()
            chain_desc = " → ".join(t.p for t in step.triples)
            c = _conf(f"Follow {chain_desc} from: {', '.join(sorted(seeds)[:5])}")
            if c > tau:
                result = _run_value_prompt(_chain_cot_prompt(step.triples, seeds), llm)
            else:
                result = seeds
                for t in step.triples:
                    new = set()
                    for s in result:
                        env = Environment()
                        env.set("?_s", {s})
                        new |= {tr.o for tr in
                                LLMScan(TriplePattern("?_s", t.p, "?x"),
                                        env, gamma2, llm, tau_strategie=tau)}
                    result = new

        elif step.kind == "expand":
            t     = step.triples[0]
            seeds = (bindings.get(t.s, {t.s})
                     if not t.s.startswith("?") or t.s in bindings
                     else bindings.get(t.o, {t.o}))
            if t.is_closure:
                seed = next(iter(seeds)) if seeds else "?s"
                pat  = RecursivePattern(seed, t.p, "?x", t.operator)
                c    = LLMAtomicRecConf(pat, llm)
                if c > tau:
                    result = LLMRecScan(pat, llm, motivational=False)
                else:
                    result = LLMFixpointScan(pat, llm, gamma=gamma2)
            else:
                seed_str = ", ".join(sorted(seeds)[:5])
                c = _conf(f"Find all via '{t.p}' from: {seed_str}")
                if c > tau:
                    env = Environment()
                    env.set("?_s", seeds)
                    result = {tr.o for tr in
                              LLMSeedScan(TriplePattern("?_s", t.p, "?x"), env, llm, [], [])}
                else:
                    result = set()
                    for seed in seeds:
                        env = Environment()
                        env.set("?_s", {seed})
                        result |= {tr.o for tr in
                                   LLMKeyScan(TriplePattern("?_s", t.p, "?x"),
                                              env, llm, [], [])}
        else:
            result = set()

        if step.kind in ("expand", "chain"):
            expand_by_var.setdefault(var, []).extend(step.triples)

        if var in bindings and bindings[var]:
            bindings[var] &= result
        else:
            bindings[var] = result

        if not bindings.get(var):
            return set()

    return bindings.get(plan.target_var, set())


# ── Registry ───────────────────────────────────────────────────────────────────

MODELS = {
    "mugalois_hol": (MuGaloisHol, "µ-Galois_Hol"),
    "mugalois_dec": (MuGaloisDec, "µ-Galois_Dec"),
    "mugalois_c":   (MuGaloisC,   "µ-Galois_C"),
    "mugalois_s":   (MuGaloisS,   "µ-Galois_S"),
    "mugalois_fm":  (MuGaloisFM,  "µ-Galois_F-M"),
    "mugalois_f":   (MuGaloisF,   "µ-Galois_F"),
}