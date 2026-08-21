# src/mugalois/rec/mugalois_rec.py
"""
MuGaloisRec — µ-Galois orchestrator for recursive property paths (T5).
Decision tree:
  structure = LLMStructureDetect()   [1 focused call]
  conf      = LLMAtomicRecConf()     [1 focused call]
  conf > τ_high:
    → LLMRecScan(motivational=motivational)
  DAG + conf > τ_low:
    → LLMRecScan(motivational=motivational)
  DAG + conf ≤ τ_low:
    → Fixpoint (last resort on DAG)
  Chain + conf > τ_low:
    → LLMRecScan(motivational=motivational)
  Chain + conf ≤ τ_low:
    size = LLMEstimateRecSize()
    size ≤ CARD_MAX → Fixpoint
    size > CARD_MAX → LLMRecScan(motivational=False)
"""
from __future__ import annotations
from mugalois.core.types import Environment, RecursivePattern
from mugalois.core.prompts import build_messages
from mugalois.rec.simple_rec import LLMRecScan, LLMAtomicRecConf, LLMEstimateRecSize
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.llm.llm_client import BaseLLM

TAU_HIGH = 0.50
TAU_LOW  = 0.35
CARD_MAX = 15


def LLMStructureDetect(pattern: RecursivePattern, llm: BaseLLM) -> str:
    s = pattern.s if not pattern.s_is_var() else pattern.o
    prompt = (
        f"For the relation '{pattern.p}' starting from {s}:\n\n"
        f"Does this typically form:\n"
        f"- a LINEAR CHAIN: each entity has at most one direct successor, or\n"
        f"- a BRANCHING DAG: one entity can have multiple direct successors?\n\n"
        f"Answer with a single word: 'chain' or 'dag'. No explanation."
    )
    resp = llm.chat(build_messages(prompt))
    text = resp.text.strip().lower()
    return "dag" if "dag" in text else "chain"


def MuGaloisRec(
    pattern:      RecursivePattern,
    llm:          BaseLLM,
    gamma:        Environment = None,
    tau_high:     float = TAU_HIGH,
    tau_low:      float = TAU_LOW,
    max_depth:    int   = 50,
    verbose:      bool  = False,
    motivational: bool  = True,    # ← nouveau paramètre
) -> set[str]:
    """µ-Galois orchestrator for recursive paths (T5).
    
    motivational=False disables the follow-up prompt in LLMRecScan.
    Used by µ-Galois_F-M to isolate the contribution of motivational prompting.
    """
    # ── Call 1 : structure ────────────────────────────────────────────
    structure = LLMStructureDetect(pattern, llm)
    # ── Call 2 : confidence ───────────────────────────────────────────
    conf = LLMAtomicRecConf(pattern, llm)
    if verbose:
        print(f"[MuGaloisRec] structure={structure} conf={conf:.2f} "
              f"τ_high={tau_high} τ_low={tau_low} motivational={motivational}")
    # ── 1. High confidence ────────────────────────────────────────────
    if conf > tau_high:
        if verbose: print(f"  → Holistic, motivational={motivational}")
        return LLMRecScan(pattern, llm, motivational=motivational)
    # ── 2. DAG branch ─────────────────────────────────────────────────
    if structure == "dag":
        if conf > tau_low:
            if verbose: print(f"  → Holistic + motivational={motivational} (DAG)")
            return LLMRecScan(pattern, llm, motivational=motivational)
        else:
            if verbose: print(f"  → Fixpoint (DAG, conf ≤ τ_low)")
            return LLMFixpointScan(pattern, llm, gamma=gamma,
                                   max_depth=max_depth, verbose=verbose)
    # ── 3. Chain + moderate confidence ────────────────────────────────
    if conf > tau_low:
        if verbose: print(f"  → Holistic + motivational={motivational} (chain)")
        return LLMRecScan(pattern, llm, motivational=motivational)
    # ── 4. Chain + low confidence → size decides ──────────────────────
    size = LLMEstimateRecSize(pattern, llm)
    if verbose: print(f"  size={size} CARD_MAX={CARD_MAX}")
    if 0 < size <= CARD_MAX:
        if verbose: print(f"  → Fixpoint (chain, size ≤ CARD_MAX)")
        return LLMFixpointScan(pattern, llm, gamma=gamma,
                               max_depth=max_depth, verbose=verbose)
    else:
        if verbose: print(f"  → NL, motivational=False (chain, size > CARD_MAX)")
        return LLMRecScan(pattern, llm, motivational=False)