# src/mugalois/rec/mugalois_rec.py
"""
MuGaloisRec — µ-Galois orchestrator for recursive property paths (T5).

1. LLMAtomicRecConf > tau_A → LLMRecScan (single-call closure attempt)
2. Otherwise              → LLMFixpointScan (hop-by-hop, mu-RA style)

Mirrors mugalois.paths.mugalois.MuGaloisPath (T3) but takes a single
RecursivePattern instead of (s, p1, p2, t).
"""
from __future__ import annotations
from mugalois.core.types import Environment, RecursivePattern
from mugalois.rec.simple_rec import LLMRecScan, LLMAtomicRecConf
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.llm.llm_client import BaseLLM


def MuGaloisRec(
    pattern:   RecursivePattern,
    llm:       BaseLLM,
    gamma:     Environment = None,
    tau_a:     float = 0.7,
    max_depth: int = 8,
    verbose:   bool = False,
) -> set[str]:
    """
    µ-Galois recursive path orchestrator.

    1. LLMAtomicRecConf → if > tau_a → LLMRecScan (atomic, cheap)
    2. else             → LLMFixpointScan (iterative, more reliable
                            on deep/unfamiliar closures)
    """
    conf = LLMAtomicRecConf(pattern, llm)
    if verbose:
        print(f"  [µGaloisRec] AtomicRecConf={conf:.2f} tau_a={tau_a} "
              f"pattern={pattern}")

    if conf > tau_a:
        if verbose:
            print(f"  [µGaloisRec] → LLMRecScan (atomic)")
        return LLMRecScan(pattern, llm)

    if verbose:
        print(f"  [µGaloisRec] → LLMFixpointScan (iterative)")
    return LLMFixpointScan(
        pattern, llm, gamma=gamma, max_depth=max_depth, verbose=verbose,
    )