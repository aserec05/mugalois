# src/mugalois/paths/mugalois.py
"""
MuGaloisPath — full µ-Galois orchestrator for two-hop paths.

1. LLMSimpleConf > tau_simple → LLMSimpleScan
2. LLMDirectionConf → left/right/join → GD/DG/Join
3. Each scan uses motivational closure for better recall
"""
from __future__ import annotations
from mugalois.core.types import Environment
from mugalois.paths.gd import scan_gd
from mugalois.paths.dg import scan_dg
from mugalois.paths.join import scan_join
from mugalois.paths.simple import LLMSimpleScan, LLMSimpleConf, LLMDirectionConf
from mugalois.llm.llm_client import BaseLLM


def MuGaloisPath(
    s:           str,
    p1:          str,
    p2:          str,
    t:           str,
    llm:         BaseLLM,
    gamma:       Environment = None,
    tau_simple:  float = 0.8,
    max_iter:    int   = 5,
    verbose:     bool  = False,
    use_closure: bool  = True,
) -> set[str]:
    """
    µ-Galois path orchestrator.

    1. LLMSimpleConf → if > tau_simple → SimpleScan
    2. LLMDirectionConf → left/right/join → GD/DG/Join
    3. use_closure=True → motivational closure on hop 1 for better recall
    """
    from mugalois.closures.motivational import MotivationalClosure
    from mugalois.closures.pipeline import ClosurePipeline

    conf = LLMSimpleConf(s, p1, p2, t, llm)
    if verbose:
        print(f"  [µGalois] SimpleConf={conf:.2f} tau={tau_simple}")

    if conf > tau_simple:
        if verbose:
            print(f"  [µGalois] → SimpleScan")
        return LLMSimpleScan(s, p1, p2, t, llm)

    direction = LLMDirectionConf(s, p1, p2, t, llm)
    if verbose:
        print(f"  [µGalois] → direction={direction}")

    # Build closure pipeline for hop 1
    pipeline = None
    if use_closure:
        pipeline = ClosurePipeline(
            [MotivationalClosure()], n_break=2, max_iter=max_iter + 3
        )

    if direction == "left":
        return scan_gd(s, p1, p2, t, llm, gamma=gamma,
                       pipeline=pipeline, max_iter=max_iter)
    elif direction == "right":
        return scan_dg(s, p1, p2, t, llm, gamma=gamma,
                       pipeline=pipeline, max_iter=max_iter)
    else:
        return scan_join(s, p1, p2, t, llm, gamma=gamma, max_iter=max_iter)