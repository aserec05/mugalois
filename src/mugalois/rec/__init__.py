# src/mugalois/rec/__init__.py
"""
mugalois.rec — recursive property path evaluation (T5).

p+ / p* property paths, evaluated either atomically (single LLM call,
LLMRecScan) when LLMAtomicRecConf is high, or hop-by-hop via a fixpoint
iteration (LLMFixpointScan) otherwise — mirroring the mu-RA fixpoint
operator U_{i+1} = U_i ∪ [[phi]][X/U_i].
"""
from mugalois.rec.simple_rec import LLMRecScan, LLMAtomicRecConf
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.rec.mugalois_rec import MuGaloisRec

__all__ = [
    "LLMRecScan",
    "LLMAtomicRecConf",
    "LLMFixpointScan",
    "MuGaloisRec",
]
