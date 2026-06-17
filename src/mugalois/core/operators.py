# src/mugalois/core/operators.py
from __future__ import annotations
from mugalois.core.types import Triple


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    """Extract local name and normalize for comparison."""
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    if "#" in value:
        value = value.split("#")[-1]
    if ":" in value:
        value = value.split(":")[-1]
    return value.replace("_", " ").lower().strip()


# ── Comparison ────────────────────────────────────────────────────────────────

def _apply(value: str, op: str, threshold: str) -> bool:
    """
    Apply a single comparison operator between value and threshold.
    Tries numeric comparison first, falls back to lexicographic.

    Operators: =, !=, >, <, >=, <=
    """
    try:
        v = float(str(value).replace(",", ""))
        t = float(threshold.replace(",", ""))
        if op == "=":  return v == t
        if op == ">":  return v > t
        if op == "<":  return v < t
        if op == ">=": return v >= t
        if op == "<=": return v <= t
        if op == "!=": return v != t
    except ValueError:
        if op == "=":  return value == threshold
        if op == "!=": return value != threshold
        if op == ">":  return value > threshold
        if op == "<":  return value < threshold
        if op == ">=": return value >= threshold
        if op == "<=": return value <= threshold
    return False


# ── Triple set operators ──────────────────────────────────────────────────────

def filter_by_subject(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Keep triples from T1 whose subject appears as subject in T2."""
    subjects_T2 = {_normalize(t.s) for t in T2}
    return {t for t in T1 if _normalize(t.s) in subjects_T2}


def filter_by_object(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Keep triples from T1 whose object appears as object in T2."""
    objects_T2 = {_normalize(t.o) for t in T2}
    return {t for t in T1 if _normalize(t.o) in objects_T2}


def filter_by_subject_object(T1: set[Triple], T2: set[Triple]) -> set[Triple]:
    """Keep triples from T1 whose subject appears as object in T2."""
    objects_T2 = {_normalize(t.o) for t in T2}
    return {t for t in T1 if _normalize(t.s) in objects_T2}


# ── Path join ─────────────────────────────────────────────────────────────────

def path_join(T1: set[Triple], T2: set[Triple]) -> set[str]:
    """
    Join T1 and T2 on the shared intermediate variable ?b.

    T1 : (s, p1, ?b)  — ?b is the object
    T2 : (?b, p2, t)  — ?b is the subject

    Returns the set of ?b values that appear in both T1.o and T2.s.
    These are the valid intermediate nodes connecting s to t.
    """
    b_from_T1 = {_normalize(t.o) for t in T1}
    b_from_T2 = {_normalize(t.s) for t in T2}
    return b_from_T1 & b_from_T2