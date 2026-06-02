"""
Metrics for µ-Galois evaluation.
Inspired by Galois (Satriani et al., SIGMOD 2025).

Metrics
-------
- Precision     : fraction of returned triples that are correct
- Recall        : fraction of expected triples that were found
- F1-Triple     : harmonic mean of Precision and Recall (main metric)
- Cardinality   : ratio of result sizes  min(|actual|, |expected|) / max(...)
- AVG-Score     : average of F1-Triple, Cardinality  (summary metric)

All metrics return values in [0, 1]. Higher is better.

Example
-------
    from metrics import Metrics

    actual = {
        Triple("Einstein", "birthPlace", "Ulm"),
        Triple("Curie",    "birthPlace", "Paris"),   # wrong
    }
    expected = {
        Triple("Einstein", "birthPlace", "Ulm"),
        Triple("Curie",    "birthPlace", "Warsaw"),
    }

    scores = Metrics.compute(actual, expected)
    print(scores)
    # MetricScores(precision=0.5, recall=0.5, f1_triple=0.5, cardinality=1.0, avg_score=0.75)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from mugalois.core.types import Triple




def _normalize(value: str) -> str:
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    if "#" in value:
        value = value.split("#")[-1]
    if ":" in value:                          
        value = value.split(":")[-1]
    return value.replace("_", " ").lower().strip()




def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False


def _values_match(
    actual: str,
    expected: str,
    similarity_threshold: float = 0.10,
    numeric_tolerance: float = 0.10,
) -> bool:
    """
    Return True if two values are considered equal.

    Rules (same as Galois):
    - Normalize both values first.
    - Numeric  : allow ± numeric_tolerance relative difference.
    - String   : allow edit distance ≤ similarity_threshold * len(expected).
    """
    a = _normalize(actual)
    e = _normalize(expected)

    if a == e:
        return True

    if _is_numeric(a) and _is_numeric(e):
        va = float(a.replace(",", "").replace("%", ""))
        ve = float(e.replace(",", "").replace("%", ""))
        if ve == 0:
            return va == 0
        return abs(va - ve) / abs(ve) <= numeric_tolerance

    max_dist = max(1, math.ceil(len(e) * similarity_threshold))
    return _edit_distance(a, e) <= max_dist



def _triples_match(
    actual: Triple,
    expected: Triple,
    similarity_threshold: float,
    numeric_tolerance: float,
) -> bool:
    """
    Return True if two triples match.
    Both s and o must match — p is ignored (it is always fixed in our patterns).
    """
    return _values_match(
        actual.s, expected.s, similarity_threshold, numeric_tolerance,
    ) and _values_match(
        actual.o, expected.o, similarity_threshold, numeric_tolerance,
    )


#  dataclass 

@dataclass
class MetricScores:
    precision:   float
    recall:      float
    f1_triple:   float
    cardinality: float
    avg_score:   float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Precision={self.precision:.4f}  "
            f"Recall={self.recall:.4f}  "
            f"F1-Triple={self.f1_triple:.4f}  "
            f"Cardinality={self.cardinality:.4f}  "
            f"AVG-Score={self.avg_score:.4f}"
        )



class Metrics:
    """
    Computes evaluation metrics for RDF triple extraction.
    All methods are static — no instantiation needed.

    Parameters (all methods)
    ------------------------
    actual   : set[Triple]
    expected : set[Triple]
    """

    @staticmethod
    def compute(
        actual: set[Triple],
        expected: set[Triple],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> MetricScores:
        p    = Metrics.precision(actual, expected, similarity_threshold, numeric_tolerance)
        r    = Metrics.recall(actual, expected, similarity_threshold, numeric_tolerance)
        f1   = Metrics.f1_triple(actual, expected, similarity_threshold, numeric_tolerance)
        card = Metrics.cardinality(actual, expected)
        avg  = (f1 + card) / 2.0

        return MetricScores(
            precision=round(p, 4),
            recall=round(r, 4),
            f1_triple=round(f1, 4),
            cardinality=round(card, 4),
            avg_score=round(avg, 4),
        )

    @staticmethod
    def precision(
        actual: set[Triple],
        expected: set[Triple],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        if not actual:
            return 1.0 if not expected else 0.0
        tp = _count_true_positives(actual, expected, similarity_threshold, numeric_tolerance)
        return tp / len(actual)

    @staticmethod
    def recall(
        actual: set[Triple],
        expected: set[Triple],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        if not expected:
            return 1.0
        tp = _count_true_positives(actual, expected, similarity_threshold, numeric_tolerance)
        return tp / len(expected)

    @staticmethod
    def f1_triple(
        actual: set[Triple],
        expected: set[Triple],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        p = Metrics.precision(actual, expected, similarity_threshold, numeric_tolerance)
        r = Metrics.recall(actual, expected, similarity_threshold, numeric_tolerance)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def cardinality(actual: set[Triple], expected: set[Triple]) -> float:
        n_a = len(actual)
        n_e = len(expected)
        if n_a == 0 and n_e == 0:
            return 1.0
        if n_a == 0 or n_e == 0:
            return 0.0
        return min(n_a, n_e) / max(n_a, n_e)


# ─── Internal helper ──────────────────────────────────────────────────────────

def _count_true_positives(
    actual: set[Triple],
    expected: set[Triple],
    similarity_threshold: float,
    numeric_tolerance: float,
) -> int:
    """Greedy one-to-one matching."""
    expected_list = list(expected)
    matched = [False] * len(expected_list)
    tp = 0
    for act in actual:
        for i, exp in enumerate(expected_list):
            if not matched[i] and _triples_match(
                act, exp, similarity_threshold, numeric_tolerance
            ):
                matched[i] = True
                tp += 1
                break
    return tp