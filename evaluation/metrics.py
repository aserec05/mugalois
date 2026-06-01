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

    actual = [
        {"s": "Einstein", "p": "birthPlace", "o": "Ulm"},
        {"s": "Curie",    "p": "birthPlace", "o": "Paris"},   # wrong
    ]
    expected = [
        {"s": "Einstein", "p": "birthPlace", "o": "Ulm"},
        {"s": "Curie",    "p": "birthPlace", "o": "Warsaw"},
    ]

    scores = Metrics.compute(actual, expected)
    print(scores)
    # MetricScores(precision=0.5, recall=0.5, f1_triple=0.5, cardinality=1.0, avg_score=0.75)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


# ─── Normalisation ────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    """
    Normalize a cell value for comparison.
    - Strips URI prefixes (keeps local name only)
    - Lowercases
    - Replaces underscores with spaces
    """
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    if "#" in value:
        value = value.split("#")[-1]
    return value.replace("_", " ").lower().strip()


# ─── String similarity ────────────────────────────────────────────────────────

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

    # Numeric comparison
    if _is_numeric(a) and _is_numeric(e):
        va = float(a.replace(",", "").replace("%", ""))
        ve = float(e.replace(",", "").replace("%", ""))
        if ve == 0:
            return va == 0
        return abs(va - ve) / abs(ve) <= numeric_tolerance

    # String similarity
    max_dist = max(1, math.ceil(len(e) * similarity_threshold))
    return _edit_distance(a, e) <= max_dist


# ─── Triple matching ──────────────────────────────────────────────────────────

def _triples_match(
    actual: dict,
    expected: dict,
    similarity_threshold: float,
    numeric_tolerance: float,
) -> bool:
    """
    Return True if two triples match.
    Both s and o must match — p is ignored (it is always fixed in our patterns).
    """
    return _values_match(
        actual.get("s", ""), expected.get("s", ""),
        similarity_threshold, numeric_tolerance,
    ) and _values_match(
        actual.get("o", ""), expected.get("o", ""),
        similarity_threshold, numeric_tolerance,
    )


# ─── MetricScores dataclass ───────────────────────────────────────────────────

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


# ─── Metrics class ────────────────────────────────────────────────────────────

class Metrics:
    """
    Computes evaluation metrics for RDF triple extraction.

    All methods are static — no instantiation needed.

    Parameters (all methods)
    ------------------------
    actual : list[dict]
        Triples returned by µ-Galois. Each dict has keys "s", "p", "o".
    expected : list[dict]
        Ground-truth triples. Same format.
    similarity_threshold : float
        Edit-distance tolerance as fraction of expected value length (default 0.10).
    numeric_tolerance : float
        Relative tolerance for numeric values (default 0.10).
    """

    @staticmethod
    def compute(
        actual: list[dict],
        expected: list[dict],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> MetricScores:
        """
        Compute all metrics and return a MetricScores dataclass.
        """
        p    = Metrics.precision(actual, expected, similarity_threshold, numeric_tolerance)
        r    = Metrics.recall(actual, expected, similarity_threshold, numeric_tolerance)
        f1   = Metrics.f1_triple(actual, expected, similarity_threshold, numeric_tolerance)
        card = Metrics.cardinality(actual, expected)
        # avg computed on raw values before rounding to avoid precision loss
        avg  = (f1 + card) / 2.0

        return MetricScores(
            precision=round(p, 4),
            recall=round(r, 4),
            f1_triple=round(f1, 4),
            cardinality=round(card, 4),
            avg_score=round(avg, 4),
        )

    # ── Precision ─────────────────────────────────────────────────────────────

    @staticmethod
    def precision(
        actual: list[dict],
        expected: list[dict],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        """
        Fraction of returned triples that are correct.

        precision = |actual ∩ expected| / |actual|
        """
        if not actual:
            return 1.0 if not expected else 0.0

        tp = _count_true_positives(actual, expected, similarity_threshold, numeric_tolerance)
        return tp / len(actual)

    # ── Recall ────────────────────────────────────────────────────────────────

    @staticmethod
    def recall(
        actual: list[dict],
        expected: list[dict],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        """
        Fraction of expected triples that were found.

        recall = |actual ∩ expected| / |expected|
        """
        if not expected:
            return 1.0

        tp = _count_true_positives(actual, expected, similarity_threshold, numeric_tolerance)
        return tp / len(expected)

    # ── F1-Triple ─────────────────────────────────────────────────────────────

    @staticmethod
    def f1_triple(
        actual: list[dict],
        expected: list[dict],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> float:
        """
        Harmonic mean of Precision and Recall at the triple level.

        A triple counts as a true positive only if both s and o match.
        """
        p = Metrics.precision(actual, expected, similarity_threshold, numeric_tolerance)
        r = Metrics.recall(actual, expected, similarity_threshold, numeric_tolerance)

        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    # ── Cardinality ───────────────────────────────────────────────────────────

    @staticmethod
    def cardinality(actual: list[dict], expected: list[dict]) -> float:
        """
        Ratio of result sizes.

        cardinality = min(|actual|, |expected|) / max(|actual|, |expected|)

        Returns 1.0 if both are empty, 0.0 if one is empty and the other is not.
        """
        n_a = len(actual)
        n_e = len(expected)

        if n_a == 0 and n_e == 0:
            return 1.0
        if n_a == 0 or n_e == 0:
            return 0.0
        return min(n_a, n_e) / max(n_a, n_e)


# ─── Internal helper ──────────────────────────────────────────────────────────

def _count_true_positives(
    actual: list[dict],
    expected: list[dict],
    similarity_threshold: float,
    numeric_tolerance: float,
) -> int:
    """
    Count true positives with greedy one-to-one matching.
    Each expected triple can only be matched once.
    """
    matched = [False] * len(expected)
    tp = 0
    for act in actual:
        for i, exp in enumerate(expected):
            if not matched[i] and _triples_match(
                act, exp, similarity_threshold, numeric_tolerance
            ):
                matched[i] = True
                tp += 1
                break
    return tp