from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, asdict, field
from typing import Union

from mugalois.core.types import Triple

_re_paren = re.compile(r"\s*\(.*?\)")


def _normalize(value: str) -> str:
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    if "#" in value:
        value = value.split("#")[-1]
    if ":" in value:
        value = value.split(":")[-1]
    value = _re_paren.sub("", value)
    return value.replace("_", " ").lower().strip()


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _values_match(actual: str, expected: str,
                  similarity_threshold: float = 0.30) -> bool:
    a = _normalize(actual)
    e = _normalize(expected)
    if a == e:
        return True
    if a in e or e in a:
        return True
    max_dist = max(1, math.ceil(len(e) * similarity_threshold))
    return _edit_distance(a, e) <= max_dist


def _triple_match(actual: Triple, expected: Triple,
                  similarity_threshold: float) -> bool:
    return (
        _values_match(actual.s, expected.s, similarity_threshold)
        and _values_match(actual.o, expected.o, similarity_threshold)
    )


def _count_tp_values(actual, expected, similarity_threshold):
    expected_list = list(expected)
    matched = [False] * len(expected_list)
    tp = 0
    for a in actual:
        for i, e in enumerate(expected_list):
            if not matched[i] and _values_match(a, e, similarity_threshold):
                matched[i] = True
                tp += 1
                break
    return tp


def _count_tp_triples(actual, expected, similarity_threshold):
    expected_list = list(expected)
    matched = [False] * len(expected_list)
    tp = 0
    for a in actual:
        for i, e in enumerate(expected_list):
            if not matched[i] and _triple_match(a, e, similarity_threshold):
                matched[i] = True
                tp += 1
                break
    return tp


@dataclass
class MetricScores:
    precision: float
    recall:    float
    f1:        float
    n_actual:  int   = 0
    time_s:    float = 0.0
    tokens:    int   = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"P={self.precision:.4f} R={self.recall:.4f} "
                f"F1={self.f1:.4f} t={self.time_s:.1f}s tok={self.tokens}")


@dataclass
class AggregatedScores:
    precision_mean: float
    precision_std:  float
    recall_mean:    float
    recall_std:     float
    f1_mean:        float
    f1_std:         float
    n_runs:         int
    n_generated:    float = 0.0
    time_mean:      float = 0.0
    time_std:       float = 0.0
    tokens_mean:    float = 0.0
    tokens_std:     float = 0.0

    @classmethod
    def from_runs(cls, scores: list[MetricScores]) -> "AggregatedScores":
        if not scores:
            raise ValueError("No scores to aggregate.")
        n = len(scores)
        std_fn = statistics.stdev if n > 1 else lambda _: 0.0
        ps = [s.precision for s in scores]
        rs = [s.recall    for s in scores]
        fs = [s.f1        for s in scores]
        gs = [s.n_actual  for s in scores]
        ts = [s.time_s    for s in scores]
        tk = [float(s.tokens) for s in scores]
        return cls(
            precision_mean=round(statistics.mean(ps), 4),
            precision_std =round(std_fn(ps), 4),
            recall_mean   =round(statistics.mean(rs), 4),
            recall_std    =round(std_fn(rs), 4),
            f1_mean       =round(statistics.mean(fs), 4),
            f1_std        =round(std_fn(fs), 4),
            n_runs        =n,
            n_generated   =round(statistics.mean(gs), 1),
            time_mean     =round(statistics.mean(ts), 2),
            time_std      =round(std_fn(ts), 2),
            tokens_mean   =round(statistics.mean(tk), 0),
            tokens_std    =round(std_fn(tk), 0),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"F1={self.f1_mean:.4f}±{self.f1_std:.4f} "
                f"t={self.time_mean:.1f}s tok={self.tokens_mean:.0f}")


class Metrics:
    @staticmethod
    def compute(
        actual:   Union[set, set],
        expected: Union[set, set],
        mode: str = "values",
        similarity_threshold: float = 0.30,
    ) -> MetricScores:
        if mode == "values":
            tp = _count_tp_values(actual, expected, similarity_threshold)
        elif mode == "triples":
            tp = _count_tp_triples(actual, expected, similarity_threshold)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        n_actual   = len(actual)
        n_expected = len(expected)

        precision = (tp / n_actual)   if n_actual   else (1.0 if not expected else 0.0)
        recall    = (tp / n_expected) if n_expected  else 1.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return MetricScores(
            precision=round(precision, 4),
            recall   =round(recall,    4),
            f1       =round(f1,        4),
            n_actual =n_actual,
        )