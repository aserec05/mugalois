from __future__ import annotations

from typing import Union
from mugalois.core.types import Triple
from evaluation.metrics import Metrics, MetricScores, AggregatedScores


class Evaluator:
    """
    Compares actual output against expected (ground truth) for one run.

    Parameters
    ----------
    actual   : set[str] or set[Triple]
    expected : set[str] or set[Triple]
    mode     : "values" or "triples"
    similarity_threshold : float (default 0.30)
    """

    def __init__(
        self,
        actual:   Union[set[str], set[Triple]],
        expected: Union[set[str], set[Triple]],
        mode: str = "values",
        similarity_threshold: float = 0.30,
    ) -> None:
        self.actual               = actual
        self.expected             = expected
        self.mode                 = mode
        self.similarity_threshold = similarity_threshold

    def evaluate(self) -> MetricScores:
        return Metrics.compute(
            self.actual,
            self.expected,
            mode=self.mode,
            similarity_threshold=self.similarity_threshold,
        )