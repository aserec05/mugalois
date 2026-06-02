"""
Evaluator for µ-Galois.

Compares actual triples against expected triples (ground truth).
Both inputs are set[Triple].

Example
-------
    from evaluator import Evaluator

    actual   = {Triple("Einstein", "birthPlace", "Ulm")}
    expected = {Triple("Einstein", "birthPlace", "Ulm"),
                Triple("Curie",    "birthPlace", "Warsaw")}

    ev = Evaluator(actual, expected)
    scores = ev.evaluate()
    print(scores)

    # From JSON files
    ev = Evaluator.from_files("actual.json", "expected.json")
    scores = ev.evaluate()
"""

from __future__ import annotations

import json
from pathlib import Path

from mugalois.core.types import Triple
from evaluation.metrics import Metrics, MetricScores


class Evaluator:
    """
    Compares actual triples against expected triples.

    Parameters
    ----------
    actual : set[Triple]
    expected : set[Triple]
    similarity_threshold : float  (default 0.10)
    numeric_tolerance : float     (default 0.10)
    """

    def __init__(
        self,
        actual: set[Triple],
        expected: set[Triple],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> None:
        self.actual               = set(actual)
        self.expected             = set(expected)
        self.similarity_threshold = similarity_threshold
        self.numeric_tolerance    = numeric_tolerance



    @classmethod
    def from_files(
        cls,
        actual_path: str | Path,
        expected_path: str | Path,
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> "Evaluator":
        """
        Load from two JSON files.
        Each file must contain a JSON array: [{"s": ..., "p": ..., "o": ...}]
        """
        actual   = _load_triples(actual_path)
        expected = _load_triples(expected_path)
        return cls(actual, expected, similarity_threshold, numeric_tolerance)

    @classmethod
    def from_dicts(
        cls,
        data: dict,
        actual_key: str = "actual",
        expected_key: str = "expected",
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> "Evaluator":
        """
        Load from a single dict with two keys.
        {
            "actual":   [{"s":..., "p":..., "o":...}],
            "expected": [{"s":..., "p":..., "o":...}]
        }
        """
        if actual_key not in data:
            raise ValueError(f"Key '{actual_key}' not found in dict.")
        if expected_key not in data:
            raise ValueError(f"Key '{expected_key}' not found in dict.")

        return cls(
            _dicts_to_triples(data[actual_key]),
            _dicts_to_triples(data[expected_key]),
            similarity_threshold,
            numeric_tolerance,
        )


    def evaluate(self) -> MetricScores:
        return Metrics.compute(
            self.actual,
            self.expected,
            similarity_threshold=self.similarity_threshold,
            numeric_tolerance=self.numeric_tolerance,
        )

    def summary(self) -> dict:
        """Plain dict with counts and scores — ready to serialize to JSON."""
        scores = self.evaluate()
        return {
            "n_actual":   len(self.actual),
            "n_expected": len(self.expected),
            **scores.to_dict(),
        }

    def save_summary(self, path: str | Path) -> None:
        """Save the evaluation summary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)



def _dicts_to_triples(data: list[dict]) -> set[Triple]:
    return {Triple(t["s"], t["p"], t["o"]) for t in data}


def _load_triples(path: str | Path) -> set[Triple]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"'{path}' must contain a JSON array.")
    return _dicts_to_triples(data)