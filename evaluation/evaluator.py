"""
Evaluator for µ-Galois.

Compares actual triples against expected triples (ground truth).
Both inputs follow the same JSON schema.


Example
-------
    from evaluator import Evaluator

    actual = [
        {"s": "Einstein", "p": "birthPlace", "o": "Ulm"},
        {"s": "Curie",    "p": "birthPlace", "o": "Paris"},
    ]
    expected = [
        {"s": "Einstein", "p": "birthPlace", "o": "Ulm"},
        {"s": "Curie",    "p": "birthPlace", "o": "Warsaw"},
    ]

    ev = Evaluator(actual, expected)
    scores = ev.evaluate()
    print(scores)

    # From files
    ev = Evaluator.from_files("actual.json", "expected.json")
    scores = ev.evaluate()
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import Metrics, MetricScores


# let's type check first

def _validate(triples: list, label: str) -> None:
    """
    Raise ValueError if the list does not follow the triple schema.
    Each item must be a dict with at least keys "s" and "o".
    "p" is optional but recommended.
    """
    if not isinstance(triples, list):
        raise ValueError(f"'{label}' must be a list, got {type(triples).__name__}")

    for i, t in enumerate(triples):
        if not isinstance(t, dict):
            raise ValueError(
                f"'{label}[{i}]' must be a dict, got {type(t).__name__}"
            )
        for key in ("s", "o"):
            if key not in t:
                raise ValueError(
                    f"'{label}[{i}]' is missing required key '{key}'. "
                    f"Got keys: {list(t.keys())}"
                )
            if not isinstance(t[key], str):
                raise ValueError(
                    f"'{label}[{i}][{key}]' must be a string, "
                    f"got {type(t[key]).__name__}"
                )


class Evaluator:
    """
    Compares actual triples against expected triples.

    Parameters
    ----------
    actual : list[dict]
        Triples returned by a strategy. Schema: [{"s": ..., "p": ..., "o": ...}]
    expected : list[dict]
        Ground-truth triples. Same schema.
    similarity_threshold : float
        Edit-distance tolerance as fraction of expected value length (default 0.10).
    numeric_tolerance : float
        Relative tolerance for numeric values (default 0.10).
    """

    def __init__(
        self,
        actual: list[dict],
        expected: list[dict],
        similarity_threshold: float = 0.10,
        numeric_tolerance: float = 0.10,
    ) -> None:
        _validate(actual,   "actual")
        _validate(expected, "expected")

        self.actual               = actual
        self.expected             = expected
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
        Load actual and expected triples from two JSON files.
        Each file must contain a JSON array of triple objects.
        """
        actual   = _load_json(actual_path)
        expected = _load_json(expected_path)
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
        Load actual and expected triples from a single dict with two keys.

        Useful when both lists are stored in the same JSON file:
        {
            "actual":   [...],
            "expected": [...]
        }
        """
        if actual_key not in data:
            raise ValueError(f"Key '{actual_key}' not found in dict.")
        if expected_key not in data:
            raise ValueError(f"Key '{expected_key}' not found in dict.")

        return cls(
            data[actual_key],
            data[expected_key],
            similarity_threshold,
            numeric_tolerance,
        )

    def evaluate(self) -> MetricScores:
        """
        Compute all metrics and return a MetricScores object.
        """
        return Metrics.compute(
            self.actual,
            self.expected,
            similarity_threshold=self.similarity_threshold,
            numeric_tolerance=self.numeric_tolerance,
        )

    def summary(self) -> dict:
        """
        Return a plain dict with counts and scores — ready to serialize to JSON.
        """
        scores = self.evaluate()
        return {
            "n_actual":   len(self.actual),
            "n_expected": len(self.expected),
            **scores.to_dict(),
        }

    def save_summary(self, path: str | Path) -> None:
        """
        Save the evaluation summary to a JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)


def _load_json(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"'{path}' must contain a JSON array, got {type(data).__name__}"
        )
    return data