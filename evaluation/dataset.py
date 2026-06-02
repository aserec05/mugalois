"""
dataset.py
==========
Loads the scientists ground truth JSON and exposes it as set[Triple].

Extracts local names from URIs for comparison with LLM outputs.

Format (ground_truth.json)
--------------------------
{
  "T1": { "subject_uri|predicate_uri": ["object_uri", ...] },
  "T2": { "predicate_uri|object_uri":  ["subject_uri", ...] },
  "T3": { "predicate_uri":             [["subject_uri", "object_uri"], ...] }
}

Usage
-----
    from evaluation.dataset import ScientistDataset

    ds = ScientistDataset("evaluation/ground_truth/ground_truth.json")

    # T3 — both variables free (?x, birthPlace, ?y)
    triples = ds.T3("http://dbpedia.org/ontology/birthPlace")

    # T1 — subject fixed (Einstein, birthPlace, ?y)
    triples = ds.T1("http://dbpedia.org/resource/Albert_Einstein",
                    "http://dbpedia.org/ontology/birthPlace")

    # T2 — object fixed (?x, birthPlace, Ulm)
    triples = ds.T2("http://dbpedia.org/ontology/birthPlace",
                    "http://dbpedia.org/resource/Ulm")

    # Available predicates
    ds.predicates()
"""

from __future__ import annotations

import json
from pathlib import Path

from mugalois.core.types import Triple


def _local(uri: str) -> str:
    """
    Extract local name from a URI and normalize for LLM comparison.

    http://dbpedia.org/resource/Albert_Einstein  →  Albert Einstein
    http://dbpedia.org/ontology/birthPlace       →  birthPlace
    http://dbpedia.org/resource/Ulm              →  Ulm
    """
    uri = uri.strip().strip('"')
    if "/" in uri:
        local = uri.rstrip("/").split("/")[-1]
    else:
        local = uri
    # Remove parenthetical disambiguation e.g. "David_Allen_(botanist)"
    if "(" in local:
        local = local[:local.index("(")].strip("_")
    return local.replace("_", " ").strip()


class ScientistDataset:
    """
    Loads the scientists ground truth JSON.

    Parameters
    ----------
    path : str | Path
        Path to ground_truth.json
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as f:
            self._raw = json.load(f)

    # ── Public API ────────────────────────────────────────────────────────────

    def T3(self, predicate_uri: str) -> set[Triple]:
        """
        Pattern (?x, predicate, ?y) — both variables free.
        Returns all (subject, predicate, object) triples for this predicate.
        """
        pairs = self._raw.get("T3", {}).get(predicate_uri, [])
        p     = _local(predicate_uri)
        return {
            Triple(_local(s), p, _local(o))
            for s, o in pairs
        }

    def T1(self, subject_uri: str, predicate_uri: str) -> set[Triple]:
        """
        Pattern (subject, predicate, ?y) — subject fixed.
        Returns all objects for this (subject, predicate) pair.
        """
        key  = f"{subject_uri}|{predicate_uri}"
        objs = self._raw.get("T1", {}).get(key, [])
        s    = _local(subject_uri)
        p    = _local(predicate_uri)
        return {Triple(s, p, _local(o)) for o in objs}

    def T2(self, predicate_uri: str, object_uri: str) -> set[Triple]:
        """
        Pattern (?x, predicate, object) — object fixed.
        Returns all subjects for this (predicate, object) pair.
        """
        key   = f"{predicate_uri}|{object_uri}"
        subjs = self._raw.get("T2", {}).get(key, [])
        p     = _local(predicate_uri)
        o     = _local(object_uri)
        return {Triple(_local(s), p, o) for s in subjs}

    def predicates(self) -> list[str]:
        """Return all predicate URIs available in T3."""
        return list(self._raw.get("T3", {}).keys())

    def subjects(self, predicate_uri: str) -> set[str]:
        """Return all distinct subject local names for a predicate (from T3)."""
        pairs = self._raw.get("T3", {}).get(predicate_uri, [])
        return {_local(s) for s, _ in pairs}

    def objects(self, predicate_uri: str) -> set[str]:
        """Return all distinct object local names for a predicate (from T3)."""
        pairs = self._raw.get("T3", {}).get(predicate_uri, [])
        return {_local(o) for _, o in pairs}

    def stats(self) -> None:
        """Print a summary of the dataset."""
        t1 = self._raw.get("T1", {})
        t2 = self._raw.get("T2", {})
        t3 = self._raw.get("T3", {})
        print(f"\n  Dataset : {self.path.name}")
        print(f"  T1 keys : {len(t1)}")
        print(f"  T2 keys : {len(t2)}")
        print(f"  T3 predicates : {len(t3)}")
        print(f"\n  {'Predicate':<30} {'T3 triples':>10}")
        print("  " + "-" * 42)
        for pred_uri, pairs in sorted(t3.items(), key=lambda x: -len(x[1])):
            print(f"  {_local(pred_uri):<30} {len(pairs):>10}")
        print()