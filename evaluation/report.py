
"""
Report for µ-Galois evaluation.
 
Displays and exports evaluation results for one or several strategies.
 
Example
-------
    from report import Report
 
    report = Report(predicate="birthPlace")
    report.add("TableScan",   evaluator_1.evaluate())
    report.add("KeyCrank",    evaluator_2.evaluate())
    report.add("SeedCrank",   evaluator_3.evaluate())
    report.add("TripletScan", evaluator_4.evaluate())
 
    report.print_table()
    report.save_json("results/birthplace.json")
    report.save_csv("results/birthplace.csv")
"""
 
from __future__ import annotations
 
import csv
import json
from datetime import datetime
from pathlib import Path
 
from evaluation.metrics import MetricScores
 
 
METRIC_KEYS = ["precision", "recall", "f1_triple", "cardinality", "avg_score"]
 
METRIC_LABELS = {
    "precision":   "Precision",
    "recall":      "Recall",
    "f1_triple":   "F1-Triple",
    "cardinality": "Cardinality",
    "avg_score":   "AVG-Score",
}
 
COL_WIDTH = 12
 
 

 
class Report:
    """
    Collects MetricScores from multiple strategies and displays/exports them.
 
    Parameters
    ----------
    predicate : str
        The predicate being evaluated, e.g. "birthPlace". Used in display and export.
    """
 
    def __init__(self, predicate: str = "") -> None:
        self.predicate  = predicate
        self.timestamp  = datetime.now().isoformat(timespec="seconds")
        self._results: dict[str, MetricScores] = {}
 
 
    def add(self, strategy: str, scores: MetricScores) -> "Report":
        """
        Register the MetricScores for one strategy.
 
        Parameters
        ----------
        strategy : str
            Human-readable strategy name, e.g. "TableScan".
        scores : MetricScores
            Output of Evaluator.evaluate().
        """
        if not isinstance(scores, MetricScores):
            raise ValueError(
                f"Expected MetricScores, got {type(scores).__name__}. "
                "Call Evaluator.evaluate() first."
            )
        self._results[strategy] = scores
        return self
 
    def strategies(self) -> list[str]:
        """Return the list of registered strategy names."""
        return list(self._results.keys())
 
    # DISPLAY PART
 
    def print_table(self) -> None:
        """Print a summary table to stdout."""
        if not self._results:
            print("No results to display.")
            return
 
        strat_w = max(len(s) for s in self._results) + 4
        header  = f"{'Strategy':<{strat_w}}" + "".join(
            f"{METRIC_LABELS[m]:>{COL_WIDTH}}" for m in METRIC_KEYS
        )
        sep = "─" * len(header)
 
        best_avg = max(s.avg_score for s in self._results.values())
 
        print()
        print(f"  Predicate : {self.predicate}")
        print(f"  Date      : {self.timestamp}")
        print()
        print(sep)
        print(header)
        print(sep)
 
        for strategy, scores in self._results.items():
            marker = " ✓" if scores.avg_score == best_avg else "  "
            d = scores.to_dict()
            row = f"{strategy + marker:<{strat_w}}" + "".join(
                f"{d[m]:{COL_WIDTH}.4f}" for m in METRIC_KEYS
            )
            print(row)
 
        print(sep)
        print("  ✓ Best AVG-Score")
        print()
 
    # EXPORT PART
 
    def save_json(self, path: str | Path) -> None:
        """Save all results to a JSON file."""
        _ensure_dir(path)
        payload = {
            "predicate": self.predicate,
            "timestamp": self.timestamp,
            "results": {
                strategy: scores.to_dict()
                for strategy, scores in self._results.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved → {path}")
 
    def save_csv(self, path: str | Path) -> None:
        """Save all results to a CSV file."""
        _ensure_dir(path)
        fieldnames = ["strategy"] + METRIC_KEYS
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for strategy, scores in self._results.items():
                row = {"strategy": strategy, **scores.to_dict()}
                writer.writerow(row)
        print(f"  Saved → {path}")
 
    def to_dict(self) -> dict:
        """Return all results as a plain dict."""
        return {
            strategy: scores.to_dict()
            for strategy, scores in self._results.items()
        }
 
 
def _ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)