from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from evaluation.metrics import AggregatedScores, _values_match

COL_W  = 14
MAX_EX = 3


class Report:
    def __init__(self, query_id: str = "", template: str = "",
                 predicate: str = "", n_expected: int = 0) -> None:
        self.query_id   = query_id
        self.template   = template
        self.predicate  = predicate
        self.n_expected = n_expected
        self.timestamp  = datetime.now().isoformat(timespec="seconds")
        self._results:  dict[str, AggregatedScores] = {}
        self._examples: dict[str, dict] = {}

    def add(self, strategy: str, scores: AggregatedScores,
            actual=None, expected=None,
            similarity_threshold: float = 0.30) -> "Report":
        if not isinstance(scores, AggregatedScores):
            raise ValueError(f"Expected AggregatedScores, got {type(scores).__name__}.")
        self._results[strategy] = scores
        if actual is not None and expected is not None:
            self._examples[strategy] = _fp_fn_examples(
                actual, expected, similarity_threshold)
        return self

    def print_table(self) -> None:
        if not self._results:
            print("No results to display.")
            return

        strat_w = max(len(s) for s in self._results) + 4
        best_f1 = max(s.f1_mean for s in self._results.values())

        header = (f"{'Strategy':<{strat_w}}"
                  f"{'GT':>6}{'Gen':>6}"
                  f"{'Precision':>{COL_W}}{'Recall':>{COL_W}}{'F1':>{COL_W}}"
                  f"{'Time(s)':>10}{'Tokens':>10}")
        sep = "─" * len(header)

        print()
        print(f"  Query    : {self.query_id}  |  Template : {self.template}")
        print(f"  Predicate: {self.predicate}")
        print(f"  Date     : {self.timestamp}")
        print()
        print(sep)
        print(header)
        print(sep)

        for strategy, scores in self._results.items():
            marker = " ✓" if scores.f1_mean == best_f1 else ""
            p = f"{scores.precision_mean:.3f}±{scores.precision_std:.3f}"
            r = f"{scores.recall_mean:.3f}±{scores.recall_std:.3f}"
            f = f"{scores.f1_mean:.3f}±{scores.f1_std:.3f}"
            gen = f"{scores.n_generated:.0f}"
            t = f"{scores.time_mean:.1f}±{scores.time_std:.1f}"
            tok = f"{scores.tokens_mean:.0f}"

            print(f"{strategy:<{strat_w}}"
                  f"{self.n_expected:>6}{gen:>6}"
                  f"{p:>{COL_W}}{r:>{COL_W}}{f:>{COL_W}}"
                  f"{t:>10}{tok:>10}{marker}")

            if strategy in self._examples:
                ex = self._examples[strategy]
                if ex["fp"]:
                    fp_str = ", ".join(f'"{v}"' for v in ex["fp"][:MAX_EX])
                    print(f"  {'':>{strat_w-2}}  FP: {fp_str}")
                if ex["fn"]:
                    fn_str = ", ".join(f'"{v}"' for v in ex["fn"][:MAX_EX])
                    print(f"  {'':>{strat_w-2}}  FN: {fn_str}")

        print(sep)
        print(f"  N={next(iter(self._results.values())).n_runs}  |  ✓ Best F1")
        print()

    def save_json(self, path) -> None:
        _ensure_dir(path)
        payload = {
            "query_id": self.query_id, "template": self.template,
            "predicate": self.predicate, "n_expected": self.n_expected,
            "timestamp": self.timestamp,
            "results": {
                s: {**sc.to_dict(),
                    **({"examples": self._examples[s]} if s in self._examples else {})}
                for s, sc in self._results.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved → {path}")

    def save_csv(self, path) -> None:
        _ensure_dir(path)
        keys = ["precision_mean", "precision_std", "recall_mean", "recall_std",
                "f1_mean", "f1_std", "n_runs", "n_generated",
                "time_mean", "time_std", "tokens_mean", "tokens_std"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["strategy", "n_expected"] + keys)
            writer.writeheader()
            for s, sc in self._results.items():
                writer.writerow({"strategy": s,
                                 "n_expected": self.n_expected,
                                 **{k: getattr(sc, k, 0) for k in keys}})
        print(f"  Saved → {path}")

    def to_dict(self) -> dict:
        return {s: sc.to_dict() for s, sc in self._results.items()}


def _fp_fn_examples(actual, expected, similarity_threshold):
    expected_list = list(expected)
    matched_exp   = [False] * len(expected_list)
    matched_act   = []
    for a in actual:
        found = False
        for i, e in enumerate(expected_list):
            if not matched_exp[i] and _values_match(a, e, similarity_threshold):
                matched_exp[i] = True
                found = True
                break
        matched_act.append(found)
    fp = [a for a, m in zip(actual, matched_act) if not m]
    fn = [e for e, m in zip(expected_list, matched_exp) if not m]
    return {"fp": sorted(fp)[:MAX_EX], "fn": sorted(fn)[:MAX_EX]}


def _ensure_dir(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)