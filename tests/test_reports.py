"""


- Unitary tests for report.py
- No network needed

pytest tests/test_report.py -v
"""

import sys
import os
import csv
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.report import Report
from evaluation.metrics import MetricScores


def make_scores(f1=0.8, card=0.9, p=0.85, r=0.75):
    avg = round((f1 + card) / 2, 4)
    return MetricScores(
        precision=p, recall=r,
        f1_triple=f1, cardinality=card,
        avg_score=avg,
    )

SCORES_A = make_scores(f1=0.8,  card=0.9)   # avg=0.85
SCORES_B = make_scores(f1=0.6,  card=0.7)   # avg=0.65
SCORES_C = make_scores(f1=1.0,  card=1.0)   # avg=1.0  ← best


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.__init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_empty_report(self):
        r = Report()
        assert r.strategies() == []

    def test_predicate_stored(self):
        r = Report(predicate="birthPlace")
        assert r.predicate == "birthPlace"

    def test_timestamp_set(self):
        r = Report()
        assert r.timestamp != ""


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.add
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdd:

    def test_add_returns_self(self):
        r = Report()
        result = r.add("TableScan", SCORES_A)
        assert result is r

    def test_strategy_registered(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        assert "TableScan" in r.strategies()

    def test_multiple_strategies(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        r.add("KeyCrank",  SCORES_B)
        assert len(r.strategies()) == 2

    def test_chaining(self):
        r = Report()
        r.add("A", SCORES_A).add("B", SCORES_B).add("C", SCORES_C)
        assert len(r.strategies()) == 3

    def test_wrong_type_raises(self):
        r = Report()
        with pytest.raises(ValueError, match="MetricScores"):
            r.add("TableScan", {"f1_triple": 0.8})

    def test_overwrite_existing_strategy(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        r.add("TableScan", SCORES_B)
        assert r.to_dict()["TableScan"]["f1_triple"] == SCORES_B.f1_triple


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.to_dict
# ═══════════════════════════════════════════════════════════════════════════════

class TestToDict:

    def test_empty_report(self):
        assert Report().to_dict() == {}

    def test_keys_are_strategy_names(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        r.add("KeyCrank",  SCORES_B)
        assert set(r.to_dict().keys()) == {"TableScan", "KeyCrank"}

    def test_values_contain_metric_keys(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        d = r.to_dict()["TableScan"]
        for key in ("precision", "recall", "f1_triple", "cardinality", "avg_score"):
            assert key in d

    def test_is_json_serializable(self):
        r = Report()
        r.add("TableScan", SCORES_A)
        json.dumps(r.to_dict())   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.save_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveJson:

    def test_creates_file(self, tmp_path):
        r = Report(predicate="birthPlace")
        r.add("TableScan", SCORES_A)
        out = tmp_path / "report.json"
        r.save_json(out)
        assert out.exists()

    def test_content_is_valid_json(self, tmp_path):
        r = Report(predicate="birthPlace")
        r.add("TableScan", SCORES_A)
        out = tmp_path / "report.json"
        r.save_json(out)
        data = json.loads(out.read_text())
        assert "results" in data
        assert "predicate" in data
        assert "timestamp" in data

    def test_results_contain_strategy(self, tmp_path):
        r = Report(predicate="birthPlace")
        r.add("TableScan", SCORES_A)
        out = tmp_path / "report.json"
        r.save_json(out)
        data = json.loads(out.read_text())
        assert "TableScan" in data["results"]

    def test_creates_parent_dirs(self, tmp_path):
        r = Report()
        r.add("TableScan", SCORES_A)
        out = tmp_path / "nested" / "deep" / "report.json"
        r.save_json(out)
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.save_csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveCsv:

    def test_creates_file(self, tmp_path):
        r = Report()
        r.add("TableScan", SCORES_A)
        out = tmp_path / "report.csv"
        r.save_csv(out)
        assert out.exists()

    def test_csv_has_header(self, tmp_path):
        r = Report()
        r.add("TableScan", SCORES_A)
        out = tmp_path / "report.csv"
        r.save_csv(out)
        with open(out) as f:
            reader = csv.DictReader(f)
            assert "strategy"  in reader.fieldnames
            assert "f1_triple" in reader.fieldnames

    def test_csv_has_one_row_per_strategy(self, tmp_path):
        r = Report()
        r.add("TableScan", SCORES_A)
        r.add("KeyCrank",  SCORES_B)
        out = tmp_path / "report.csv"
        r.save_csv(out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_creates_parent_dirs(self, tmp_path):
        r = Report()
        r.add("TableScan", SCORES_A)
        out = tmp_path / "nested" / "report.csv"
        r.save_csv(out)
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Report.print_table (smoke test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrintTable:

    def test_empty_report_does_not_crash(self, capsys):
        Report().print_table()
        out = capsys.readouterr().out
        assert "No results" in out

    def test_prints_strategy_names(self, capsys):
        r = Report(predicate="birthPlace")
        r.add("TableScan", SCORES_A)
        r.add("KeyCrank",  SCORES_B)
        r.print_table()
        out = capsys.readouterr().out
        assert "TableScan" in out
        assert "KeyCrank"  in out

    def test_prints_predicate(self, capsys):
        r = Report(predicate="birthPlace")
        r.add("TableScan", SCORES_A)
        r.print_table()
        out = capsys.readouterr().out
        assert "birthPlace" in out

    def test_best_strategy_marked(self, capsys):
        r = Report()
        r.add("TableScan", SCORES_A)   # avg=0.85
        r.add("KeyCrank",  SCORES_C)   # avg=1.0  ← best
        r.print_table()
        out = capsys.readouterr().out
        assert "✓" in out