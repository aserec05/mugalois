"""


- Unitary tests for evaluator.py
- No network needed

pytest tests/test_evaluator.py -v
"""

import sys
import os
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mugalois.core.types import Triple
from evaluation.evaluator import Evaluator, _dicts_to_triples, _load_triples
from evaluation.metrics import MetricScores


# ─── Fixtures ─────────────────────────────────────────────────────────────────

EINSTEIN_ULM = Triple("https://dbpedia.org/resource/Albert_Einstein", "birthPlace", "https://dbpedia.org/resource/Ulm")
CURIE_WARSAW = Triple("https://dbpedia.org/resource/Marie_Curie",     "birthPlace", "https://dbpedia.org/resource/Warsaw")
CURIE_PARIS  = Triple("https://dbpedia.org/resource/Marie_Curie",     "birthPlace", "https://dbpedia.org/resource/Paris")

ACTUAL_PERFECT   = {EINSTEIN_ULM, CURIE_WARSAW}
EXPECTED_PERFECT = {EINSTEIN_ULM, CURIE_WARSAW}
ACTUAL_PARTIAL   = {EINSTEIN_ULM, CURIE_PARIS}


def to_json_list(triples: set) -> list[dict]:
    return [{"s": t.s, "p": t.p, "o": t.o} for t in triples]


@pytest.fixture
def tmp_actual(tmp_path):
    p = tmp_path / "actual.json"
    p.write_text(json.dumps(to_json_list(ACTUAL_PERFECT)), encoding="utf-8")
    return p


@pytest.fixture
def tmp_expected(tmp_path):
    p = tmp_path / "expected.json"
    p.write_text(json.dumps(to_json_list(EXPECTED_PERFECT)), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _dicts_to_triples
# ═══════════════════════════════════════════════════════════════════════════════

class TestDictsToTriples:

    def test_converts_correctly(self):
        data = [{"s": "Einstein", "p": "birthPlace", "o": "Ulm"}]
        result = _dicts_to_triples(data)
        assert Triple("Einstein", "birthPlace", "Ulm") in result

    def test_empty_list(self):
        assert _dicts_to_triples([]) == set()

    def test_returns_set(self):
        data = [{"s": "A", "p": "p", "o": "B"}]
        assert isinstance(_dicts_to_triples(data), set)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _load_triples
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadTriples:

    def test_loads_valid_file(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps(to_json_list({EINSTEIN_ULM})), encoding="utf-8")
        result = _load_triples(p)
        assert EINSTEIN_ULM in result

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_triples(tmp_path / "nonexistent.json")

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"s": "x"}), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_triples(p)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.__init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_stores_as_sets(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        assert isinstance(ev.actual,   set)
        assert isinstance(ev.expected, set)

    def test_default_thresholds(self):
        ev = Evaluator(set(), set())
        assert ev.similarity_threshold == 0.10
        assert ev.numeric_tolerance    == 0.10

    def test_accepts_list_input(self):
        # Should also work with list input (converted to set)
        ev = Evaluator(list(ACTUAL_PERFECT), list(EXPECTED_PERFECT))
        assert isinstance(ev.actual, set)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.from_files
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromFiles:

    def test_loads_from_two_files(self, tmp_actual, tmp_expected):
        ev = Evaluator.from_files(tmp_actual, tmp_expected)
        assert ev.actual   == ACTUAL_PERFECT
        assert ev.expected == EXPECTED_PERFECT

    def test_missing_file_raises(self, tmp_path, tmp_expected):
        with pytest.raises(FileNotFoundError):
            Evaluator.from_files(tmp_path / "missing.json", tmp_expected)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.from_dicts
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromDicts:

    def test_loads_from_combined_dict(self):
        data = {
            "actual":   to_json_list(ACTUAL_PERFECT),
            "expected": to_json_list(EXPECTED_PERFECT),
        }
        ev = Evaluator.from_dicts(data)
        assert ev.actual   == ACTUAL_PERFECT
        assert ev.expected == EXPECTED_PERFECT

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Evaluator.from_dicts({"expected": to_json_list(EXPECTED_PERFECT)})


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.evaluate
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def test_returns_metric_scores(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        assert isinstance(ev.evaluate(), MetricScores)

    def test_perfect_scores(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        scores = ev.evaluate()
        assert scores.f1_triple   == 1.0
        assert scores.avg_score   == 1.0

    def test_empty_both(self):
        ev = Evaluator(set(), set())
        assert ev.evaluate().f1_triple == 1.0

    def test_empty_actual(self):
        ev = Evaluator(set(), EXPECTED_PERFECT)
        assert ev.evaluate().f1_triple == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.summary / save_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummary:

    def test_has_counts_and_metrics(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        s = ev.summary()
        assert s["n_actual"]   == len(ACTUAL_PERFECT)
        assert s["n_expected"] == len(EXPECTED_PERFECT)
        assert "f1_triple" in s

    def test_is_json_serializable(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        json.dumps(ev.summary())

    def test_save_summary_creates_file(self, tmp_path):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        out = tmp_path / "summary.json"
        ev.save_summary(out)
        assert out.exists()

    def test_save_summary_creates_parent_dirs(self, tmp_path):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        out = tmp_path / "nested" / "summary.json"
        ev.save_summary(out)
        assert out.exists()