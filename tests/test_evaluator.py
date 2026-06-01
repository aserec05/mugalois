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
from evaluation.evaluator import Evaluator, _validate, _load_json
from evaluation.metrics import MetricScores


# ─── Fixtures ─────────────────────────────────────────────────────────────────

EINSTEIN_ULM  = {"s": "https://dbpedia.org/resource/Albert_Einstein", "p": "birthPlace", "o": "https://dbpedia.org/resource/Ulm"}
CURIE_WARSAW  = {"s": "https://dbpedia.org/resource/Marie_Curie",     "p": "birthPlace", "o": "https://dbpedia.org/resource/Warsaw"}
CURIE_PARIS   = {"s": "https://dbpedia.org/resource/Marie_Curie",     "p": "birthPlace", "o": "https://dbpedia.org/resource/Paris"}

ACTUAL_PERFECT   = [EINSTEIN_ULM, CURIE_WARSAW]
EXPECTED_PERFECT = [EINSTEIN_ULM, CURIE_WARSAW]

ACTUAL_PARTIAL   = [EINSTEIN_ULM, CURIE_PARIS]   # one wrong
EXPECTED_PARTIAL = [EINSTEIN_ULM, CURIE_WARSAW]


@pytest.fixture
def tmp_actual(tmp_path):
    p = tmp_path / "actual.json"
    p.write_text(json.dumps(ACTUAL_PERFECT), encoding="utf-8")
    return p


@pytest.fixture
def tmp_expected(tmp_path):
    p = tmp_path / "expected.json"
    p.write_text(json.dumps(EXPECTED_PERFECT), encoding="utf-8")
    return p


@pytest.fixture
def tmp_combined(tmp_path):
    p = tmp_path / "combined.json"
    p.write_text(json.dumps({
        "actual":   ACTUAL_PERFECT,
        "expected": EXPECTED_PERFECT,
    }), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _validate
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidate:

    def test_valid_list_passes(self):
        _validate([EINSTEIN_ULM], "test")   # no exception

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate({"s": "x", "o": "y"}, "test")

    def test_item_not_dict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate(["not a dict"], "test")

    def test_missing_s_raises(self):
        with pytest.raises(ValueError, match="missing required key 's'"):
            _validate([{"o": "Ulm"}], "test")

    def test_missing_o_raises(self):
        with pytest.raises(ValueError, match="missing required key 'o'"):
            _validate([{"s": "Einstein"}], "test")

    def test_s_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            _validate([{"s": 42, "o": "Ulm"}], "test")

    def test_empty_list_passes(self):
        _validate([], "test")   # no exception

    def test_p_is_optional(self):
        # p is not required — should not raise
        _validate([{"s": "Einstein", "o": "Ulm"}], "test")


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _load_json
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadJson:

    def test_loads_valid_file(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([EINSTEIN_ULM]), encoding="utf-8")
        result = _load_json(p)
        assert result == [EINSTEIN_ULM]

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_json(tmp_path / "nonexistent.json")

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"s": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a JSON array"):
            _load_json(p)

    def test_empty_array_passes(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        assert _load_json(p) == []


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.__init__
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_stores_actual_and_expected(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        assert ev.actual   == ACTUAL_PERFECT
        assert ev.expected == EXPECTED_PERFECT

    def test_default_thresholds(self):
        ev = Evaluator([], [])
        assert ev.similarity_threshold == 0.10
        assert ev.numeric_tolerance    == 0.10

    def test_custom_thresholds(self):
        ev = Evaluator([], [], similarity_threshold=0.20, numeric_tolerance=0.05)
        assert ev.similarity_threshold == 0.20
        assert ev.numeric_tolerance    == 0.05

    def test_invalid_actual_raises(self):
        with pytest.raises(ValueError):
            Evaluator("not a list", [])

    def test_invalid_expected_raises(self):
        with pytest.raises(ValueError):
            Evaluator([], "not a list")


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.from_files
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromFiles:

    def test_loads_from_two_files(self, tmp_actual, tmp_expected):
        ev = Evaluator.from_files(tmp_actual, tmp_expected)
        assert ev.actual   == ACTUAL_PERFECT
        assert ev.expected == EXPECTED_PERFECT

    def test_missing_actual_raises(self, tmp_path, tmp_expected):
        with pytest.raises(FileNotFoundError):
            Evaluator.from_files(tmp_path / "missing.json", tmp_expected)

    def test_missing_expected_raises(self, tmp_path, tmp_actual):
        with pytest.raises(FileNotFoundError):
            Evaluator.from_files(tmp_actual, tmp_path / "missing.json")

    def test_accepts_string_paths(self, tmp_actual, tmp_expected):
        ev = Evaluator.from_files(str(tmp_actual), str(tmp_expected))
        assert len(ev.actual) == len(ACTUAL_PERFECT)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.from_dicts
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromDicts:

    def test_loads_from_combined_dict(self):
        data = {"actual": ACTUAL_PERFECT, "expected": EXPECTED_PERFECT}
        ev = Evaluator.from_dicts(data)
        assert ev.actual   == ACTUAL_PERFECT
        assert ev.expected == EXPECTED_PERFECT

    def test_custom_keys(self):
        data = {"results": ACTUAL_PERFECT, "ground_truth": EXPECTED_PERFECT}
        ev = Evaluator.from_dicts(data, actual_key="results", expected_key="ground_truth")
        assert ev.actual == ACTUAL_PERFECT

    def test_missing_actual_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Evaluator.from_dicts({"expected": EXPECTED_PERFECT})

    def test_missing_expected_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Evaluator.from_dicts({"actual": ACTUAL_PERFECT})


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
        assert scores.cardinality == 1.0
        assert scores.avg_score   == 1.0

    def test_partial_scores(self):
        ev = Evaluator(ACTUAL_PARTIAL, EXPECTED_PARTIAL)
        scores = ev.evaluate()
        assert 0.0 < scores.f1_triple < 1.0

    def test_empty_both(self):
        ev = Evaluator([], [])
        scores = ev.evaluate()
        assert scores.f1_triple == 1.0

    def test_empty_actual(self):
        ev = Evaluator([], EXPECTED_PERFECT)
        scores = ev.evaluate()
        assert scores.f1_triple   == 0.0
        assert scores.cardinality == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummary:

    def test_summary_has_counts(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        s = ev.summary()
        assert s["n_actual"]   == len(ACTUAL_PERFECT)
        assert s["n_expected"] == len(EXPECTED_PERFECT)

    def test_summary_has_all_metric_keys(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        s = ev.summary()
        for key in ("precision", "recall", "f1_triple", "cardinality", "avg_score"):
            assert key in s

    def test_summary_is_json_serializable(self):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        json.dumps(ev.summary())   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Evaluator.save_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveSummary:

    def test_creates_file(self, tmp_path):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        out = tmp_path / "summary.json"
        ev.save_summary(out)
        assert out.exists()

    def test_file_content_is_valid_json(self, tmp_path):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        out = tmp_path / "summary.json"
        ev.save_summary(out)
        data = json.loads(out.read_text())
        assert "f1_triple" in data

    def test_creates_parent_dirs(self, tmp_path):
        ev = Evaluator(ACTUAL_PERFECT, EXPECTED_PERFECT)
        out = tmp_path / "nested" / "deep" / "summary.json"
        ev.save_summary(out)
        assert out.exists()