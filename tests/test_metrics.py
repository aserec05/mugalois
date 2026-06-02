"""


- Unitary tests for metrics.py
- No network needed

pytest tests/test_metrics.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mugalois.core.types import Triple
from evaluation.metrics import (
    Metrics,
    MetricScores,
    _normalize,
    _edit_distance,
    _values_match,
    _triples_match,
    _count_true_positives,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

EINSTEIN_ULM  = Triple("https://dbpedia.org/resource/Albert_Einstein", "birthPlace", "https://dbpedia.org/resource/Ulm")
CURIE_WARSAW  = Triple("https://dbpedia.org/resource/Marie_Curie",     "birthPlace", "https://dbpedia.org/resource/Warsaw")
NEWTON_WOOL   = Triple("https://dbpedia.org/resource/Isaac_Newton",    "birthPlace", "https://dbpedia.org/resource/Woolsthorpe")
CURIE_PARIS   = Triple("https://dbpedia.org/resource/Marie_Curie",     "birthPlace", "https://dbpedia.org/resource/Paris")
DARWIN_SHREWS = Triple("https://dbpedia.org/resource/Charles_Darwin",  "birthPlace", "https://dbpedia.org/resource/Shrewsbury")


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _normalize
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalize:

    def test_uri_stripped_to_local_name(self):
        assert _normalize("https://dbpedia.org/resource/Albert_Einstein") == "albert einstein"

    def test_underscore_replaced_by_space(self):
        assert _normalize("Albert_Einstein") == "albert einstein"

    def test_lowercased(self):
        assert _normalize("Warsaw") == "warsaw"

    def test_hash_uri(self):
        assert _normalize("http://example.org/ontology#birthPlace") == "birthplace"

    def test_trailing_slash_ignored(self):
        assert _normalize("https://dbpedia.org/resource/Ulm/") == "ulm"


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _edit_distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestEditDistance:

    def test_identical_strings(self):
        assert _edit_distance("ulm", "ulm") == 0

    def test_one_insertion(self):
        assert _edit_distance("ulm", "ulms") == 1

    def test_one_substitution(self):
        assert _edit_distance("ulm", "elm") == 1

    def test_completely_different(self):
        assert _edit_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _values_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestValuesMatch:

    def test_exact_match(self):
        assert _values_match("Ulm", "Ulm") is True

    def test_uri_vs_local_name(self):
        assert _values_match("https://dbpedia.org/resource/Ulm", "Ulm") is True

    def test_different_values(self):
        assert _values_match("Ulm", "Warsaw") is False

    def test_typo_within_threshold(self):
        assert _values_match("Varsaw", "Warsaw") is True

    def test_numeric_within_tolerance(self):
        assert _values_match("1000", "1050", numeric_tolerance=0.10) is True

    def test_numeric_beyond_tolerance(self):
        assert _values_match("500", "1000", numeric_tolerance=0.10) is False


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _triples_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriplesMatch:

    def test_identical_triples(self):
        assert _triples_match(EINSTEIN_ULM, EINSTEIN_ULM, 0.10, 0.10) is True

    def test_wrong_object(self):
        assert _triples_match(CURIE_PARIS, CURIE_WARSAW, 0.10, 0.10) is False

    def test_wrong_subject(self):
        wrong = Triple("https://dbpedia.org/resource/Napoleon", "birthPlace", "https://dbpedia.org/resource/Ulm")
        assert _triples_match(wrong, EINSTEIN_ULM, 0.10, 0.10) is False

    def test_predicate_ignored(self):
        t1 = Triple("Einstein", "schema:birthPlace",   "Ulm")
        t2 = Triple("Einstein", "dbo:birthPlace", "Ulm")
        assert _triples_match(t1, t2, 0.10, 0.10) is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Precision
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrecision:

    def test_all_correct(self):
        assert Metrics.precision({EINSTEIN_ULM, CURIE_WARSAW}, {EINSTEIN_ULM, CURIE_WARSAW}) == 1.0

    def test_none_correct(self):
        assert Metrics.precision({CURIE_PARIS}, {EINSTEIN_ULM}) == 0.0

    def test_half_correct(self):
        actual   = {EINSTEIN_ULM, CURIE_PARIS}
        expected = {EINSTEIN_ULM, CURIE_WARSAW}
        assert Metrics.precision(actual, expected) == 0.5

    def test_empty_both(self):
        assert Metrics.precision(set(), set()) == 1.0

    def test_empty_actual(self):
        assert Metrics.precision(set(), {EINSTEIN_ULM}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Recall
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecall:

    def test_all_found(self):
        assert Metrics.recall({EINSTEIN_ULM, CURIE_WARSAW}, {EINSTEIN_ULM, CURIE_WARSAW}) == 1.0

    def test_none_found(self):
        assert Metrics.recall({CURIE_PARIS}, {EINSTEIN_ULM}) == 0.0

    def test_half_found(self):
        assert Metrics.recall({EINSTEIN_ULM}, {EINSTEIN_ULM, CURIE_WARSAW}) == 0.5

    def test_empty_expected(self):
        assert Metrics.recall({EINSTEIN_ULM}, set()) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — F1-Triple
# ═══════════════════════════════════════════════════════════════════════════════

class TestF1Triple:

    def test_perfect(self):
        assert Metrics.f1_triple({EINSTEIN_ULM}, {EINSTEIN_ULM}) == 1.0

    def test_zero(self):
        assert Metrics.f1_triple({CURIE_PARIS}, {EINSTEIN_ULM}) == 0.0

    def test_both_empty(self):
        assert Metrics.f1_triple(set(), set()) == 1.0

    def test_wrong_o_is_not_partial_credit(self):
        assert Metrics.f1_triple({CURIE_PARIS}, {CURIE_WARSAW}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Cardinality
# ═══════════════════════════════════════════════════════════════════════════════

class TestCardinality:

    def test_same_size(self):
        assert Metrics.cardinality({EINSTEIN_ULM}, {EINSTEIN_ULM}) == 1.0

    def test_actual_smaller(self):
        assert Metrics.cardinality({EINSTEIN_ULM}, {EINSTEIN_ULM, CURIE_WARSAW}) == 0.5

    def test_both_empty(self):
        assert Metrics.cardinality(set(), set()) == 1.0

    def test_actual_empty(self):
        assert Metrics.cardinality(set(), {EINSTEIN_ULM}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — compute
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompute:

    def test_returns_metric_scores(self):
        assert isinstance(Metrics.compute({EINSTEIN_ULM}, {EINSTEIN_ULM}), MetricScores)

    def test_perfect_scores(self):
        scores = Metrics.compute({EINSTEIN_ULM, CURIE_WARSAW}, {EINSTEIN_ULM, CURIE_WARSAW})
        assert scores.f1_triple   == 1.0
        assert scores.cardinality == 1.0
        assert scores.avg_score   == 1.0

    def test_avg_score_is_mean_of_f1_and_cardinality(self):
        actual   = {EINSTEIN_ULM}
        expected = {EINSTEIN_ULM, CURIE_WARSAW}
        scores = Metrics.compute(actual, expected)
        p    = Metrics.precision(actual, expected)
        r    = Metrics.recall(actual, expected)
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        card = Metrics.cardinality(actual, expected)
        assert scores.avg_score == round((f1 + card) / 2, 4)

    def test_to_dict_has_all_keys(self):
        scores = Metrics.compute({EINSTEIN_ULM}, {EINSTEIN_ULM})
        assert set(scores.to_dict().keys()) == {
            "precision", "recall", "f1_triple", "cardinality", "avg_score"
        }