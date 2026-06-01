"""


- Unitary tests for metrics.py
- No network needed

pytest tests/test_metrics.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.metrics import (
    Metrics,
    MetricScores,
    _normalize,
    _edit_distance,
    _values_match,
    _triples_match,
    _count_true_positives,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def triple(s, o, p="https://schema.org/birthPlace"):
    return {"s": s, "p": p, "o": o}


EINSTEIN_ULM     = triple("https://dbpedia.org/resource/Albert_Einstein", "https://dbpedia.org/resource/Ulm")
CURIE_WARSAW     = triple("https://dbpedia.org/resource/Marie_Curie",     "https://dbpedia.org/resource/Warsaw")
NEWTON_WOOL      = triple("https://dbpedia.org/resource/Isaac_Newton",    "https://dbpedia.org/resource/Woolsthorpe")
CURIE_PARIS      = triple("https://dbpedia.org/resource/Marie_Curie",     "https://dbpedia.org/resource/Paris")     # wrong o
DARWIN_SHREWS    = triple("https://dbpedia.org/resource/Charles_Darwin",  "https://dbpedia.org/resource/Shrewsbury") # not in expected


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

    def test_plain_string_unchanged_except_lower(self):
        assert _normalize("ulm") == "ulm"

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

    def test_one_empty(self):
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _values_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestValuesMatch:

    def test_exact_match(self):
        assert _values_match("Ulm", "Ulm") is True

    def test_uri_vs_local_name(self):
        assert _values_match(
            "https://dbpedia.org/resource/Ulm", "Ulm"
        ) is True

    def test_both_uris_same_local_name(self):
        assert _values_match(
            "https://dbpedia.org/resource/Albert_Einstein",
            "https://yago.org/resource/Albert_Einstein",
        ) is True

    def test_different_values(self):
        assert _values_match("Ulm", "Warsaw") is False

    def test_typo_within_threshold(self):
        # "Warsaw" vs "Varsaw" — 1 edit, threshold 10% of 6 = 1
        assert _values_match("Varsaw", "Warsaw") is True

    def test_typo_beyond_threshold(self):
        assert _values_match("xyz", "Warsaw") is False

    def test_numeric_within_tolerance(self):
        assert _values_match("1000", "1050", numeric_tolerance=0.10) is True

    def test_numeric_beyond_tolerance(self):
        assert _values_match("500", "1000", numeric_tolerance=0.10) is False

    def test_numeric_exact(self):
        assert _values_match("42", "42") is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _triples_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriplesMatch:

    def test_identical_triples(self):
        assert _triples_match(EINSTEIN_ULM, EINSTEIN_ULM, 0.10, 0.10) is True

    def test_wrong_object(self):
        # Same subject, different object
        assert _triples_match(CURIE_PARIS, CURIE_WARSAW, 0.10, 0.10) is False

    def test_wrong_subject(self):
        wrong_s = triple("https://dbpedia.org/resource/Napoleon", "https://dbpedia.org/resource/Ulm")
        assert _triples_match(wrong_s, EINSTEIN_ULM, 0.10, 0.10) is False

    def test_predicate_ignored(self):
        # p is different but s and o match — should still match
        t1 = {"s": "Einstein", "p": "https://schema.org/birthPlace", "o": "Ulm"}
        t2 = {"s": "Einstein", "p": "http://dbpedia.org/ontology/birthPlace", "o": "Ulm"}
        assert _triples_match(t1, t2, 0.10, 0.10) is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Precision
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrecision:

    def test_all_correct(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.precision(actual, expected) == 1.0

    def test_none_correct(self):
        actual   = [CURIE_PARIS, DARWIN_SHREWS]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.precision(actual, expected) == 0.0

    def test_half_correct(self):
        actual   = [EINSTEIN_ULM, CURIE_PARIS]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.precision(actual, expected) == 0.5

    def test_empty_actual_empty_expected(self):
        assert Metrics.precision([], []) == 1.0

    def test_empty_actual_nonempty_expected(self):
        assert Metrics.precision([], [EINSTEIN_ULM]) == 0.0

    def test_extra_triples_penalise_precision(self):
        # actual has more triples than expected — extra ones are false positives
        actual   = [EINSTEIN_ULM, CURIE_WARSAW, DARWIN_SHREWS]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert round(Metrics.precision(actual, expected), 4) == round(2/3, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Recall
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecall:

    def test_all_found(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.recall(actual, expected) == 1.0

    def test_none_found(self):
        actual   = [CURIE_PARIS]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.recall(actual, expected) == 0.0

    def test_half_found(self):
        actual   = [EINSTEIN_ULM]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.recall(actual, expected) == 0.5

    def test_empty_expected(self):
        assert Metrics.recall([EINSTEIN_ULM], []) == 1.0

    def test_missing_triples_penalise_recall(self):
        # actual has fewer triples than expected
        actual   = [EINSTEIN_ULM]
        expected = [EINSTEIN_ULM, CURIE_WARSAW, NEWTON_WOOL]
        assert round(Metrics.recall(actual, expected), 4) == round(1/3, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — F1-Triple
# ═══════════════════════════════════════════════════════════════════════════════

class TestF1Triple:

    def test_perfect(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.f1_triple(actual, expected) == 1.0

    def test_zero(self):
        actual   = [CURIE_PARIS]
        expected = [EINSTEIN_ULM]
        assert Metrics.f1_triple(actual, expected) == 0.0

    def test_both_empty(self):
        assert Metrics.f1_triple([], []) == 1.0

    def test_partial(self):
        # precision=0.5, recall=0.5 → F1=0.5
        actual   = [EINSTEIN_ULM, CURIE_PARIS]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.f1_triple(actual, expected) == 0.5

    def test_wrong_o_is_not_partial_credit(self):
        # F1-Triple gives no credit for a wrong object — unlike F1-Cell
        actual   = [CURIE_PARIS]    # wrong o
        expected = [CURIE_WARSAW]
        assert Metrics.f1_triple(actual, expected) == 0.0

    def test_no_double_counting(self):
        # Same triple returned twice should not count as two true positives
        actual   = [EINSTEIN_ULM, EINSTEIN_ULM]
        expected = [EINSTEIN_ULM]
        p = Metrics.precision(actual, expected)
        assert p == 0.5   # only one match out of two returned


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — Cardinality
# ═══════════════════════════════════════════════════════════════════════════════

class TestCardinality:

    def test_same_size(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.cardinality(actual, expected) == 1.0

    def test_actual_smaller(self):
        actual   = [EINSTEIN_ULM]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        assert Metrics.cardinality(actual, expected) == 0.5

    def test_actual_larger(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW, NEWTON_WOOL]
        expected = [EINSTEIN_ULM]
        assert round(Metrics.cardinality(actual, expected), 4) == round(1/3, 4)

    def test_both_empty(self):
        assert Metrics.cardinality([], []) == 1.0

    def test_actual_empty(self):
        assert Metrics.cardinality([], [EINSTEIN_ULM]) == 0.0

    def test_expected_empty(self):
        assert Metrics.cardinality([EINSTEIN_ULM], []) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — compute (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompute:

    def test_returns_metric_scores(self):
        result = Metrics.compute([EINSTEIN_ULM], [EINSTEIN_ULM])
        assert isinstance(result, MetricScores)

    def test_perfect_scores(self):
        actual   = [EINSTEIN_ULM, CURIE_WARSAW]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        scores = Metrics.compute(actual, expected)
        assert scores.precision   == 1.0
        assert scores.recall      == 1.0
        assert scores.f1_triple   == 1.0
        assert scores.cardinality == 1.0
        assert scores.avg_score   == 1.0

    def test_zero_scores(self):
        actual   = [CURIE_PARIS]
        expected = [EINSTEIN_ULM]
        scores = Metrics.compute(actual, expected)
        assert scores.f1_triple == 0.0

    def test_avg_score_is_mean_of_f1_and_cardinality(self):
        actual   = [EINSTEIN_ULM]
        expected = [EINSTEIN_ULM, CURIE_WARSAW]
        scores = Metrics.compute(actual, expected)
        # avg is computed on raw (unrounded) values, then rounded once
        # so we recompute the same way here
        p    = Metrics.precision(actual, expected)
        r    = Metrics.recall(actual, expected)
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        card = Metrics.cardinality(actual, expected)
        expected_avg = round((f1 + card) / 2, 4)
        assert scores.avg_score == expected_avg

    def test_to_dict_has_all_keys(self):
        scores = Metrics.compute([EINSTEIN_ULM], [EINSTEIN_ULM])
        d = scores.to_dict()
        assert set(d.keys()) == {
            "precision", "recall", "f1_triple", "cardinality", "avg_score"
        }

    def test_str_representation(self):
        scores = Metrics.compute([EINSTEIN_ULM], [EINSTEIN_ULM])
        s = str(scores)
        assert "F1-Triple" in s
        assert "AVG-Score" in s