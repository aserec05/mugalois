"""
Unitary tests for mugalois.core.operators

pytest tests/test_operators.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mugalois.core.types import Triple
from mugalois.core.operators import (
    filter_by_subject,
    filter_by_object,
    filter_by_subject_object,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────

EINSTEIN_ULM      = Triple("Einstein", "birthPlace", "Ulm")
CURIE_WARSAW      = Triple("Curie",    "birthPlace", "Warsaw")
NEWTON_WOOL       = Triple("Newton",   "birthPlace", "Woolsthorpe")

EINSTEIN_REL      = Triple("Einstein", "knownFor",   "Relativity")
NEWTON_GRAV       = Triple("Newton",   "knownFor",   "Gravity")
DARWIN_EVO        = Triple("Darwin",   "knownFor",   "Evolution")

ULM_GERMANY       = Triple("Ulm",      "locatedIn",  "Germany")
WARSAW_POLAND     = Triple("Warsaw",   "locatedIn",  "Poland")
PARIS_FRANCE      = Triple("Paris",    "locatedIn",  "France")

T1 = {EINSTEIN_ULM, CURIE_WARSAW, NEWTON_WOOL}
T2 = {EINSTEIN_REL, NEWTON_GRAV, DARWIN_EVO}
T3 = {ULM_GERMANY, WARSAW_POLAND, PARIS_FRANCE}


# ═══════════════════════════════════════════════════════════════════════════════
# filter_by_subject
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterBySubject:

    def test_keeps_common_subjects(self):
        result = filter_by_subject(T1, T2)
        assert EINSTEIN_ULM in result
        assert NEWTON_WOOL  in result

    def test_removes_missing_subjects(self):
        result = filter_by_subject(T1, T2)
        assert CURIE_WARSAW not in result  # Curie not in T2

    def test_empty_T1(self):
        assert filter_by_subject(set(), T2) == set()

    def test_empty_T2(self):
        assert filter_by_subject(T1, set()) == set()

    def test_no_common_subjects(self):
        T_other = {Triple("Darwin", "birthPlace", "Shrewsbury")}
        result = filter_by_subject(T1, T_other)
        assert result == set()

    def test_all_common_subjects(self):
        result = filter_by_subject(T1, T1)
        assert result == T1

    def test_returns_set_of_triples(self):
        result = filter_by_subject(T1, T2)
        assert isinstance(result, set)
        assert all(isinstance(t, Triple) for t in result)


# ═══════════════════════════════════════════════════════════════════════════════
# filter_by_object
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterByObject:

    def test_keeps_common_objects(self):
        # T1 objects: Ulm, Warsaw, Woolsthorpe
        # T3 subjects: Ulm, Warsaw, Paris
        # T3 objects: Germany, Poland, France
        # filter T1 by objects of T3 → nothing in common
        result = filter_by_object(T1, T3)
        assert result == set()

    def test_basic_match(self):
        T_a = {Triple("A", "p", "X"), Triple("B", "p", "Y")}
        T_b = {Triple("C", "p", "X"), Triple("D", "p", "Z")}
        result = filter_by_object(T_a, T_b)
        assert Triple("A", "p", "X") in result
        assert Triple("B", "p", "Y") not in result

    def test_empty_T1(self):
        assert filter_by_object(set(), T2) == set()

    def test_empty_T2(self):
        assert filter_by_object(T1, set()) == set()

    def test_returns_set_of_triples(self):
        result = filter_by_object(T1, T2)
        assert isinstance(result, set)


# ═══════════════════════════════════════════════════════════════════════════════
# filter_by_subject_object
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterBySubjectObject:

    def test_subject_of_T1_matches_object_of_T2(self):
        # T1 subjects: Einstein, Curie, Newton
        # T3 objects: Germany, Poland, France
        # no match
        result = filter_by_subject_object(T1, T3)
        assert result == set()

    def test_basic_match(self):
        # T1: (Ulm, locatedIn, Germany)
        # T2: (Einstein, birthPlace, Ulm)  ← object = Ulm
        # filter T1 by T2: keep T1 triples whose subject appears as object in T2
        T_a = {Triple("Ulm",    "locatedIn", "Germany"),
               Triple("Warsaw", "locatedIn", "Poland")}
        T_b = {Triple("Einstein", "birthPlace", "Ulm"),
               Triple("Darwin",   "birthPlace", "Shrewsbury")}
        result = filter_by_subject_object(T_a, T_b)
        assert Triple("Ulm", "locatedIn", "Germany") in result
        assert Triple("Warsaw", "locatedIn", "Poland") not in result

    def test_empty_T1(self):
        assert filter_by_subject_object(set(), T2) == set()

    def test_empty_T2(self):
        assert filter_by_subject_object(T1, set()) == set()

    def test_returns_set_of_triples(self):
        result = filter_by_subject_object(T1, T2)
        assert isinstance(result, set)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases communs
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_triple_match(self):
        T_a = {Triple("X", "p", "Y")}
        T_b = {Triple("X", "q", "Z")}
        assert filter_by_subject(T_a, T_b) == T_a

    def test_single_triple_no_match(self):
        T_a = {Triple("X", "p", "Y")}
        T_b = {Triple("W", "q", "Z")}
        assert filter_by_subject(T_a, T_b) == set()

    def test_does_not_modify_input(self):
        T_a = {EINSTEIN_ULM, CURIE_WARSAW}
        T_b = {EINSTEIN_REL}
        original_size = len(T_a)
        filter_by_subject(T_a, T_b)
        assert len(T_a) == original_size
