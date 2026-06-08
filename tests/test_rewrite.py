"""
Tests for RW1 and Condition types.

pytest tests/test_rewrite.py -v
"""

import pytest
from mugalois.core.types import (
    Condition, ConditionIN, Environment, AnyCondition
)
from mugalois.core.rewrite import RW1, _apply


# ══════════════════════════════════════════════════════════════════════════════
# Condition
# ══════════════════════════════════════════════════════════════════════════════

class TestCondition:

    def test_equality(self):
        c = Condition("?x", "=", "dbr:Einstein")
        assert c.is_equality() is True
        assert c.is_numeric() is False
        assert c.is_inequality() is False

    def test_numeric_gt(self):
        c = Condition("?y", ">", "1900")
        assert c.is_numeric() is True
        assert c.is_equality() is False

    def test_numeric_lt(self):
        c = Condition("?y", "<", "2000")
        assert c.is_numeric() is True

    def test_numeric_gte(self):
        c = Condition("?y", ">=", "1900")
        assert c.is_numeric() is True

    def test_numeric_lte(self):
        c = Condition("?y", "<=", "2000")
        assert c.is_numeric() is True

    def test_inequality(self):
        c = Condition("?y", "!=", "dbr:Curie")
        assert c.is_inequality() is True
        assert c.is_numeric() is False

    def test_variables(self):
        c = Condition("?x", "=", "dbr:Einstein")
        assert c.variables() == {"?x"}

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            Condition("?x", "??", "val")

    def test_invalid_var_raises(self):
        with pytest.raises(ValueError):
            Condition("x", "=", "val")

    def test_repr(self):
        c = Condition("?x", "=", "dbr:Einstein")
        assert "?x" in repr(c)
        assert "=" in repr(c)
        assert "dbr:Einstein" in repr(c)

    def test_hashable(self):
        c1 = Condition("?x", "=", "dbr:Einstein")
        c2 = Condition("?x", "=", "dbr:Einstein")
        assert c1 == c2
        assert len({c1, c2}) == 1

    def test_immutable(self):
        c = Condition("?x", "=", "val")
        with pytest.raises(Exception):
            c.var = "?y"


# ══════════════════════════════════════════════════════════════════════════════
# ConditionIN
# ══════════════════════════════════════════════════════════════════════════════

class TestConditionIN:

    def test_creation(self):
        c = ConditionIN("?y", frozenset({"Nobel Physics", "Nobel Chemistry"}))
        assert c.var == "?y"
        assert "Nobel Physics" in c.values

    def test_is_finite_set(self):
        c = ConditionIN("?y", frozenset({"a", "b"}))
        assert c.is_finite_set() is True
        assert c.is_equality() is False

    def test_variables(self):
        c = ConditionIN("?x", frozenset({"a"}))
        assert c.variables() == {"?x"}

    def test_invalid_var_raises(self):
        with pytest.raises(ValueError):
            ConditionIN("x", frozenset({"a"}))

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            ConditionIN("?x", frozenset())

    def test_hashable(self):
        c1 = ConditionIN("?x", frozenset({"a", "b"}))
        c2 = ConditionIN("?x", frozenset({"a", "b"}))
        assert c1 == c2
        assert len({c1, c2}) == 1

    def test_repr(self):
        c = ConditionIN("?x", frozenset({"a"}))
        assert "?x" in repr(c)
        assert "IN" in repr(c)


# ══════════════════════════════════════════════════════════════════════════════
# RW1 — equality
# ══════════════════════════════════════════════════════════════════════════════

class TestRW1Equality:

    def test_equality_no_seeds(self):
        """FILTER(?x = Einstein) → env.set(?x, {Einstein})"""
        c = Condition("?x", "=", "Einstein")
        env, gamma = RW1(frozenset({c}), Environment(), Environment())
        assert env.get("?x") == {"Einstein"}
        assert not gamma.has("?x")

    def test_equality_intersects_with_seeds(self):
        """Seeds = {Einstein, Curie}, FILTER(?x = Einstein) → {Einstein}"""
        c = Condition("?x", "=", "Einstein")
        env = Environment()
        env.set("?x", {"Einstein", "Curie"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?x") == {"Einstein"}

    def test_equality_no_intersection(self):
        """Seeds = {Curie}, FILTER(?x = Einstein) → {} (empty)"""
        c = Condition("?x", "=", "Einstein")
        env = Environment()
        env.set("?x", {"Curie"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?x") == set()

    def test_two_equalities_same_var(self):
        """FILTER(?x = A AND ?x = B) → {} (impossible intersection)"""
        c1 = Condition("?x", "=", "A")
        c2 = Condition("?x", "=", "B")
        env, gamma = RW1(frozenset({c1, c2}), Environment(), Environment())
        assert env.get("?x") == set()

    def test_two_equalities_different_vars(self):
        """FILTER(?x = Einstein AND ?y = Curie)"""
        c1 = Condition("?x", "=", "Einstein")
        c2 = Condition("?y", "=", "Curie")
        env, gamma = RW1(frozenset({c1, c2}), Environment(), Environment())
        assert env.get("?x") == {"Einstein"}
        assert env.get("?y") == {"Curie"}


# ══════════════════════════════════════════════════════════════════════════════
# RW1 — ConditionIN
# ══════════════════════════════════════════════════════════════════════════════

class TestRW1ConditionIN:

    def test_in_no_seeds(self):
        """FILTER(?y IN {a, b}) → env.set(?y, {a, b})"""
        c = ConditionIN("?y", frozenset({"a", "b"}))
        env, gamma = RW1(frozenset({c}), Environment(), Environment())
        assert env.get("?y") == {"a", "b"}

    def test_in_intersects_with_seeds(self):
        """Seeds = {a, b, c}, FILTER(?y IN {a, b}) → {a, b}"""
        c = ConditionIN("?y", frozenset({"a", "b"}))
        env = Environment()
        env.set("?y", {"a", "b", "c"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == {"a", "b"}

    def test_in_no_intersection(self):
        """Seeds = {c, d}, FILTER(?y IN {a, b}) → {}"""
        c = ConditionIN("?y", frozenset({"a", "b"}))
        env = Environment()
        env.set("?y", {"c", "d"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == set()


# ══════════════════════════════════════════════════════════════════════════════
# RW1 — numeric / inequality with seeds
# ══════════════════════════════════════════════════════════════════════════════

class TestRW1NumericWithSeeds:

    def test_gt_filters_seeds(self):
        """Seeds = {1800, 1920, 1850}, FILTER(?y > 1900) → {1920}"""
        c = Condition("?y", ">", "1900")
        env = Environment()
        env.set("?y", {"1800", "1920", "1850"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == {"1920"}

    def test_lt_filters_seeds(self):
        """Seeds = {1800, 1920, 1850}, FILTER(?y < 1900) → {1800, 1850}"""
        c = Condition("?y", "<", "1900")
        env = Environment()
        env.set("?y", {"1800", "1920", "1850"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == {"1800", "1850"}

    def test_gte_filters_seeds(self):
        c = Condition("?y", ">=", "1900")
        env = Environment()
        env.set("?y", {"1800", "1900", "1920"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == {"1900", "1920"}

    def test_lte_filters_seeds(self):
        c = Condition("?y", "<=", "1900")
        env = Environment()
        env.set("?y", {"1800", "1900", "1920"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == {"1800", "1900"}

    def test_neq_filters_seeds(self):
        """Seeds = {Einstein, Curie}, FILTER(?x != Curie) → {Einstein}"""
        c = Condition("?x", "!=", "Curie")
        env = Environment()
        env.set("?x", {"Einstein", "Curie", "Newton"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert "Curie" not in env.get("?x")
        assert "Einstein" in env.get("?x")

    def test_numeric_all_filtered_out(self):
        """Seeds all below threshold → empty"""
        c = Condition("?y", ">", "2000")
        env = Environment()
        env.set("?y", {"1800", "1900"})
        env, gamma = RW1(frozenset({c}), env, Environment())
        assert env.get("?y") == set()


# ══════════════════════════════════════════════════════════════════════════════
# RW1 — goes to gamma (no seeds, not equality)
# ══════════════════════════════════════════════════════════════════════════════

class TestRW1Gamma:

    def test_numeric_no_seeds_goes_to_gamma(self):
        """FILTER(?y > 1900) with no seeds → gamma"""
        c = Condition("?y", ">", "1900")
        env, gamma = RW1(frozenset({c}), Environment(), Environment())
        assert not env.has("?y")
        assert c in gamma.get("?y")

    def test_inequality_no_seeds_goes_to_gamma(self):
        """FILTER(?x != Curie) with no seeds → gamma"""
        c = Condition("?x", "!=", "Curie")
        env, gamma = RW1(frozenset({c}), Environment(), Environment())
        assert not env.has("?x")
        assert c in gamma.get("?x")

    def test_gamma_does_not_touch_env(self):
        """Condition in gamma should not pollute env"""
        c = Condition("?y", ">", "1900")
        env, gamma = RW1(frozenset({c}), Environment(), Environment())
        assert env.get("?y") == set()


# ══════════════════════════════════════════════════════════════════════════════
# RW1 — mixed conditions
# ══════════════════════════════════════════════════════════════════════════════

class TestRW1Mixed:

    def test_equality_and_numeric_no_seeds(self):
        """
        FILTER(?x = Einstein AND ?y > 1900)
        ?x → env, ?y → gamma
        """
        c1 = Condition("?x", "=", "Einstein")
        c2 = Condition("?y", ">", "1900")
        env, gamma = RW1(frozenset({c1, c2}), Environment(), Environment())
        assert env.get("?x") == {"Einstein"}
        assert not env.has("?y")
        assert c2 in gamma.get("?y")

    def test_in_and_numeric_with_seeds(self):
        """
        Seeds ?y = {1800, 1920, 1850}
        FILTER(?x IN {Einstein} AND ?y > 1900)
        ?x → env from IN, ?y → env filtered
        """
        c1 = ConditionIN("?x", frozenset({"Einstein"}))
        c2 = Condition("?y", ">", "1900")
        env = Environment()
        env.set("?y", {"1800", "1920", "1850"})
        env, gamma = RW1(frozenset({c1, c2}), env, Environment())
        assert env.get("?x") == {"Einstein"}
        assert env.get("?y") == {"1920"}
        assert not gamma.has("?y")

    def test_empty_conditions(self):
        """No conditions → env and gamma unchanged"""
        env = Environment()
        env.set("?x", {"Einstein"})
        env, gamma = RW1(frozenset(), env, Environment())
        assert env.get("?x") == {"Einstein"}
        assert not gamma.has("?x")


# ══════════════════════════════════════════════════════════════════════════════
# _apply helper
# ══════════════════════════════════════════════════════════════════════════════

class TestApply:

    def test_gt_numeric(self):
        assert _apply("1920", ">", "1900") is True
        assert _apply("1800", ">", "1900") is False

    def test_lt_numeric(self):
        assert _apply("1800", "<", "1900") is True
        assert _apply("1920", "<", "1900") is False

    def test_gte(self):
        assert _apply("1900", ">=", "1900") is True
        assert _apply("1899", ">=", "1900") is False

    def test_lte(self):
        assert _apply("1900", "<=", "1900") is True
        assert _apply("1901", "<=", "1900") is False

    def test_neq_numeric(self):
        assert _apply("1800", "!=", "1900") is True
        assert _apply("1900", "!=", "1900") is False

    def test_neq_string(self):
        assert _apply("Curie", "!=", "Einstein") is True
        assert _apply("Einstein", "!=", "Einstein") is False

    def test_string_comparison(self):
        assert _apply("B", ">", "A") is True
        assert _apply("A", ">", "B") is False
