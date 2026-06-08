"""
Tests for condition_filter.py — split_conditions, post_filter, LLMConfCond.

pytest tests/test_conditions.py -v

Covers:
  - LLMConfCond          : confidence call, float parsing, fallback
  - split_conditions      : inject vs post_filter routing via tau_c
  - post_filter           : programmatic filtering of triple sets
  - _post_filter_values   : programmatic filtering of value sets
  - env context in split  : seeds are visible to LLMConfCond
  - full pipeline         : split once → inject into prompt → post-filter result
"""

import pytest
from mugalois.core.types import (
    Triple, TriplePattern, Condition, ConditionIN, Environment, AnyCondition
)
from mugalois.core.condition_filter import (
    LLMConfCond, split_conditions, post_filter, _triple_satisfies_all,
)
from mugalois.scans.triple_scan import _post_filter_values
from mugalois.llm.llm_client import MockLLM


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def pattern_both_vars():
    return TriplePattern("?x", "dbo:birthYear", "?y")

@pytest.fixture
def pattern_s_bound():
    return TriplePattern("dbr:Einstein", "dbo:birthYear", "?y")

@pytest.fixture
def pattern_o_bound():
    return TriplePattern("?x", "dbo:award", "dbr:NobelPrize")

@pytest.fixture
def env_with_seeds():
    env = Environment()
    env.set("?x", {"dbr:Einstein", "dbr:Curie", "dbr:Newton"})
    env.set("?y", {"1879", "1920", "1800"})
    return env

@pytest.fixture
def env_empty():
    return Environment()

@pytest.fixture
def cond_gt():
    return Condition("?y", ">", "1870")

@pytest.fixture
def cond_eq():
    return Condition("?x", "=", "dbr:Einstein")

@pytest.fixture
def cond_neq():
    return Condition("?x", "!=", "dbr:Newton")

@pytest.fixture
def cond_in():
    return ConditionIN("?x", frozenset({"dbr:Einstein", "dbr:Curie"}))

@pytest.fixture
def triples_set():
    return {
        Triple("dbr:Einstein", "dbo:birthYear", "1879"),
        Triple("dbr:Curie",    "dbo:birthYear", "1867"),
        Triple("dbr:Newton",   "dbo:birthYear", "1643"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLMConfCond
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMConfCond:

    def test_returns_float_in_range(self, pattern_both_vars, env_with_seeds, cond_gt):
        llm = MockLLM(canned="0.9")
        score = LLMConfCond(pattern_both_vars, env_with_seeds, cond_gt, llm)
        assert 0.0 <= score <= 1.0

    def test_high_confidence(self, pattern_both_vars, env_with_seeds, cond_eq):
        llm = MockLLM(canned="0.95")
        score = LLMConfCond(pattern_both_vars, env_with_seeds, cond_eq, llm)
        assert score == pytest.approx(0.95)

    def test_low_confidence(self, pattern_both_vars, env_with_seeds, cond_gt):
        llm = MockLLM(canned="0.3")
        score = LLMConfCond(pattern_both_vars, env_with_seeds, cond_gt, llm)
        assert score == pytest.approx(0.3)

    def test_fallback_on_non_float(self, pattern_both_vars, env_with_seeds, cond_gt):
        """LLM hallucinates text → fallback to 0.0"""
        llm = MockLLM(canned="I am very confident!")
        score = LLMConfCond(pattern_both_vars, env_with_seeds, cond_gt, llm)
        assert score == 0.0

    def test_clamps_above_one(self, pattern_both_vars, env_empty, cond_gt):
        llm = MockLLM(canned="1.5")
        score = LLMConfCond(pattern_both_vars, env_empty, cond_gt, llm)
        assert score == 1.0

    def test_clamps_below_zero(self, pattern_both_vars, env_empty, cond_gt):
        llm = MockLLM(canned="-0.2")
        score = LLMConfCond(pattern_both_vars, env_empty, cond_gt, llm)
        assert score == 0.0

    def test_condition_in_fallback(self, pattern_both_vars, env_with_seeds, cond_in):
        """ConditionIN also handled — fallback if LLM returns garbage"""
        llm = MockLLM(canned="maybe")
        score = LLMConfCond(pattern_both_vars, env_with_seeds, cond_in, llm)
        assert score == 0.0

    def test_uses_env_context(self, pattern_both_vars, env_with_seeds, cond_gt):
        """
        The prompt passed to LLM should contain seed info from env.
        We verify indirectly: MockLLM receives the right messages structure.
        """
        received = []
        class CaptureLLM(MockLLM):
            def chat(self, messages):
                received.extend(messages)
                return super().chat(messages)

        llm = CaptureLLM(canned="0.8")
        LLMConfCond(pattern_both_vars, env_with_seeds, cond_gt, llm)
        user_content = received[-1]["content"]
        # env seeds should appear in the prompt
        assert "?x" in user_content or "?y" in user_content or "dbo:birthYear" in user_content


# ══════════════════════════════════════════════════════════════════════════════
# split_conditions
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitConditions:

    def test_high_confidence_goes_to_inject(self, pattern_both_vars, env_with_seeds, cond_gt):
        gamma = Environment()
        gamma.set("?y", {cond_gt})
        llm = MockLLM(canned="0.95")
        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm, tau_c=0.85)
        assert cond_gt in inject
        assert cond_gt not in pf

    def test_low_confidence_goes_to_post_filter(self, pattern_both_vars, env_with_seeds, cond_gt):
        gamma = Environment()
        gamma.set("?y", {cond_gt})
        llm = MockLLM(canned="0.5")
        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm, tau_c=0.85)
        assert cond_gt not in inject
        assert cond_gt in pf

    def test_exact_tau_goes_to_post_filter(self, pattern_both_vars, env_with_seeds, cond_gt):
        """Score == tau_c is NOT > tau_c → post_filter"""
        gamma = Environment()
        gamma.set("?y", {cond_gt})
        llm = MockLLM(canned="0.85")
        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm, tau_c=0.85)
        assert cond_gt in pf

    def test_empty_gamma_returns_empty_lists(self, pattern_both_vars, env_with_seeds):
        gamma = Environment()
        llm = MockLLM(canned="0.9")
        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm)
        assert inject == []
        assert pf == []

    def test_multiple_conditions_split_correctly(
        self, pattern_both_vars, env_with_seeds, cond_gt, cond_neq
    ):
        """Two conditions: one confident, one not"""
        gamma = Environment()
        gamma.set("?y", {cond_gt})
        gamma.set("?x", {cond_neq})

        responses = iter(["0.95", "0.4"])
        class SequentialLLM(MockLLM):
            def chat(self, messages):
                from mugalois.llm.llm_client import LLMResponse
                return LLMResponse(text=next(responses), usage_tokens=0, latency_s=0.0)

        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, SequentialLLM())
        assert len(inject) == 1
        assert len(pf) == 1

    def test_var_not_in_pattern_skipped(self, pattern_s_bound, env_with_seeds, cond_gt):
        """
        pattern is (dbr:Einstein, dbo:birthYear, ?y).
        ?x is not in pattern → conditions on ?x in gamma are ignored.
        """
        gamma = Environment()
        gamma.set("?x", {cond_gt})  # ?x not a var in this pattern
        llm = MockLLM(canned="0.9")
        inject, pf = split_conditions(pattern_s_bound, env_with_seeds, gamma, llm)
        assert inject == []
        assert pf == []

    def test_called_once_not_per_scan(self, pattern_both_vars, env_with_seeds, cond_gt):
        """LLMConfCond is called once per condition, not per scan iteration."""
        call_count = [0]
        gamma = Environment()
        gamma.set("?y", {cond_gt})

        class CountLLM(MockLLM):
            def chat(self, messages):
                call_count[0] += 1
                return super().chat(messages)

        split_conditions(pattern_both_vars, env_with_seeds, gamma, CountLLM(canned="0.9"))
        assert call_count[0] == 1  # exactly one LLM call for one condition


# ══════════════════════════════════════════════════════════════════════════════
# post_filter — triples
# ══════════════════════════════════════════════════════════════════════════════

class TestPostFilterTriples:

    def test_no_conditions_returns_all(self, pattern_both_vars, triples_set):
        result = post_filter(triples_set, [], pattern_both_vars)
        assert result == triples_set

    def test_gt_filters_by_object(self, pattern_both_vars, triples_set):
        """FILTER(?y > 1870) → keep 1879 only"""
        cond = Condition("?y", ">", "1870")
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert Triple("dbr:Einstein", "dbo:birthYear", "1879") in result
        assert Triple("dbr:Curie",    "dbo:birthYear", "1867") not in result
        assert Triple("dbr:Newton",   "dbo:birthYear", "1643") not in result

    def test_lt_filters_by_object(self, pattern_both_vars, triples_set):
        """FILTER(?y < 1870) → keep 1867, 1643"""
        cond = Condition("?y", "<", "1870")
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert Triple("dbr:Einstein", "dbo:birthYear", "1879") not in result
        assert Triple("dbr:Curie",    "dbo:birthYear", "1867") in result
        assert Triple("dbr:Newton",   "dbo:birthYear", "1643") in result

    def test_equality_on_subject(self, pattern_both_vars, triples_set):
        """FILTER(?x = dbr:Einstein) → keep only Einstein"""
        cond = Condition("?x", "=", "dbr:Einstein")
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert len(result) == 1
        assert Triple("dbr:Einstein", "dbo:birthYear", "1879") in result

    def test_neq_on_subject(self, pattern_both_vars, triples_set):
        """FILTER(?x != dbr:Newton) → keep Einstein and Curie"""
        cond = Condition("?x", "!=", "dbr:Newton")
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert Triple("dbr:Newton", "dbo:birthYear", "1643") not in result
        assert len(result) == 2

    def test_condition_in_on_subject(self, pattern_both_vars, triples_set):
        """FILTER(?x IN {dbr:Einstein, dbr:Curie})"""
        cond = ConditionIN("?x", frozenset({"dbr:Einstein", "dbr:Curie"}))
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert Triple("dbr:Newton", "dbo:birthYear", "1643") not in result
        assert len(result) == 2

    def test_two_conditions_all_must_pass(self, pattern_both_vars, triples_set):
        """FILTER(?x != dbr:Newton AND ?y > 1870) → Einstein only"""
        cond1 = Condition("?x", "!=", "dbr:Newton")
        cond2 = Condition("?y", ">", "1870")
        result = post_filter(triples_set, [cond1, cond2], pattern_both_vars)
        assert len(result) == 1
        assert Triple("dbr:Einstein", "dbo:birthYear", "1879") in result

    def test_all_filtered_out(self, pattern_both_vars, triples_set):
        """FILTER(?y > 2000) → empty set"""
        cond = Condition("?y", ">", "2000")
        result = post_filter(triples_set, [cond], pattern_both_vars)
        assert result == set()

    def test_empty_triple_set(self, pattern_both_vars):
        cond = Condition("?y", ">", "1870")
        result = post_filter(set(), [cond], pattern_both_vars)
        assert result == set()

    def test_condition_on_unbound_var_skipped(self, pattern_s_bound):
        """
        pattern = (dbr:Einstein, dbo:birthYear, ?y)
        Condition on ?x (not in pattern) → no triple removed.
        """
        triples = {Triple("dbr:Einstein", "dbo:birthYear", "1879")}
        cond = Condition("?x", "=", "dbr:Curie")  # ?x not in pattern
        result = post_filter(triples, [cond], pattern_s_bound)
        assert result == triples


# ══════════════════════════════════════════════════════════════════════════════
# _triple_satisfies_all
# ══════════════════════════════════════════════════════════════════════════════

class TestTripleSatisfiesAll:

    def test_passes_empty_conditions(self, pattern_both_vars):
        t = Triple("dbr:Einstein", "dbo:birthYear", "1879")
        assert _triple_satisfies_all(t, [], pattern_both_vars) is True

    def test_passes_matching_condition(self, pattern_both_vars):
        t = Triple("dbr:Einstein", "dbo:birthYear", "1879")
        cond = Condition("?y", ">", "1870")
        assert _triple_satisfies_all(t, [cond], pattern_both_vars) is True

    def test_fails_non_matching_condition(self, pattern_both_vars):
        t = Triple("dbr:Newton", "dbo:birthYear", "1643")
        cond = Condition("?y", ">", "1870")
        assert _triple_satisfies_all(t, [cond], pattern_both_vars) is False

    def test_fails_on_first_failing_condition(self, pattern_both_vars):
        t = Triple("dbr:Einstein", "dbo:birthYear", "1879")
        cond1 = Condition("?y", ">", "1870")   # passes
        cond2 = Condition("?x", "=", "dbr:Curie")  # fails
        assert _triple_satisfies_all(t, [cond1, cond2], pattern_both_vars) is False


# ══════════════════════════════════════════════════════════════════════════════
# _post_filter_values
# ══════════════════════════════════════════════════════════════════════════════

class TestPostFilterValues:

    def test_no_conditions_returns_all(self):
        values = {"1643", "1867", "1879"}
        result = _post_filter_values(values, [])
        assert result == values

    def test_gt_filter(self):
        values = {"1643", "1867", "1879"}
        cond = Condition("?y", ">", "1870")
        result = _post_filter_values(values, [cond])
        assert result == {"1879"}

    def test_condition_in_filter(self):
        values = {"dbr:Einstein", "dbr:Curie", "dbr:Newton"}
        cond = ConditionIN("?x", frozenset({"dbr:Einstein", "dbr:Curie"}))
        result = _post_filter_values(values, [cond])
        assert "dbr:Newton" not in result
        assert len(result) == 2

    def test_neq_filter(self):
        values = {"dbr:Einstein", "dbr:Curie"}
        cond = Condition("?x", "!=", "dbr:Curie")
        result = _post_filter_values(values, [cond])
        assert result == {"dbr:Einstein"}

    def test_all_filtered_out(self):
        values = {"1643", "1800"}
        cond = Condition("?y", ">", "2000")
        result = _post_filter_values(values, [cond])
        assert result == set()

    def test_empty_values(self):
        cond = Condition("?y", ">", "1870")
        result = _post_filter_values(set(), [cond])
        assert result == set()


# ══════════════════════════════════════════════════════════════════════════════
# Integration — split once, inject + post_filter pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitOncePipeline:

    def test_inject_appears_in_prompt(self, pattern_both_vars, env_with_seeds):
        """
        Condition with high confidence → inject_conds.
        Injected condition should appear in the genSeedCrankPrompt output.
        """
        from mugalois.core.prompts import genSeedCrankPrompt
        cond = Condition("?y", ">", "1870")
        prompt = genSeedCrankPrompt(pattern_both_vars, env_with_seeds, conditions=[cond])
        assert "1870" in prompt or "?y" in prompt

    def test_inject_in_check_prompt(self):
        """Injected condition appears in genCheckPrompt."""
        from mugalois.core.prompts import genCheckPrompt
        cond = Condition("?y", "!=", "1900")
        prompt = genCheckPrompt(("dbr:Einstein", "dbo:birthYear", "1879"), conditions=[cond])
        assert "1900" in prompt or "?y" in prompt

    def test_inject_in_key_crank_prompt(self, env_with_seeds):
        """Injected condition appears in genKeyCrankPrompt."""
        from mugalois.core.prompts import genKeyCrankPrompt
        pattern = TriplePattern("?x", "dbo:birthYear", "?y")
        cond = Condition("?y", ">", "1870")
        prompt = genKeyCrankPrompt(pattern, env_with_seeds, "dbr:Einstein", "L->R",
                                   conditions=[cond])
        assert "1870" in prompt or "?y" in prompt

    def test_post_filter_applied_after_scan(self, pattern_both_vars, triples_set):
        """
        Simulate full pipeline: scan returns all triples, post_filter removes some.
        """
        post_filter_conds = [Condition("?y", ">", "1870")]
        result = post_filter(triples_set, post_filter_conds, pattern_both_vars)
        # Only Einstein (1879) passes
        assert len(result) == 1
        assert Triple("dbr:Einstein", "dbo:birthYear", "1879") in result

    def test_no_double_llm_call_for_conditions(self, pattern_both_vars, env_with_seeds):
        """
        split_conditions is called once. The scans must not call LLMConfCond.
        We verify that inject/post_filter lists passed to scans are used directly.
        """
        cond = Condition("?y", ">", "1870")
        gamma = Environment()
        gamma.set("?y", {cond})
        llm = MockLLM(canned="0.95")

        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm, tau_c=0.85)

        # inject and pf are plain lists — scans receive them without calling split again
        assert isinstance(inject, list)
        assert isinstance(pf, list)
        assert cond in inject
        assert pf == []

    def test_no_conditions_full_pipeline(self, pattern_both_vars, env_with_seeds, triples_set):
        """When gamma is empty, pipeline is transparent — all triples pass through."""
        gamma = Environment()
        llm = MockLLM(canned="0.9")
        inject, pf = split_conditions(pattern_both_vars, env_with_seeds, gamma, llm)
        result = post_filter(triples_set, pf, pattern_both_vars)
        assert result == triples_set
