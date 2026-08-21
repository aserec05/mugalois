"""
Tests for T5 — recursive property path evaluation (rec/), using the
RecursivePattern abstraction.

pytest tests/test_rec.py -v
"""

import json
import pytest

from mugalois.core.types import Environment, RecursivePattern
from mugalois.llm.llm_client import BaseLLM, LLMResponse
from mugalois.rec.simple_rec import LLMRecScan, LLMAtomicRecConf
from mugalois.rec.fixpoint import LLMFixpointScan
from mugalois.rec.mugalois_rec import MuGaloisRec


# ══════════════════════════════════════════════════════════════════════════════
# RecursivePattern
# ══════════════════════════════════════════════════════════════════════════════

class TestRecursivePattern:

    def test_plus_pattern_seed_and_var(self):
        rp = RecursivePattern("seed", "p", "?b", "+")
        assert rp.is_plus() is True
        assert rp.is_star() is False
        assert rp.seed() == "seed"
        assert rp.var() == "?b"
        assert rp.direction() == "forward"

    def test_star_pattern_seed_and_var(self):
        rp = RecursivePattern("?a", "p", "target", "*")
        assert rp.is_star() is True
        assert rp.is_plus() is False
        assert rp.seed() == "target"
        assert rp.var() == "?a"
        assert rp.direction() == "backward"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError):
            RecursivePattern("seed", "p", "?b", "?")

    def test_both_variables_raises(self):
        """A recursive scan always needs exactly one fixed anchor."""
        with pytest.raises(ValueError):
            RecursivePattern("?a", "p", "?b", "+")

    def test_both_bound_raises(self):
        with pytest.raises(ValueError):
            RecursivePattern("a", "p", "b", "+")

    def test_repr(self):
        rp = RecursivePattern("seed", "p", "?b", "+")
        assert "p+" in repr(rp)


# ══════════════════════════════════════════════════════════════════════════════
# Test doubles
# ══════════════════════════════════════════════════════════════════════════════

class SequenceLLM(BaseLLM):
    """
    Returns one canned response per call, in order. Records every prompt
    it receives so tests can assert on call count / content.
    Raises if exhausted, to catch unexpected extra calls.
    """
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("SequenceLLM exhausted — unexpected extra call")
        text = self._responses.pop(0)
        return LLMResponse(text=text, usage_tokens=1, latency_s=0.01)


class ConstLLM(BaseLLM):
    """Always returns the same canned response. Counts calls."""
    def __init__(self, text: str):
        self.text = text
        self.n_calls = 0

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.n_calls += 1
        return LLMResponse(text=self.text, usage_tokens=1, latency_s=0.01)


def values_json(vals) -> str:
    return json.dumps({"values": sorted(vals)})


# ══════════════════════════════════════════════════════════════════════════════
# LLMRecScan / LLMAtomicRecConf
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMRecScan:

    def test_plus_pattern_returns_values(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM(values_json({"A", "B", "C"}))
        result = LLMRecScan(rp, llm)
        assert result == {"A", "B", "C"}
        assert llm.n_calls == 1

    def test_star_pattern_returns_values(self):
        rp  = RecursivePattern("?a", "p", "target", "*")
        llm = ConstLLM(values_json({"X", "Y"}))
        result = LLMRecScan(rp, llm)
        assert result == {"X", "Y"}

    def test_empty_response(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM(values_json(set()))
        result = LLMRecScan(rp, llm)
        assert result == set()


class TestLLMAtomicRecConf:

    def test_plain_float(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM("0.85")
        assert LLMAtomicRecConf(rp, llm) == pytest.approx(0.85)

    def test_json_confidence(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM(json.dumps({"confidence": 0.4}))
        assert LLMAtomicRecConf(rp, llm) == pytest.approx(0.4)

    def test_invalid_response_returns_zero(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM("not a number")
        assert LLMAtomicRecConf(rp, llm) == 0.0

    def test_clamped_above_one(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM("5.0")
        assert LLMAtomicRecConf(rp, llm) == 1.0

    def test_clamped_below_zero(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = ConstLLM("-3.0")
        assert LLMAtomicRecConf(rp, llm) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LLMFixpointScan — '+' pattern (forward, irreflexive)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMFixpointScanPlus:

    def test_single_hop_then_stops(self):
        """seed -> {B, C}; B, C have no further successors -> fixpoint."""
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"B", "C"}),   # hop 1: from {seed}
            values_json(set()),        # hop 2: from {B, C} -> nothing new
        ])
        result = LLMFixpointScan(rp, llm, max_depth=8)
        assert result == {"B", "C"}

    def test_excludes_seed_itself(self):
        """'+' is irreflexive: never returns the seed, even if hallucinated."""
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"seed", "B"}),
            values_json(set()),
        ])
        result = LLMFixpointScan(rp, llm, max_depth=8)
        assert result == {"B"}
        assert "seed" not in result

    def test_multi_hop_chain(self):
        """seed -> A -> B -> (nothing): three hops to reach fixpoint."""
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"A"}),
            values_json({"B"}),
            values_json(set()),
        ])
        result = LLMFixpointScan(rp, llm, max_depth=8)
        assert result == {"A", "B"}

    def test_stops_at_max_depth_even_if_more_remain(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"A"}),
            values_json({"B"}),
            values_json({"C"}),  # would continue, but max_depth=2 cuts here
        ])
        result = LLMFixpointScan(rp, llm, max_depth=2)
        assert result == {"A", "B"}
        assert "C" not in result

    def test_cycle_does_not_loop_forever(self):
        """A -> B -> A: must terminate because 'A' is already visited."""
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"A"}),
            values_json({"A", "B"}),
            values_json({"A"}),
        ])
        result = LLMFixpointScan(rp, llm, max_depth=10)
        assert result == {"A", "B"}

    def test_no_successors_returns_empty(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([values_json(set())])
        result = LLMFixpointScan(rp, llm)
        assert result == set()


# ══════════════════════════════════════════════════════════════════════════════
# LLMFixpointScan — '*' pattern (backward, reflexive)
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMFixpointScanStar:

    def test_includes_seed_itself(self):
        """'*' is reflexive: the target is always included in the result."""
        rp  = RecursivePattern("?a", "p", "target", "*")
        llm = SequenceLLM([values_json(set())])
        result = LLMFixpointScan(rp, llm)
        assert "target" in result

    def test_backward_expansion(self):
        """target <- B <- A: both predecessors should be found."""
        rp  = RecursivePattern("?a", "p", "target", "*")
        llm = SequenceLLM([
            values_json({"B"}),
            values_json({"A"}),
            values_json(set()),
        ])
        result = LLMFixpointScan(rp, llm, max_depth=8)
        assert result == {"target", "B", "A"}


# ══════════════════════════════════════════════════════════════════════════════
# LLMFixpointScan — batching behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMFixpointScanBatching:

    def test_batches_frontier_in_single_call(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"A", "B", "C"}),
            values_json(set()),
        ])
        LLMFixpointScan(rp, llm, max_depth=8, batch=True)
        assert len(llm.calls) == 2  # not 1 + 3

    def test_batch_prompt_mentions_all_frontier_nodes(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            values_json({"A", "B"}),
            values_json(set()),
        ])
        LLMFixpointScan(rp, llm, max_depth=8, batch=True)
        second_call_content = llm.calls[1][-1]["content"]
        assert "A" in second_call_content
        assert "B" in second_call_content


# ══════════════════════════════════════════════════════════════════════════════
# MuGaloisRec — routing
# ══════════════════════════════════════════════════════════════════════════════

class TestMuGaloisRecRouting:

    def test_high_confidence_routes_to_atomic_scan(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            "0.9",
            values_json({"A", "B"}),
        ])
        result = MuGaloisRec(rp, llm, tau_a=0.7)
        assert result == {"A", "B"}
        assert len(llm.calls) == 2

    def test_low_confidence_routes_to_fixpoint(self):
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            "0.3",
            values_json({"A"}),
            values_json(set()),
        ])
        result = MuGaloisRec(rp, llm, tau_a=0.7)
        assert result == {"A"}

    def test_confidence_exactly_at_threshold_uses_fixpoint(self):
        """Strict '>' comparison: conf == tau_a must NOT trigger atomic scan."""
        rp  = RecursivePattern("seed", "p", "?b", "+")
        llm = SequenceLLM([
            "0.7",
            values_json(set()),
        ])
        result = MuGaloisRec(rp, llm, tau_a=0.7)
        assert result == set()
        assert len(llm.calls) == 2