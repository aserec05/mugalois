from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.scans.table_scan import LLMTableScan
from mugalois.llm.llm_client import MockLLM


def make_pattern():
    return TriplePattern("?x", "st:isFriendWith", "?y")


def test_returns_parsed_triples():
    """TableScan must return the triples parsed from the LLM response."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    T = LLMTableScan(make_pattern(), Environment(), llm=mock)
    assert Triple("Mike", "st:isFriendWith", "Eleven") in T


def test_returns_empty_when_llm_has_nothing():
    """TableScan must return an empty set when the LLM finds no triples."""
    mock = MockLLM(canned='{"triples":[]}')
    assert LLMTableScan(make_pattern(), Environment(), llm=mock) == set()


def test_returns_empty_on_malformed_response():
    """TableScan must handle malformed LLM responses gracefully."""
    mock = MockLLM(canned="not json at all")
    assert LLMTableScan(make_pattern(), Environment(), llm=mock) == set()


def test_updates_env_subject():
    """TableScan must update the subject variable in the environment."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    env = Environment()
    LLMTableScan(make_pattern(), env, llm=mock)
    assert "Mike" in env.get("?x")


def test_updates_env_object():
    """TableScan must update the object variable in the environment."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    env = Environment()
    LLMTableScan(make_pattern(), env, llm=mock)
    assert "Eleven" in env.get("?y")


def test_does_not_update_env_for_bound_terms():
    """TableScan must not update the env for bound (non-variable) terms."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"st:Mike"}]}')
    env = Environment()
    pattern = TriplePattern("?x", "st:isFriendWith", "st:Mike")
    LLMTableScan(pattern, env, llm=mock)
    assert "Mike" in env.get("?x")
    assert env.get("?y") == set()


def test_stops_at_saturation():
    """TableScan must stop iterating when the LLM returns no new triples."""
    call_count = 0

    class CountingMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return type("R", (), {
                    "text": '{"triples":[{"s":"Mike","p":"p","o":"Eleven"}]}'
                })()
            return type("R", (), {"text": '{"triples":[]}'})()

    LLMTableScan(make_pattern(), Environment(), llm=CountingMock())
    assert call_count == 2


def test_respects_max_iter():
    """TableScan must not exceed max_iter calls to the LLM."""
    call_count = 0

    class InfiniteMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            return type("R", (), {
                "text": f'{{"triples":[{{"s":"E{call_count}","p":"p","o":"O"}}]}}'
            })()

    LLMTableScan(make_pattern(), Environment(), llm=InfiniteMock(), max_iter=3)
    assert call_count == 3
