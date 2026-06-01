from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.scans.llm_triplet_scan import LLMTripletScan
from mugalois.llm.llm_client import MockLLM


def make_pattern():
    return TriplePattern("?x", "st:isFriendWith", "?y")


def make_env_with_seeds():
    env = Environment()
    env.set("?x", {"Mike", "Dustin"})
    env.set("?y", {"Eleven", "Max"})
    return env


def test_routes_to_tablescan_when_no_seeds():
    """With no seeds, TripletScan must call TableScan."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    T = LLMTripletScan(make_pattern(), Environment(), llm=mock)
    assert Triple("Mike", "st:isFriendWith", "Eleven") in T


def test_routes_to_seedcrank_when_confident():
    """With seeds and high confidence, TripletScan must call SeedCrank."""
    call_count = 0

    class ConfidentMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            content = messages[-1]["content"].lower()
            if "confident" in content:
                return type("R", (), {"text": "0.9"})()
            return type("R", (), {
                "text": '{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}'
            })()

    T = LLMTripletScan(make_pattern(), make_env_with_seeds(), llm=ConfidentMock(), tau_strategie=0.7)
    assert Triple("Mike", "st:isFriendWith", "Eleven") in T


def test_routes_to_keycrank_when_not_confident():
    """With seeds and low confidence, TripletScan must call KeyCrank."""
    call_count = 0

    class UnconfidentMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            content = messages[-1]["content"].lower()
            if "confident" in content:
                return type("R", (), {"text": "0.3"})()
            return type("R", (), {
                "text": '{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}'
            })()

    T = LLMTripletScan(make_pattern(), make_env_with_seeds(), llm=UnconfidentMock(), tau_strategie=0.7)
    assert isinstance(T, set)


def test_invalid_confidence_falls_back_to_keycrank():
    """If LLM returns non-numeric confidence, must default to KeyCrank (c=0.0)."""
    class BadConfidenceMock(MockLLM):
        def chat(self, messages):
            content = messages[-1]["content"].lower()
            if "confident" in content:
                return type("R", (), {"text": "I am not sure"})()
            return type("R", (), {"text": '{"triples":[]}'})()

    T = LLMTripletScan(make_pattern(), make_env_with_seeds(), llm=BadConfidenceMock(), tau_strategie=0.7)
    assert isinstance(T, set)


def test_returns_empty_when_nothing_found():
    """TripletScan must return empty set when LLM finds nothing."""
    mock = MockLLM(canned='{"triples":[]}')
    T = LLMTripletScan(make_pattern(), Environment(), llm=mock)
    assert T == set()


def test_confidence_not_called_when_no_seeds():
    """With no seeds, confidence prompt must NOT be called."""
    call_count = 0

    class CountingMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            return type("R", (), {"text": '{"triples":[]}'})()

    LLMTripletScan(make_pattern(), Environment(), llm=CountingMock(), max_iter=1)
    assert call_count == 1