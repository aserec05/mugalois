from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.scans.seed_crank import LLMSeedCrank
from mugalois.llm.llm_client import MockLLM


def make_pattern():
    return TriplePattern("?x", "st:isFriendWith", "?y")


def make_env_T2():
    env = Environment()
    env.set("?x", {"Mike", "Dustin", "Doc"})
    env.set("?y", {"Demogorgon", "Papa"})
    return env


def test_returns_parsed_triples():
    """SeedCrank must return the triples parsed from the LLM response."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Demogorgon"}]}')
    T = LLMSeedCrank(make_pattern(), make_env_T2(), llm=mock)
    assert Triple("Mike", "st:isFriendWith", "Demogorgon") in T


def test_returns_empty_when_llm_has_nothing():
    """SeedCrank must return an empty set when the LLM finds no triples."""
    mock = MockLLM(canned='{"triples":[]}')
    assert LLMSeedCrank(make_pattern(), make_env_T2(), llm=mock) == set()


def test_returns_empty_on_malformed_response():
    """SeedCrank must handle malformed LLM responses gracefully."""
    mock = MockLLM(canned="not json at all")
    assert LLMSeedCrank(make_pattern(), make_env_T2(), llm=mock) == set()


def test_updates_env_subject():
    """SeedCrank must update the subject variable in the environment."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Demogorgon"}]}')
    env = make_env_T2()
    LLMSeedCrank(make_pattern(), env, llm=mock)
    assert "Mike" in env.get("?x")


def test_updates_env_object():
    """SeedCrank must update the object variable in the environment."""
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Demogorgon"}]}')
    env = make_env_T2()
    LLMSeedCrank(make_pattern(), env, llm=mock)
    assert "Demogorgon" in env.get("?y")


def test_seeds_are_passed_to_prompt():
    """SeedCrank must inject seeds into the first prompt."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": '{"triples":[]}'})()

    LLMSeedCrank(make_pattern(), make_env_T2(), llm=CaptureMock())
    assert "Mike" in captured[0] or "Dustin" in captured[0] or "Doc" in captured[0]


def test_stops_at_saturation():
    """SeedCrank must stop iterating when the LLM returns no new triples."""
    call_count = 0

    class CountingMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return type("R", (), {
                    "text": '{"triples":[{"s":"Mike","p":"p","o":"Demogorgon"}]}'
                })()
            return type("R", (), {"text": '{"triples":[]}'})()

    LLMSeedCrank(make_pattern(), make_env_T2(), llm=CountingMock())
    assert call_count == 2


def test_respects_max_iter():
    """SeedCrank must not exceed max_iter calls to the LLM."""
    call_count = 0

    class InfiniteMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            return type("R", (), {
                "text": f'{{"triples":[{{"s":"E{call_count}","p":"p","o":"O"}}]}}'
            })()

    LLMSeedCrank(make_pattern(), make_env_T2(), llm=InfiniteMock(), max_iter=3)
    assert call_count == 3