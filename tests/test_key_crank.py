from mugalois.core.types import Triple, TriplePattern, Environment
from mugalois.scans.key_crank import LLMKeyCrank
from mugalois.llm.llm_client import MockLLM


def make_pattern_T2():
    return TriplePattern("?x", "st:isFriendWith", "?y")

def make_pattern_T1():
    return TriplePattern("?x", "st:isFriendWith", "st:Mike")

def make_pattern_T3():
    return TriplePattern("st:Mike", "st:isFriendWith", "?y")


# direction selection

def test_LR_when_subject_seeds_smaller():
    """When |seeds_s| < |seeds_o|, direction must be L->R."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": '{"triples":[]}'})()

    env = Environment()
    env.set("?x", {"Mike"})
    env.set("?y", {"Eleven", "Max", "Will"})
    LLMKeyCrank(make_pattern_T2(), env, llm=CaptureMock())
    assert "subject" in captured[0].lower()


def test_RL_when_object_seeds_smaller():
    """When |seeds_o| < |seeds_s|, direction must be R->L."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": '{"triples":[]}'})()

    env = Environment()
    env.set("?x", {"Mike", "Dustin", "Will"})
    env.set("?y", {"Eleven"})
    LLMKeyCrank(make_pattern_T2(), env, llm=CaptureMock())
    assert "object" in captured[0].lower()


def test_LR_when_only_subject_seeds():
    """When only subject seeds are known, direction must be L->R."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": '{"triples":[]}'})()

    env = Environment()
    env.set("?x", {"Mike", "Dustin"})
    LLMKeyCrank(make_pattern_T2(), env, llm=CaptureMock())
    assert "subject" in captured[0].lower()


def test_RL_when_only_object_seeds():
    """When only object seeds are known, direction must be R->L."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": '{"triples":[]}'})()

    env = Environment()
    env.set("?y", {"Eleven", "Max"})
    LLMKeyCrank(make_pattern_T2(), env, llm=CaptureMock())
    assert "object" in captured[0].lower()


# check prompt fallback

def test_check_prompt_used_when_T1():
    """T1 pattern: both sides determined — Check prompt must be used."""
    captured = []

    class CaptureMock(MockLLM):
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return type("R", (), {"text": "yes"})()

    env = Environment()
    env.set("?x", {"Dustin"})
    LLMKeyCrank(make_pattern_T1(), env, llm=CaptureMock())
    assert "yes" in captured[0].lower() or "no" in captured[0].lower()


def test_check_prompt_yes_adds_triple():
    """Check prompt returning 'yes' must add the triple to T."""
    env = Environment()
    env.set("?x", {"Dustin"})
    mock = MockLLM(canned="yes")
    T = LLMKeyCrank(make_pattern_T1(), env, llm=mock)
    assert Triple("Dustin", "st:isFriendWith", "st:Mike") in T


def test_check_prompt_no_does_not_add_triple():
    """Check prompt returning 'no' must not add the triple to T."""
    env = Environment()
    env.set("?x", {"Dustin"})
    mock = MockLLM(canned="no")
    T = LLMKeyCrank(make_pattern_T1(), env, llm=mock)
    assert T == set()


# results and env

def test_returns_parsed_triples():
    """KeyCrank must return triples parsed from the LLM response."""
    env = Environment()
    env.set("?x", {"Mike"})
    env.set("?y", {"Eleven", "Max"})
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    T = LLMKeyCrank(make_pattern_T2(), env, llm=mock)
    assert Triple("Mike", "st:isFriendWith", "Eleven") in T


def test_updates_env_after_keycrank():
    """KeyCrank must update the environment with discovered values."""
    env = Environment()
    env.set("?x", {"Mike"})
    env.set("?y", {"Eleven", "Max"})
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    LLMKeyCrank(make_pattern_T2(), env, llm=mock)
    assert "Eleven" in env.get("?y")


def test_fallback_to_tablescan_when_no_seeds():
    """KeyCrank must redirect to TableScan when no seeds are available."""
    env = Environment()
    mock = MockLLM(canned='{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}')
    T = LLMKeyCrank(make_pattern_T2(), env, llm=mock)
    assert Triple("Mike", "st:isFriendWith", "Eleven") in T


def test_one_call_per_seed():
    """KeyCrank must make exactly one LLM call per seed value."""
    call_count = 0

    class CountingMock(MockLLM):
        def chat(self, messages):
            nonlocal call_count
            call_count += 1
            return type("R", (), {"text": '{"triples":[]}'})()

    env = Environment()
    env.set("?x", {"Mike", "Dustin", "Doc"})
    env.set("?y", {"Eleven", "Max", "Will", "Lucas"})
    LLMKeyCrank(make_pattern_T2(), env, llm=CountingMock())
    assert call_count == 3