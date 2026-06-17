import pytest
from mugalois.core.types import Triple, TriplePattern, Environment

def test_triple_creation():
    t = Triple(s="st:Mike", p="st:isFriendWith", o="st:Dustin")
    assert t.s == "st:Mike"
    assert t.p == "st:isFriendWith"
    assert t.o == "st:Dustin"

def test_triple_is_immutable():
    t = Triple("a", "b", "c")
    with pytest.raises(Exception):
        t.s = "z"

def test_triple_pattern_variable_detection():
    tp = TriplePattern(s="?x", p="st:isFriendWith", o="st:Mike")
    assert tp.s_is_var() is True
    assert tp.o_is_var() is False

def test_triple_pattern_T2():
    tp = TriplePattern(s="?x", p="st:isFriendWith", o="?y")
    assert tp.s_is_var() is True
    assert tp.o_is_var() is True

def test_environment_get_empty():
    env = Environment()
    assert env.get("?x") == set()

def test_environment_set_and_get():
    env = Environment()
    env.set("?x", {"Mike", "Dustin"})
    assert env.get("?x") == {"Mike", "Dustin"}

def test_environment_update():
    env = Environment()
    env.set("?x", {"Mike"})
    env.update("?x", {"Dustin", "Doc"})
    assert env.get("?x") == {"Mike", "Dustin", "Doc"}

def test_environment_update_no_duplicate():
    env = Environment()
    env.set("?x", {"Mike"})
    env.update("?x", {"Mike"})
    assert env.get("?x") == {"Mike"}
