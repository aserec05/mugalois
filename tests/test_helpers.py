import pytest
from mugalois.core.types import Triple, Environment
from mugalois.core.helpers import seedsOf, updateEnv

# ----- seedsOf ──────────────────────────────────────────────────────────────────

def test_seedsOf_variable_avec_valeurs():
    env = Environment()
    env.set("?x", {"Mike", "Dustin"})
    assert seedsOf("?x", env) == {"Mike", "Dustin"}

def test_seedsOf_variable_vide():
    """Variable but without known values → emptyset."""
    env = Environment()
    assert seedsOf("?x", env) == set()

def test_seedsOf_uri_liee():
    """A free term is not a variable → singleton."""
    env = Environment()
    assert seedsOf("st:Mike", env) == {"st:Mike"}

def test_seedsOf_uri_liee_ignore_env():
    """Even if env is not empty, a free term returns just {term}."""
    env = Environment()
    env.set("st:Mike", {"quelquechose"})   # ne devrait jamais arriver
    assert seedsOf("st:Mike", env) == {"st:Mike"}

# -------- updateEnv ───────────────────────────────

def test_updateEnv_T2_les_deux_variables():
    """
    Pattern T2 : ?x :isFriendWith ?y
    """
    env = Environment()
    triples = {
        Triple("Mike",   "st:isFriendWith", "Eleven"),
        Triple("Dustin", "st:isFriendWith", "Max"),
    }
    updateEnv(env, triples, s="?x", o="?y")

    assert env.get("?x") == {"Mike", "Dustin"}
    assert env.get("?y") == {"Eleven", "Max"}

def test_updateEnv_T1_sujet_variable():
    """
    Pattern T1 : ?x :isFriendWith st:Mike. Subject is variable.
    """
    env = Environment()
    triples = {
        Triple("Dustin", "st:isFriendWith", "st:Mike"),
        Triple("Doc",    "st:isFriendWith", "st:Mike"),
    }
    updateEnv(env, triples, s="?x", o="st:Mike")

    assert env.get("?x") == {"Dustin", "Doc"}
    assert env.get("?y") == set()   # jamais touché

def test_updateEnv_T3_objet_variable():
    """
    Pattern T3 : st:Mike :isFriendWith ?y. Object is variable.
    """
    env = Environment()
    triples = {
        Triple("st:Mike", "st:isFriendWith", "Eleven"),
        Triple("st:Mike", "st:isFriendWith", "Will"),
    }
    updateEnv(env, triples, s="st:Mike", o="?y")

    assert env.get("?y") == {"Eleven", "Will"}
    assert env.get("?x") == set()   # jamais touché

def test_updateEnv_accumule_les_appels():
    """
    updateEnv should union the values!!
    important for itérations of KeyCrank.
    """
    env = Environment()

    triples1 = {Triple("Mike", "st:isFriendWith", "Eleven")}
    updateEnv(env, triples1, s="?x", o="?y")

    triples2 = {Triple("Dustin", "st:isFriendWith", "Max")}
    updateEnv(env, triples2, s="?x", o="?y")

    assert env.get("?x") == {"Mike", "Dustin"}
    assert env.get("?y") == {"Eleven", "Max"}

def test_updateEnv_triples_vides():
    """If T is empty, env don't move."""
    env = Environment()
    env.set("?x", {"Mike"})
    updateEnv(env, set(), s="?x", o="?y")
    assert env.get("?x") == {"Mike"}
