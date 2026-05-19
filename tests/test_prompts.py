"""In fact, I don't know hot to test prompts patrons. It's trivial and useless. Consider them as examples. """

from mugalois.core.types import TriplePattern, Environment
from mugalois.core.prompts import (
    genTableScanPrompt,
    genSeedCrankPrompt,
    genKeyCrankPrompt,
    genCheckPrompt,
    genConfidencePrompt,
    genIterativePrompt,
    build_messages,
    SYSTEM_PROMPT,
    JSON_SCHEMA,
)
 
def make_env_T2():
    env = Environment()
    env.set("?x", {"Mike", "Dustin", "Doc"})
    env.set("?y", {"Demogorgon", "Papa"})
    return env
 
 
# SeedCrank — constraint injection logic
 
def test_seedCrank_no_constraint_when_no_seeds():
    """Without seeds, no 'is one of' line should appear — avoids confusing the LLM."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genSeedCrankPrompt(tp, Environment())
    assert "is one of" not in prompt
 
 
def test_seedCrank_only_subject_constraint_when_object_is_bound():
    """T1 pattern: object is bound, so only the subject constraint should appear."""
    tp = TriplePattern("?x", "st:isFriendWith", "st:Mike")
    env = Environment()
    env.set("?x", {"Dustin", "Doc"})
    prompt = genSeedCrankPrompt(tp, env)
    # sujet contraint
    assert "?x is one of" in prompt
    # objet lié, pas de contrainte générée pour lui
    assert "?y is one of" not in prompt
 
 
def test_seedCrank_only_object_constraint_when_subject_is_bound():
    """T3 pattern: subject is bound, so only the object constraint should appear."""
    tp = TriplePattern("st:Mike", "st:isFriendWith", "?y")
    env = Environment()
    env.set("?y", {"Eleven", "Will"})
    prompt = genSeedCrankPrompt(tp, env)
    assert "?y is one of" in prompt
    assert "?x is one of" not in prompt
 
 
def test_seedCrank_both_constraints_for_T2():
    """T2 pattern: both sides are variables with seeds, both constraints must appear."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genSeedCrankPrompt(tp, make_env_T2())
    assert "?x is one of" in prompt
    assert "?y is one of" in prompt
 
 
# KeyCrank — direction logic
 
def test_keyCrank_LR_fixed_term_is_subject():
    """L->R: the seed value must appear as the fixed subject, not object."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genKeyCrankPrompt(tp, Environment(), "Mike", "L->R")
    assert "subject" in prompt.lower()
    assert "object" not in prompt.lower()
 
 
def test_keyCrank_RL_fixed_term_is_object():
    """R->L: the seed value must appear as the fixed object, not subject."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genKeyCrankPrompt(tp, Environment(), "Papa", "R->L")
    assert "object" in prompt.lower()
    assert "subject" not in prompt.lower()
 
 
def test_keyCrank_LR_injects_object_seeds_not_subject_seeds():
    """L->R: the constraint on the other side is for the object, not the subject."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genKeyCrankPrompt(tp, make_env_T2(), "Mike", "L->R")
    # seeds de l'objet doivent apparaître
    assert "?y is one of" in prompt
    # seeds du sujet ne doivent pas apparaître (c'est le côté fixé)
    assert "?x is one of" not in prompt
 
 
def test_keyCrank_RL_injects_subject_seeds_not_object_seeds():
    """R->L: the constraint on the other side is for the subject, not the object."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genKeyCrankPrompt(tp, make_env_T2(), "Papa", "R->L")
    assert "?x is one of" in prompt
    assert "?y is one of" not in prompt
 
 
def test_keyCrank_no_other_constraint_when_env_empty():
    """KeyCrank with empty env: no constraint line should appear on the other side."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genKeyCrankPrompt(tp, Environment(), "Mike", "L->R")
    assert "is one of" not in prompt
 
 
# Check — yes/no safety
 
def test_check_uncertainty_explicitly_leads_to_no():
    """The check prompt must explicitly state that uncertainty means 'no'."""
    prompt = genCheckPrompt(("X", "p", "Y"))
    assert "uncertain" in prompt
    assert "'no'" in prompt
 
 
def test_check_does_not_ask_for_json():
    """The check prompt expects yes/no, not JSON — must not mention JSON schema."""
    prompt = genCheckPrompt(("X", "p", "Y"))
    assert JSON_SCHEMA not in prompt
    assert "triples" not in prompt
 
 
# Confidence — routing
 
def test_confidence_asks_for_single_number():
    """The confidence prompt must ask for a number in [0,1], nothing else."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genConfidencePrompt(tp, make_env_T2())
    assert "[0, 1]" in prompt
    assert "Do not add any comment" in prompt
 
 
def test_confidence_does_not_ask_for_json():
    """The confidence prompt expects a number, not JSON."""
    tp = TriplePattern("?x", "st:isFriendWith", "?y")
    prompt = genConfidencePrompt(tp, make_env_T2())
    assert JSON_SCHEMA not in prompt
 
 
# Iterative — already found injection
 
def test_iterative_empty_set_does_not_crash():
    """genIterativePrompt must handle an empty already_found set gracefully."""
    prompt = genIterativePrompt(set())
    assert "Task" in prompt
    assert "repeat" in prompt
 
 
def test_iterative_injects_already_found_values():
    """Already found triples must appear in the iterative prompt context."""
    from mugalois.core.types import Triple
    already = {
        Triple("Mike", "st:isFriendWith", "Eleven"),
        Triple("Dustin", "st:isFriendWith", "Max"),
    }
    prompt = genIterativePrompt(already)
    assert "Mike" in prompt
    assert "Eleven" in prompt
 
 
# System prompt and build_messages
 
def test_system_prompt_does_not_contain_task():
    """The system prompt sets global rules only — Task sections belong in user prompts."""
    assert "Task:" not in SYSTEM_PROMPT
    assert "Context:" not in SYSTEM_PROMPT
 
 
def test_build_messages_system_is_first():
    """The system message must always be the first message sent to the LLM."""
    messages = build_messages("any prompt")
    assert messages[0]["role"] == "system"
 
 
def test_build_messages_user_content_is_unchanged():
    """The user prompt must be passed through to the message unchanged."""
    prompt = genTableScanPrompt(TriplePattern("?x", "st:isFriendWith", "?y"))
    messages = build_messages(prompt)
    assert messages[1]["content"] == prompt
