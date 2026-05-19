# src/mugalois/core/prompts.py

from mugalois.core.types import TriplePattern, Environment
from mugalois.core.helpers import seedsOf


JSON_SCHEMA = '{"triples":[{"s":"...","p":"...","o":"..."}]}'

SYSTEM_PROMPT = f"""You are an expert RDF knowledge graph assistant.

Your job is to retrieve factual RDF triples from your knowledge with high precision.
You will be given a Context describing what is already known, and a Task to perform.

Global rules that always apply:
- Never invent, speculate, or hallucinate entities or relationships.
- Only assert triples you are certain about.
- If uncertain or no triples exist, return {{"triples": []}}.
- Always respond in valid JSON using this schema: {JSON_SCHEMA}
- Never add explanations or text outside the JSON structure."""


def _build_constraints(pattern: TriplePattern, env: Environment) -> str:
    """Build seed constraint lines to inject into prompts."""
    lines = []
    seedsS = seedsOf(pattern.s, env)
    seedsO = seedsOf(pattern.o, env)

    if pattern.s_is_var() and seedsS:
        lines.append(
            f"- {pattern.s} is one of: {', '.join(sorted(seedsS))}"
        )
    if pattern.o_is_var() and seedsO:
        lines.append(
            f"- {pattern.o} is one of: {', '.join(sorted(seedsO))}"
        )

    return "\n".join(lines)


def genTableScanPrompt(pattern: TriplePattern) -> str:
    """Prompt (i) — TableScan. No seeds."""
    return (
        f"Context: No prior information is available about {pattern.p}.\n\n"
        f"Task: Given the predicate {pattern.p}, list all known triples "
        f"({pattern.s}, {pattern.p}, {pattern.o}) that factually hold. "
        f"If you are unsure or no triples exist, return an empty list."
    )


def genSeedCrankPrompt(pattern: TriplePattern, env: Environment) -> str:
    """Prompt (ii) — SeedCrank. Seeds known on one or both sides."""
    constraints = _build_constraints(pattern, env)
    return (
        f"Context: The following values are already known:\n"
        f"{constraints}\n\n"
        f"Task: List all triples ({pattern.s}, {pattern.p}, {pattern.o}) "
        f"that factually hold, knowing that:\n"
        f"{constraints}\n"
        f"Return only factual values. "
        f"Do not introduce entities outside the sets provided."
    )


def genKeyCrankPrompt(
    pattern: TriplePattern,
    env: Environment,
    seed_value: str,
    direction: str
) -> str:
    """Prompt (iii) — KeyCrank. One seed fixed, one direction.

    direction: 'L->R'  or 'R->L'.
    One prompt instance is generated per seed value (parallelizable).
    """
    if direction == "L->R":
        other_seeds = seedsOf(pattern.o, env)

        
        constraint = ""
        if pattern.o_is_var() and other_seeds:
            constraint = f"- {pattern.o} ∈ {{{', '.join(sorted(other_seeds))}}}\n"

        return (
            f"Context: The subject is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({seed_value}, {pattern.p}, {pattern.o}) "
            f"that factually hold.\n"
            f"{constraint}"
            f"Return only factual values."
        )

    else:  # R->L
        other_seeds = seedsOf(pattern.s, env)

        constraint = ""
        if pattern.s_is_var() and other_seeds:
            constraint = f"- {pattern.s} ∈ {{{', '.join(sorted(other_seeds))}}}\n"

        return (
            f"Context: The object is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({pattern.s}, {pattern.p}, {seed_value}) "
            f"that factually hold.\n"
            f"{constraint}"
            f"Return only factual values."
        )


def genCheckPrompt(triple: tuple) -> str:
    """Prompt Check — verify a complete triple, returns yes or no.

    Used in KeyCrank when both sides are fully determined.
    """
    s, p, o = triple
    return (
        f"Context: We are verifying a single factual triple "
        f"about the predicate {p}.\n\n"
        f"Task: Is this triple ({s}, {p}, {o}) factually valid? "
        f"Respond only with 'yes' or 'no'. "
        f"If you are uncertain, answer 'no'."
    )


def genConfidencePrompt(pattern: TriplePattern, env: Environment) -> str:
    """Prompt (iv) — Confidence. choice between SeedCrank and KeyCrank.

    in Algorithm 1 choose a physical strategy.
    """
    constraints = _build_constraints(pattern, env)
    return (
        f"Context: The following values are already known:\n"
        f"{constraints}\n\n"
        f"Task: Given the triple pattern ({pattern.s}, {pattern.p}, {pattern.o}), "
        f"how confident are you to list at once, in a single answer, "
        f"all factual triples that hold? "
        f"Respond only with a number in [0, 1], "
        f"where 1 means fully confident and 0 means unable. "
        f"Do not add any comment."
    )


def genIterativePrompt(already_found: set) -> str:
    """Iterative prompt — used in TableScan and SeedCrank after the first call.

   found triples are injected to avoid repetitions
    """
    triples_str = ", ".join(str(t) for t in already_found)
    return (
        f"Context: The following triples have already been retrieved:\n"
        f"{triples_str}\n\n"
        f"Task: List more triples if there are more. "
        f"Do not repeat already retrieved values. "
        f"If there are no more, return an empty list."
    )


def build_messages(user_prompt: str) -> list:
    """Build the message list for a single LLM call.

    The system prompt is sent on every call because Ollama is stateless.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
