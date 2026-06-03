# src/mugalois/core/prompts.py

from mugalois.core.types import TriplePattern, Environment
from mugalois.core.helpers import seedsOf


JSON_SCHEMA = '{"triples":[{"s":"...","p":"...","o":"..."}]}'

SYSTEM_PROMPT = f"""You are an expert RDF knowledge graph assistant.

Your job is to retrieve factual RDF triples from your knowledge with high precision.
You will be given a Context describing what is already known, and a Task to perform.

Global rules that always apply:
- Draw on your knowledge to provide factual triples.
- Only assert triples you are confident about.
- Always respond in valid JSON using this schema: {JSON_SCHEMA}
- Never add explanations or text outside the JSON structure."""


def _build_constraints(pattern: TriplePattern, env: Environment,
                       side: str = "both") -> str:
    """Build seed constraint lines to inject into prompts.

    side: 'both' (default), 's' (subject only), 'o' (object only).
    """
    lines = []
    seedsS = seedsOf(pattern.s, env)
    seedsO = seedsOf(pattern.o, env)

    if side in ("both", "s") and pattern.s_is_var() and seedsS:
        lines.append(f"- {pattern.s} is one of: {', '.join(sorted(seedsS))}")
    if side in ("both", "o") and pattern.o_is_var() and seedsO:
        lines.append(f"- {pattern.o} is one of: {', '.join(sorted(seedsO))}")

    return "\n".join(lines)


def genTableScanPrompt(pattern: TriplePattern, context: str = "", encoding: str = "current") -> str:
    """Prompt (i) — TableScan. No seeds.
 
    encoding : "sparql"   → SELECT ?x ?y WHERE { ?x p ?y }
               "pattern"  → (?x, p, ?y)
               "current"  → NL-style (default, current behaviour)
    """
    p_label = pattern.p.split(":")[-1] if ":" in pattern.p else pattern.p
 
    if context:
        return (
            f"Context: Here are known triples about {pattern.p}:\n\n"
            f"{context}\n\n"
            f"Task: Based on this context AND your knowledge, list ALL triples "
            f"({pattern.s}, {pattern.p}, {pattern.o}) that factually hold. "
            f"Be exhaustive. Do not stop after a few examples."
        )
 
    if encoding == "sparql":
        return (
            f"Execute the following SPARQL query using your knowledge and return all results:\n\n"
            f"SELECT {pattern.s} {pattern.o} WHERE {{ {pattern.s} {pattern.p} {pattern.o} }}\n\n"
            f"Be exhaustive. Return as many results as you know."
        )
 
    if encoding == "pattern":
        return (
            f"List all factual triples matching this RDF pattern:\n\n"
            f"({pattern.s}, {pattern.p}, {pattern.o})\n\n"
            f"Be exhaustive. Return as many results as you know."
        )
 
    # current — NL style
    return (
        f"Task: List ALL triples ({pattern.s}, {pattern.p}, {pattern.o}) "
        f"that factually hold, based on your knowledge.\n\n"
        f"Think of it as answering: "
        f"\"For every entity you know, what is their {p_label}?\"\n\n"
        f"Be exhaustive. Cover well-known and lesser-known entities. "
        f"Do not stop after a few examples."
    )


def genIterativePrompt(already_found: set) -> str:
    """Iterative prompt — conversational continuation.
    
    Shows already covered subjects to avoid repetition and hallucination.
    """
    subjects = ", ".join(sorted({t.s for t in already_found}))
    return (
        f"Already covered subjects : {subjects}\n\n"
        f"Task: List triples for subjects NOT in the list above. "
        f"Only return triples you are confident about. "
        f"If there are truly no more, return an empty list."
    )


def genSeedCrankPrompt(pattern: TriplePattern, env: Environment) -> str:
    constraints = _build_constraints(pattern, env)
    constraints = constraints.replace("is one of", "may be one of")
    return (
        f"Context: The following values are already known:\n"
        f"{constraints}\n\n"
        f"Task: List all triples ({pattern.s}, {pattern.p}, {pattern.o}) that factually hold.\n"
        f"- {pattern.s} must be the subject. {pattern.o} must be the object.\n"
        f"- Use exactly {pattern.p} as predicate. No variation.\n"
        f"- Only use values from the lists above that are factually correct. "
        f"If no triple holds, return an empty list."
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
    Uses soft constraint ('may be one of') + exit clause to reduce hallucinations.
    """
    if direction == "L->R":
        constraint = _build_constraints(pattern, env, side="o")
        constraint = constraint.replace("is one of", "may be one of")
        return (
            f"Context: The subject is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({seed_value}, {pattern.p}, {pattern.o}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Only return values from the list above that are factually correct. "
            f"If the subject has no valid {pattern.p}, or none of the candidate "
            f"values apply, return an empty list."
        )

    else:  # R->L
        constraint = _build_constraints(pattern, env, side="s")
        constraint = constraint.replace("is one of", "may be one of")
        return (
            f"Context: The object is fixed: {seed_value}. "
            f"The predicate is {pattern.p}.\n\n"
            f"Task: List all triples ({pattern.s}, {pattern.p}, {seed_value}) "
            f"that factually hold.\n"
            f"{constraint}\n"
            f"Only return values from the list above that are factually correct. "
            f"If no subject has {pattern.p} equal to {seed_value}, or none of "
            f"the candidate values apply, return an empty list."
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
        f"Respond only with a float between 0 and 1, "
        f"where 1 means fully confident and 0 means unable. "
        f"Do not add any comment."
    )




def build_messages(user_prompt: str) -> list:
    """Build the message list for a single LLM call.

    The system prompt is sent on every call because Ollama is stateless.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
