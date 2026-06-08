# src/mugalois/core/prompts.py

from mugalois.core.types import TriplePattern, Environment
from mugalois.core.helpers import seedsOf


# ── Schemas ───────────────────────────────────────────────────────────────────

TRIPLE_SCHEMA = '{"triples":[{"s":"...","p":"...","o":"..."}]}'
VALUE_SCHEMA  = '{"values":["..."]}'

_TRIPLE_REMIND = "Respond ONLY in valid JSON following the schema provided."
_VALUE_REMIND  = "Respond ONLY in valid JSON following the schema provided."

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an RDF knowledge graph assistant.
Using your knowledge, respond ONLY in valid JSON: {TRIPLE_SCHEMA}"""

SYSTEM_PROMPT_VALUE = f"""You are an RDF knowledge graph assistant.
Using your knowledge, respond ONLY in valid JSON: {VALUE_SCHEMA}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_constraints(pattern: TriplePattern, env: Environment,
                       side: str = "both") -> str:
    lines = []
    seedsS = seedsOf(pattern.s, env)
    seedsO = seedsOf(pattern.o, env)
    if side in ("both", "s") and pattern.s_is_var() and seedsS:
        lines.append(f"- {pattern.s} is one of: {', '.join(sorted(seedsS))}")
    if side in ("both", "o") and pattern.o_is_var() and seedsO:
        lines.append(f"- {pattern.o} is one of: {', '.join(sorted(seedsO))}")
    return "\n".join(lines)


def _bound_term(pattern: TriplePattern) -> tuple[str, str, str]:
    """Return (bound_side, bound_value, var_name) for T1 patterns."""
    if not pattern.s_is_var():
        return ("s", pattern.s, pattern.o)
    return ("o", pattern.o, pattern.s)


# ── NL baseline ───────────────────────────────────────────────────────────────

def genNLPrompt(nl_question: str) -> str:
    return (
        f"By your knowledge, {nl_question}\n"
        f"Respond ONLY in valid JSON following the schema provided."
    )


# ── SPARQL baseline ───────────────────────────────────────────────────────────

def genSPARQLPrompt(sparql_query: str) -> str:
    """SPARQL baseline — raw SPARQL query fed to the LLM."""
    return (
        f"By your knowledge, execute this SPARQL query and return all results:\n\n"
        f"{sparql_query}\n\n"
        f"Be exhaustive.\n"
        f"{_VALUE_REMIND}"
    )


# ── TripleScan ────────────────────────────────────────────────────────────────

def genTripleScanPrompt(pattern: TriplePattern, encoding: str = "pattern") -> str:
    """TripleScan — returns full triples, variable side projected by experiment.

    encoding : "pattern"     → (s, p, o) RDF pattern notation
               "constrained" → explicit fixed/variable context
               "sparql"      → SPARQL SELECT pattern
    """
    bound_side, bound_val, var_name = _bound_term(pattern)
    s, p, o = pattern.s, pattern.p, pattern.o

    if encoding == "pattern":
        return (
            f"Task: List all factual triples matching this RDF pattern:\n"
            f"({s}, {p}, {o})\n\n"
            f"Be exhaustive. Return as many results as you know.\n"
            f"{_TRIPLE_REMIND}"
        )

    if encoding == "constrained":
        if bound_side == "o":
            return (
                f"Context: The object is fixed: {bound_val}\n"
                f"         The predicate is fixed: {p}\n\n"
                f"Task: Find all values of {var_name} such that the triple\n"
                f"({var_name}, {p}, {bound_val}) factually holds.\n"
                f"Only return values you are certain about.\n"
                f"If none exist, return an empty list.\n"
                f"{_TRIPLE_REMIND}"
            )
        else:
            return (
                f"Context: The subject is fixed: {bound_val}\n"
                f"         The predicate is fixed: {p}\n\n"
                f"Task: Find all values of {var_name} such that the triple\n"
                f"({bound_val}, {p}, {var_name}) factually holds.\n"
                f"Only return values you are certain about.\n"
                f"If none exist, return an empty list.\n"
                f"{_TRIPLE_REMIND}"
            )

    if encoding == "sparql":
        return (
            f"Task: Execute this SPARQL pattern using your knowledge:\n"
            f"SELECT {var_name} WHERE {{ {s} {p} {o} }}\n\n"
            f"Return all matching triples in full (s, p, o) form.\n"
            f"Be exhaustive.\n"
            f"{_TRIPLE_REMIND}"
        )

    raise ValueError(f"Unknown encoding '{encoding}'. Use 'pattern', 'constrained', or 'sparql'.")


def genTripleScanIterativePrompt(already_found: set[str]) -> str:
    """Iterative prompt for TripleScan — avoids repetition.
    
    already_found : set[str] — already projected values (not Triples).
    """
    values = ", ".join(sorted(already_found))
    return (
        f"Context: The following values have already been retrieved:\n"
        f"{values}\n\n"
        f"Task: List more triples if there are any remaining.\n"
        f"Do not repeat already retrieved values.\n"
        f"If there are no more, return an empty list.\n"
        f"{_TRIPLE_REMIND}"
    )


# ── ValueScan ─────────────────────────────────────────────────────────────────

def genValueScanPrompt(pattern: TriplePattern, encoding: str = "pattern") -> str:
    """ValueScan — returns values directly, no triple structure.

    Same encodings as TripleScan but schema is {"values": ["..."]}.
    """
    bound_side, bound_val, var_name = _bound_term(pattern)
    s, p, o = pattern.s, pattern.p, pattern.o

    if encoding == "pattern":
        return (
            f"Task: List all values of {var_name} such that:\n"
            f"({s}, {p}, {o}) factually holds.\n\n"
            f"Be exhaustive. Return as many values as you know.\n"
            f"{_VALUE_REMIND}"
        )

    if encoding == "constrained":
        if bound_side == "o":
            return (
                f"Context: The object is fixed: {bound_val}\n"
                f"         The predicate is fixed: {p}\n\n"
                f"Task: Find all values of {var_name} such that the triple\n"
                f"({var_name}, {p}, {bound_val}) factually holds.\n"
                f"Only return values you are certain about.\n"
                f"If none exist, return an empty list.\n"
                f"{_VALUE_REMIND}"
            )
        else:
            return (
                f"Context: The subject is fixed: {bound_val}\n"
                f"         The predicate is fixed: {p}\n\n"
                f"Task: Find all values of {var_name} such that the triple\n"
                f"({bound_val}, {p}, {var_name}) factually holds.\n"
                f"Only return values you are certain about.\n"
                f"If none exist, return an empty list.\n"
                f"{_VALUE_REMIND}"
            )

    if encoding == "sparql":
        return (
            f"Task: Execute this SPARQL pattern using your knowledge:\n"
            f"SELECT {var_name} WHERE {{ {s} {p} {o} }}\n\n"
            f"Return only the values of {var_name}. Be exhaustive.\n"
            f"{_VALUE_REMIND}"
        )

    raise ValueError(f"Unknown encoding '{encoding}'. Use 'pattern', 'constrained', or 'sparql'.")


def genValueScanIterativePrompt(already_found: set[str]) -> str:
    """Iterative prompt for ValueScan — avoids repetition."""
    values = ", ".join(sorted(already_found))
    return (
        f"Context: The following values have already been retrieved:\n"
        f"{values}\n\n"
        f"Task: List more values if there are any remaining.\n"
        f"Do not repeat already retrieved values.\n"
        f"If there are no more, return an empty list.\n"
        f"{_VALUE_REMIND}"
    )


# ── Message builders ──────────────────────────────────────────────────────────

def build_messages(user_prompt: str) -> list:
    """TripleScan messages — uses TRIPLE_SCHEMA system prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def build_value_messages(user_prompt: str) -> list:
    """ValueScan / NL / SPARQL messages — uses VALUE_SCHEMA system prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_VALUE},
        {"role": "user",   "content": user_prompt},
    ]


# ── Legacy — kept for compatibility ───────────────────────────────────────────

def genTableScanPrompt(pattern: TriplePattern, context: str = "", encoding: str = "current") -> str:
    """Legacy TableScan — kept for compatibility with existing algorithms."""
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
    return (
        f"Task: List ALL triples ({pattern.s}, {pattern.p}, {pattern.o}) "
        f"that factually hold, based on your knowledge.\n\n"
        f"Think of it as answering: "
        f"\"For every entity you know, what is their {p_label}?\"\n\n"
        f"Be exhaustive. Cover well-known and lesser-known entities. "
        f"Do not stop after a few examples."
    )


def genIterativePrompt(already_found: set) -> str:
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


def genKeyCrankPrompt(pattern: TriplePattern, env: Environment,
                      seed_value: str, direction: str) -> str:
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
    else:
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
    s, p, o = triple
    return (
        f"Context: We are verifying a single factual triple about the predicate {p}.\n\n"
        f"Task: Is this triple ({s}, {p}, {o}) factually valid? "
        f"Respond only with 'yes' or 'no'. If you are uncertain, answer 'no'."
    )


def genConfidencePrompt(pattern: TriplePattern, env: Environment) -> str:
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