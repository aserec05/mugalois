# src/mugalois/closures/motivational.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern

MOTIVATIONAL_PROMPT = (
    "You can do it! Dig deeper into your memory — "
    "list entities you know that are less well-known. "
    "Push beyond the obvious ones!"
)


class MotivationalClosure(Closure):
    """
    Encourages the scanner with a motivational prompt.
    No external LLM call — token-efficient.
    """

    def __init__(self, prompt_text: str = MOTIVATIONAL_PROMPT):
        self._prompt_text = prompt_text

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        already = ", ".join(sorted(str(t) for t in found))
        return (
            f"Already retrieved:\n{already}\n\n"
            f"{self._prompt_text}\n"
            f"Find more triples. Do not repeat already retrieved values. "
            f"If there are truly no more, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Motivational"
