# src/mugalois/closures/socratic.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern


class SocraticClosure(Closure):
    """
    Socratic — uses self-questioning to expose blind spots.
    No extra LLM call: the prompt asks the LLM to reason about
    what categories or dimensions it may have missed, then retrieve them.
    General enough to work across any domain.
    """

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        already = ", ".join(sorted(str(t) for t in found)[:30])
        n = len(found)
        return (
            f"Already retrieved ({n} entities):\n{already}\n\n"
            f"Before continuing, reflect critically:\n"
            f"  - What categories, groups, or dimensions have I NOT yet explored?\n"
            f"  - Are there systematic gaps (temporal, geographic, thematic)?\n"
            f"  - Am I over-representing certain well-known examples?\n\n"
            f"Based on this reflection, find MORE {pattern.p} \"{pattern.o}\" "
            f"triples that fill the identified gaps.\n"
            f"Do not repeat already retrieved values.\n"
            f"If there are truly no more, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Socratic"