# src/mugalois/closures/contrast.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern


class ContrastClosure(Closure):
    """
    Contrast — explicitly targets the popularity bias.
    No extra LLM call: the prompt itself reframes the search
    toward underrepresented, obscure, or non-salient entities.
    """

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        already = ", ".join(sorted(str(t) for t in found)[:30])
        return (
            f"Already retrieved:\n{already}\n\n"
            f"The list above likely over-represents well-known, "
            f"frequently cited, or Western-centric entities.\n"
            f"Now focus on the ones that are often overlooked:\n"
            f"  - lesser-known, minor, or historically obscure\n"
            f"  - from underrepresented regions, periods, or domains\n"
            f"  - rarely mentioned in mainstream sources\n\n"
            f"Find more {pattern.p} \"{pattern.o}\" triples that are "
            f"NOT in the list above.\n"
            f"If there are truly no more, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Contrast"