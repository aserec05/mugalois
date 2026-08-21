# src/mugalois/closures/alphabet.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern


class AlphabetClosure(Closure):
    """
    Alphabet — forces systematic enumeration by initial letter.
    No extra LLM call: the prompt cycles through letter ranges
    to bypass the LLM's tendency to return the most salient entities first.
    Letters cycle A-F, G-L, M-R, S-Z on successive activations.
    """

    RANGES = ["A-F", "G-L", "M-R", "S-Z"]

    def __init__(self):
        self._call_count = 0

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        letter_range = self.RANGES[self._call_count % len(self.RANGES)]
        self._call_count += 1
        already = ", ".join(sorted(str(t) for t in found)[:30])
        return (
            f"Already retrieved:\n{already}\n\n"
            f"Systematically search for MORE {pattern.p} \"{pattern.o}\" "
            f"whose names start with letters {letter_range}.\n"
            f"Go through this letter range carefully and exhaustively.\n"
            f"Do not repeat already retrieved values.\n"
            f"If there are truly no more in this range, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Alphabet"