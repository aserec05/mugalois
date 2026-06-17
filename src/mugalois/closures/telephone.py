# src/mugalois/closures/telephone.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern


class TelephoneClosure(Closure):
    """
    Telephone arabe — coach gives specific names to help the scanner.
    One external LLM call per activation.
    """

    def __init__(self, n_names: int = 5):
        self._n_names = n_names

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        found_str = ", ".join(sorted(str(t) for t in found)[:20])
        coach_prompt = (
            f"Your friend is looking for entities with "
            f"{pattern.p} \"{pattern.o}\".\n"
            f"They already found: {found_str}.\n"
            f"They are stuck. Give them {self._n_names} specific names "
            f"they haven't found yet.\n"
            f"Only names, comma-separated. No explanation."
        )
        messages = [
            {"role": "system", "content":
             "You are a helpful friend. Give only a comma-separated list of names."},
            {"role": "user", "content": coach_prompt},
        ]
        hint = tracker.chat(messages).text.strip()

        already = ", ".join(sorted(str(t) for t in found))
        return (
            f"Already retrieved:\n{already}\n\n"
            f"A friend suggests these names: {hint}\n\n"
            f"Find triples for these suggestions and any others you know. "
            f"Do not repeat already retrieved values. "
            f"If there are truly no more, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Telephone"
