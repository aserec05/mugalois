# src/mugalois/closures/coach.py
from __future__ import annotations
from mugalois.closures.base import Closure
from mugalois.core.types import TriplePattern


class CoachClosure(Closure):
    """
    Coach provides a search direction (no specific names).
    One external LLM call per activation.
    """

    def prompt(self, pattern: TriplePattern, found: set, tracker) -> str:
        found_str = ", ".join(sorted(str(t) for t in found)[:20])
        coach_prompt = (
            f"A friend is trying to find all entities with "
            f"{pattern.p} \"{pattern.o}\".\n"
            f"They already found: {found_str}.\n"
            f"They are stuck. Help them explore a new direction.\n"
            f"Respond in ONE sentence starting with: "
            f"\"You should explore...\"\n"
            f"Give only a direction, no specific names."
        )
        messages = [
            {"role": "system", "content":
             "You are a helpful friend. One sentence, direction only, no names."},
            {"role": "user", "content": coach_prompt},
        ]
        hint = tracker.chat(messages).text.strip()

        already = ", ".join(sorted(str(t) for t in found))
        return (
            f"Already retrieved:\n{already}\n\n"
            f"A friend suggests: {hint}\n\n"
            f"Find more triples following this direction. "
            f"Do not repeat already retrieved values. "
            f"If there are truly no more, return an empty list.\n"
            f"Respond ONLY in valid JSON following the schema provided."
        )

    @property
    def name(self) -> str:
        return "Coach"
