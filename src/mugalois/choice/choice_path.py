# src/mugalois/choice/choice_path.py
"""
ChoicePath — dataclass for choice path queries (T6).

Represents: s  (p1 | p2 | ... | pN)  ?b
OR (backward):  ?b  (p1 | p2 | ... | pN)  o

direction = "forward"  : source [p1|p2|...] ?b   (source is subject)
direction = "backward" : ?b [p1|p2|...] source    (?b is subject)

Each branch can be:
  - str                    → 1-hop predicate (simple)
  - PathQuery              → sequential multi-hop (T4)
  - RecursivePattern       → recursive closure (T5)
  - tuple(PathQuery, Rec.) → hybrid sequential+recursive

LLMChoiceConf     — confidence that LLM can answer all branches holistically.
LLMEstimateChoiceSize — estimated number of results across all branches.
"""
from __future__ import annotations
import re
from mugalois.core.prompts import build_messages
from mugalois.llm.llm_client import BaseLLM


class ChoicePath:
    """
    Represents a choice path query.

    forward  (default): source [p1|...|pN] ?target_var
    backward           : ?target_var [p1|...|pN] source
    """
    def __init__(
        self,
        source:     str,
        target_var: str,
        branches:   list,           # list of str, PathQuery, RecursivePattern, tuple
        direction:  str = "forward" # "forward" or "backward"
    ):
        self.source     = source
        self.target_var = target_var
        self.branches   = branches
        self.direction  = direction

    def _branches_nl(self) -> str:
        """Natural language OR representation of all branches."""
        parts = []
        for b in self.branches:
            if isinstance(b, str):
                parts.append(b)
            elif hasattr(b, "hops"):
                parts.append(" → ".join(h.p for h in b.hops))
            elif hasattr(b, "p"):
                op = "+" if getattr(b, "operator", "plus") == "plus" else "*"
                parts.append(f"{b.p}{op}")
            elif isinstance(b, tuple):
                seq, rec = b
                op = "+" if getattr(rec, "operator", "plus") == "plus" else "*"
                parts.append(" → ".join(h.p for h in seq.hops) + f" → {rec.p}{op}")
        return " OR ".join(parts)

    def subject_str(self) -> str:
        """Returns the subject of the triple (fixed node or ?target_var)."""
        return self.source if self.direction == "forward" else self.target_var

    def object_str(self) -> str:
        """Returns the object of the triple (fixed node or ?target_var)."""
        return self.target_var if self.direction == "forward" else self.source

    def __repr__(self):
        arrow = "→" if self.direction == "forward" else "←"
        return (f"({self.source} {arrow} "
                f"[{self._branches_nl()}] {arrow} {self.target_var})")


def LLMChoiceConf(path: ChoicePath, llm: BaseLLM) -> float:
    """
    Single focused call: confidence [0,1] that the LLM can answer
    all branches holistically in one call.
    """
    s, o = path.subject_str(), path.object_str()
    prompt = (
        f"You need to find all values of {path.target_var} such that:\n"
        f"{s} [{path._branches_nl()}] {o}\n\n"
        f"This means {path.target_var} can satisfy ANY of these relations.\n"
        f"How confident are you (0 to 1) that you can list ALL such "
        f"{path.target_var} in a single answer?\n"
        f"Answer with a single float only. No explanation."
    )
    try:
        resp = llm.chat(build_messages(prompt))
        match = re.search(r"0?\.\d+|1\.0|^[01]$", resp.text.strip())
        return max(0.0, min(1.0, float(match.group()))) if match else 0.0
    except Exception as e:
        import warnings; warnings.warn(f"[LLMChoiceConf] content filter or error: {e}")
        return 0.0


def LLMEstimateChoiceSize(path: ChoicePath, llm: BaseLLM,
                          max_estimate: int = 50) -> int:
    """
    Single focused call: estimated number of results across all branches.
    Uses explicit triples (same format as holistic prompt) to avoid
    the LLM counting predicates instead of entities.
    Capped at max_estimate to prevent motivational hallucination.
    """
    triples = []
    for b in path.branches:
        if isinstance(b, str):
            if path.direction == "forward":
                triples.append(f"{path.source} {b} {path.target_var}")
            else:
                triples.append(f"{path.target_var} {b} {path.source}")
        elif hasattr(b, "hops"):
            chain = " → ".join(h.p for h in b.hops)
            triples.append(f"{path.source} {chain} {path.target_var}")
        elif hasattr(b, "p"):
            op = "+" if getattr(b, "operator", "plus") == "plus" else "*"
            triples.append(f"{path.source} {b.p}{op} {path.target_var}")

    n_branches = len(triples)
    subject    = path.source if path.direction == "forward" else path.target_var
    prompt  = (
        f"Approximately how many distinct entities are associated with "
        f"{subject} via {n_branches} different relation(s) in total?\n"
        f"Relations: {'; '.join(triples)}\n\n"
        f"Give a single integer estimate only. No explanation."
    )
    try:
        resp  = llm.chat(build_messages(prompt))
        match = re.search(r"\d+", resp.text.strip())
        val = max(0, int(match.group())) if match else 0
        return min(val, max_estimate)
    except Exception as e:
        import warnings; warnings.warn(f"[LLMEstimateChoiceSize] content filter or error: {e}, returning 0")
        return 0


def LLMBranchSimilarity(path: ChoicePath, llm: BaseLLM) -> str:
    """
    Single focused call: are the branches semantically similar or distinct?

    Similar  : branches describe the same type of relation from different angles
               e.g. "won | lost final | finished third" — all about tournament results
               → parallel LLMScan per branch is safer (avoids over-broad holistic)

    Distinct : branches describe fundamentally different relations
               e.g. "spouse | employer | field of work" — different concept domains
               → Holistic OR works well (LLM handles diverse concepts together)

    Returns "similar" or "distinct".
    """
    branches_str = " | ".join(
        b if isinstance(b, str) else
        (" → ".join(h.p for h in b.hops) if hasattr(b, "hops") else str(b))
        for b in path.branches
    )
    prompt = (
        f"Consider these relations between {path.source} and {path.target_var}:\n"
        f"{branches_str}\n\n"
        f"Are these relations semantically SIMILAR (same domain, same type of concept) "
        f"or DISTINCT (different domains, different types of concepts)?\n\n"
        f"Answer with a single word: 'similar' or 'distinct'. No explanation."
    )
    try:
        resp = llm.chat(build_messages(prompt))
        text = resp.text.strip().lower()
        return "similar" if "similar" in text else "distinct"
    except Exception as e:
        import warnings; warnings.warn(f"[LLMBranchSimilarity] content filter or error: {e}, defaulting to 'distinct'")
        return "distinct"