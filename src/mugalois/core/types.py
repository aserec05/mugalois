from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Triple:
    """One triplet is (subject, predicate, object)."""
    s: str
    p: str
    o: str

    def __repr__(self):
        return f"({self.s}, {self.p}, {self.o})"


@dataclass(frozen=True)
class TriplePattern:
    """
    triplet SPARQL.
    a term is a variable (?x) or URI/littéral
    The variables begin by '?'.
    """
    s: str   # ex: "?x" ou "st:Mike"
    p: str   # toujours lié (URI) (pour l'instant), ex: "st:isFriendWith"
    o: str   # ex: "?y" ou "st:Mike"

    def is_variable(self, term: str) -> bool:
        return term.startswith("?")

    def s_is_var(self) -> bool:
        return self.is_variable(self.s)

    def o_is_var(self) -> bool:
        return self.is_variable(self.o)

    def __repr__(self):
        return f"({self.s}, {self.p}, {self.o})"


@dataclass
class Environment:
    """
    v : link each variable to its set of known values.
    Ex: v("?x") = {"Mike", "Dustin"}
    """
    _bindings: dict = field(default_factory=dict)

    def get(self, var: str) -> set:
        return set(self._bindings.get(var, set())) # emptyset in Python is set()

    def set(self, var: str, values: set):
        self._bindings[var] = set(values)

    def update(self, var: str, values: set):
        current = self.get(var)
        self._bindings[var] = current | set(values)

    def __repr__(self):
        return f"Environment({self._bindings})"
