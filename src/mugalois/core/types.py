from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


# ── Core RDF types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Triple:
    """One RDF triple (subject, predicate, object)."""
    s: str
    p: str
    o: str

    def __repr__(self):
        return f"({self.s}, {self.p}, {self.o})"

    def __lt__(self, other: "Triple") -> bool:
        return (self.s, self.p, self.o) < (other.s, other.p, other.o)


@dataclass(frozen=True)
class TriplePattern:
    """
    SPARQL triple pattern.
    A term is a variable (?x) or a URI/literal.
    Variables start with '?'. p is always bound (URI) for now.
    """
    s: str
    p: str
    o: str

    def is_variable(self, term: str) -> bool:
        return term.startswith("?")

    def s_is_var(self) -> bool:
        return self.is_variable(self.s)

    def o_is_var(self) -> bool:
        return self.is_variable(self.o)

    def __repr__(self):
        return f"({self.s}, {self.p}, {self.o})"


# ── Conditions ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Condition:
    """
    A single SPARQL FILTER condition: var op val

    Examples:
        FILTER(?x = dbr:Einstein)  → Condition("?x", "=",  "dbr:Einstein")
        FILTER(?y != dbr:Curie)    → Condition("?y", "!=", "dbr:Curie")
        FILTER(?y > 1900)          → Condition("?y", ">",  "1900")

    Compound AND is represented as multiple Condition objects in a frozenset.
    """
    var: str
    op:  str
    val: str

    EQUALITY_OPS = {"="}
    NUMERIC_OPS  = {">", "<", ">=", "<="}
    ALL_OPS      = EQUALITY_OPS | NUMERIC_OPS | {"!="}

    def __post_init__(self):
        if self.op not in self.ALL_OPS:
            raise ValueError(
                f"Unknown operator '{self.op}'. "
                f"Valid: {sorted(self.ALL_OPS)}"
            )
        if not self.var.startswith("?"):
            raise ValueError(f"var must start with '?', got '{self.var}'")

    def is_equality(self) -> bool:
        return self.op == "="

    def is_numeric(self) -> bool:
        return self.op in self.NUMERIC_OPS

    def is_inequality(self) -> bool:
        return self.op == "!="

    def variables(self) -> set[str]:
        return {self.var}

    def __repr__(self):
        return f"FILTER({self.var} {self.op} {self.val})"


@dataclass(frozen=True)
class ConditionIN:
    """
    FILTER(?x IN {val1, val2, ...})

    RW1 initialises v(?x) directly with this finite set.
    """
    var:    str
    values: frozenset[str]

    def __post_init__(self):
        if not self.var.startswith("?"):
            raise ValueError(f"var must start with '?', got '{self.var}'")
        if not self.values:
            raise ValueError("ConditionIN requires at least one value.")

    def is_equality(self) -> bool:
        return False

    def is_finite_set(self) -> bool:
        return True

    def variables(self) -> set[str]:
        return {self.var}

    def __repr__(self):
        return f"FILTER({self.var} IN {set(self.values)})"


AnyCondition = Union[Condition, ConditionIN]


# ── Query ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Query:
    """
    SELECT return_vars WHERE { pattern } FILTER conditions

    conditions is a frozenset of AnyCondition.
    Multiple conditions = AND.
    """
    pattern:     TriplePattern
    conditions:  frozenset[AnyCondition]
    return_vars: frozenset[str]

    def __repr__(self):
        conds = " AND ".join(str(c) for c in self.conditions)
        return (
            f"SELECT {set(self.return_vars)} "
            f"WHERE {self.pattern} "
            f"FILTER {conds}"
        )


# ── Environments ──────────────────────────────────────────────────────────────

@dataclass
class Environment:
    """
    v : V → P(UL)
    Maps each variable to its set of known URI/literal values.
    """
    _bindings: dict = field(default_factory=dict)

    def get(self, var: str) -> set:
        return set(self._bindings.get(var, set()))

    def set(self, var: str, values: set) -> None:
        self._bindings[var] = set(values)

    def add(self, var: str, value) -> None:
        current = self.get(var)
        current.add(value)
        self._bindings[var] = current

    def update(self, var: str, values: set) -> None:
        self._bindings[var] = self.get(var) | set(values)

    def has(self, var: str) -> bool:
        return bool(self._bindings.get(var))

    def __repr__(self):
        return f"Environment({self._bindings})"