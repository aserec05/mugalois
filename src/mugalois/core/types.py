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

    def __eq__(self, other: "Triple") -> bool:
        return (self.s, self.p, self.o) == (other.s, other.p, other.o)

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



@dataclass(frozen=True)
class RecursivePattern:
    """
    SPARQL recursive property path: (s, p OP, o) where OP is '+' or '*'.
 
    Same s/p/o vocabulary as TriplePattern for consistency, plus an
    `operator` field. Exactly one of s, o must be a variable — the
    other is the fixed seed the closure is computed from.
 
    Examples:
        (dbr:Einstein, :influencedBy+, ?x)  → RecursivePattern("dbr:Einstein", ":influencedBy", "?x", "+")
        (?x, :partOf*, dbr:Europe)          → RecursivePattern("?x", ":partOf", "dbr:Europe", "*")
 
    '+' is one-or-more (irreflexive): the seed itself is never included.
    '*' is zero-or-more (reflexive):  the seed itself is always included.
    """
    s: str
    p: str
    o: str
    operator: str
 
    VALID_OPERATORS = {"+", "*"}
    """
    def __post_init__(self):
        if self.operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Unknown recursive operator '{self.operator}'. "
                f"Valid: {sorted(self.VALID_OPERATORS)}"
            )
        if self.s_is_var() and self.o_is_var():
            raise ValueError(
                "RecursivePattern requires exactly one fixed side "
                f"(both s={self.s!r} and o={self.o!r} are variables)."
            )
        if not self.s_is_var() and not self.o_is_var():
            raise ValueError(
                "RecursivePattern requires exactly one variable side "
                f"(both s={self.s!r} and o={self.o!r} are bound)."
            )
        """
    def is_variable(self, term: str) -> bool:
        return term.startswith("?")
 
    def s_is_var(self) -> bool:
        return self.is_variable(self.s)
 
    def o_is_var(self) -> bool:
        return self.is_variable(self.o)
 
    def is_plus(self) -> bool:
        """One-or-more: irreflexive, seed excluded from the result."""
        return self.operator == "+"
 
    def is_star(self) -> bool:
        """Zero-or-more: reflexive, seed included in the result."""
        return self.operator == "*"
 
    def seed(self) -> str:
        """The fixed anchor the closure is computed from."""
        return self.o if self.s_is_var() else self.s
 
    def var(self) -> str:
        """The variable to resolve."""
        return self.s if self.s_is_var() else self.o
 
    def direction(self) -> str:
        """
        'forward'  : seed is on the left (s fixed, o is the var to find) —
                     scan goes (seed, p, ?y) hop by hop.
        'backward' : seed is on the right (o fixed, s is the var to find) —
                     scan goes (?y, p, seed) hop by hop.
        """
        return "forward" if self.o_is_var() else "backward"
 
    def __repr__(self):
        return f"({self.s}, {self.p}{self.operator}, {self.o})"
