# src/mugalois/paths/path_query.py
"""
PathQuery — data structure for T4 N-hop sequence path queries.

Reuses TriplePattern and AnyCondition from types.py — no new types.

T4 vs T5:
  T4 : s p1/p2/.../pN t  — different predicates, fixed depth
  T5 : s p+/p* t         — same predicate, variable depth
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from mugalois.core.types import TriplePattern, AnyCondition


@dataclass
class PathQuery:
    hops:       List[TriplePattern]
    target_var: str                = "?b"
    conditions: List[AnyCondition] = field(default_factory=list)

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def source(self) -> str:
        return self.hops[0].s

    @property
    def target(self) -> str:
        return self.hops[-1].o

    @property
    def intermediate_vars(self) -> List[str]:
        """Shared variables between hops (excludes source/target anchors)."""
        return [h.o for h in self.hops[:-1] if h.o_is_var()]

    def predicates(self) -> List[str]:
        return [h.p for h in self.hops]

    def _var_hop_index(self, var: str) -> int:
        for i, h in enumerate(self.hops):
            if h.o == var:
                return i
        raise ValueError(f"Variable {var!r} not found in path.")

    def left_half(self, split_var: str) -> "PathQuery":
        """Source to split_var (inclusive)."""
        idx = self._var_hop_index(split_var)
        return PathQuery(hops=self.hops[:idx + 1], target_var=split_var)

    def right_half(self, split_var: str) -> "PathQuery":
        """split_var to target (inclusive)."""
        idx = self._var_hop_index(split_var)
        return PathQuery(
            hops=self.hops[idx:],
            target_var=self.target_var,
            conditions=self.conditions,
        )

    def suffix_from(self, var: str) -> "PathQuery":
        """
        Sub-path starting at the hop that HAS var as subject.
        Used when var is already bound in gamma — skip the hop
        that produces var, evaluate the rest from var's known seeds.
        """
        idx = self._var_hop_index(var)
        return PathQuery(
            hops=self.hops[idx:],
            target_var=self.target_var,
            conditions=self.conditions,
        )

    def reversed(self) -> "PathQuery":
        return PathQuery(
            hops=[TriplePattern(h.o, h.p, h.s) for h in reversed(self.hops)],
            target_var=self.target_var,
            conditions=self.conditions,
        )

    def __str__(self) -> str:
        path_str = "/".join(h.p for h in self.hops)
        return f"({self.source}, {path_str}, {self.target_var})"

    def sparql(self) -> str:
        path_str = "/".join(h.p for h in self.hops)
        return f"SELECT {self.target_var} WHERE {{ {self.source} {path_str} {self.target_var} }}"

    @classmethod
    def two_hop(cls, s: str, p1: str, p2: str, t: str,
                var: str = "?b") -> "PathQuery":
        return cls(
            hops=[TriplePattern(s, p1, var), TriplePattern(var, p2, t)],
            target_var=var,
        )

    @classmethod
    def from_chain(cls, anchors: List[str], predicates: List[str],
                   target_var: str = "?b") -> "PathQuery":
        if len(anchors) != len(predicates) + 1:
            raise ValueError("len(anchors) must equal len(predicates) + 1")
        hops = [TriplePattern(anchors[i], predicates[i], anchors[i + 1])
                for i in range(len(predicates))]
        return cls(hops=hops, target_var=target_var)
