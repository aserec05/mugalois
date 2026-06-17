# src/mugalois/closures/base.py
"""
Base class for closures.

A closure is a continuation strategy applied when the scanner is stuck.
It takes the current state (already found triples) and returns a prompt
to inject into the scanner's next iteration.

Closures are composable via ClosurePipeline.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from mugalois.core.types import TriplePattern


class Closure(ABC):
    """
    Abstract base class for closures.

    A closure generates a prompt for the scanner when it is stuck (empty response).
    """

    @abstractmethod
    def prompt(
        self,
        pattern:  TriplePattern,
        found:    set,
        tracker,
    ) -> str:
        """
        Generate a continuation prompt.

        pattern : the triple pattern being scanned
        found   : set of triples already retrieved
        tracker : TrackingLLM instance (for coach calls)

        Returns a string prompt to inject into the scanner.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
