"""Shared dependency-free helpers for the test suite."""

from __future__ import annotations

from collections.abc import Iterable


class SequenceRandom:
    """A tiny explicit RNG whose draws are supplied by a test."""

    def __init__(self, values: Iterable[object]) -> None:
        self._values = iter(values)
        self.calls = 0

    def random(self) -> object:
        self.calls += 1
        return next(self._values)


class NeverRandom:
    """An RNG that fails a test if a component consults it."""

    def random(self) -> float:
        raise AssertionError("the RNG must not be consulted")
