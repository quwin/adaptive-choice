"""Domain-independent errors raised by :mod:`adaptive_choice`."""

from __future__ import annotations

from typing import Any


class AdaptiveChoiceError(Exception):
    """Base class for errors caused by a decision-protocol violation."""


class NoLegalActions(AdaptiveChoiceError):
    """Raised when an environment offers no action for a decision step."""

    def __init__(self) -> None:
        super().__init__("the environment returned no legal actions")


class InvalidLogits(AdaptiveChoiceError):
    """Raised when a choice model or sampler receives invalid logits."""


class ActionCountMismatch(AdaptiveChoiceError):
    """Raised when per-action data does not match the legal action count."""

    def __init__(self, expected: int, actual: int, *, source: str = "logits") -> None:
        self.expected = expected
        self.actual = actual
        self.source = source
        super().__init__(
            f"{source} count does not match legal action count "
            f"(expected {expected}, got {actual})"
        )


class InvalidProbabilityDistribution(AdaptiveChoiceError):
    """Raised when sampler probabilities are not a finite distribution."""


class InvalidChoiceIndex(AdaptiveChoiceError):
    """Raised when a sampler returns a non-integral or out-of-range index."""

    def __init__(self, index: Any, action_count: int) -> None:
        self.index = index
        self.action_count = action_count
        super().__init__(
            f"sampler returned invalid index {index!r} for {action_count} actions"
        )


class InvalidRandomValue(AdaptiveChoiceError):
    """Raised when an RNG returns a value outside the interval ``[0, 1)``."""

    def __init__(self, value: Any) -> None:
        self.value = value
        super().__init__(
            f"rng.random() must return a finite real value in [0, 1), got {value!r}"
        )


__all__ = [
    "ActionCountMismatch",
    "AdaptiveChoiceError",
    "InvalidChoiceIndex",
    "InvalidLogits",
    "InvalidProbabilityDistribution",
    "InvalidRandomValue",
    "NoLegalActions",
]
