"""Standard-library samplers for converting logits into choices."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from ._validation import coerce_logits, coerce_probabilities, random_unit_interval
from .protocols import RandomGenerator


@dataclass(frozen=True, slots=True)
class SoftmaxSampler:
    """Categorically sample temperature-scaled softmax probabilities.

    Args:
        temperature: Positive finite scaling factor. Values below one sharpen
            the distribution; values above one flatten it.

    Raises:
        ValueError: If ``temperature`` is not a positive finite real number.
    """

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise ValueError("temperature must be a positive finite real number")
        try:
            temperature = float(self.temperature)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "temperature must be a positive finite real number"
            ) from error
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a positive finite real number")
        object.__setattr__(self, "temperature", temperature)

    def probabilities(self, logits: Sequence[float]) -> tuple[float, ...]:
        """Return a numerically stable softmax distribution for ``logits``."""

        values = coerce_logits(logits)
        maximum = max(values)
        weights = tuple(
            math.exp((value - maximum) / self.temperature) for value in values
        )
        total = math.fsum(weights)
        # At least the maximum logit contributes exp(0) == 1, so total is
        # positive and finite for all validated inputs.
        return coerce_probabilities(tuple(weight / total for weight in weights))

    def sample(self, logits: Sequence[float], rng: RandomGenerator) -> int:
        """Draw and return an index from the softmax distribution."""

        probabilities = self.probabilities(logits)
        draw = random_unit_interval(rng)
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                return index
        # Rounding can leave the cumulative sum a few ulps below one.
        return len(probabilities) - 1


@dataclass(frozen=True, slots=True)
class ArgmaxSampler:
    """Deterministically select the first maximum logit."""

    def probabilities(self, logits: Sequence[float]) -> tuple[float, ...]:
        """Return a one-hot distribution at the first maximum logit."""

        values = coerce_logits(logits)
        selected = max(range(len(values)), key=values.__getitem__)
        return tuple(
            1.0 if index == selected else 0.0 for index in range(len(values))
        )

    def sample(self, logits: Sequence[float], rng: RandomGenerator) -> int:
        """Return the first maximum index; ``rng`` is intentionally unused."""

        del rng
        values = coerce_logits(logits)
        return max(range(len(values)), key=values.__getitem__)


__all__ = ["ArgmaxSampler", "SoftmaxSampler"]
