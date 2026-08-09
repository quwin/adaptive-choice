"""Internal validation helpers shared by samplers and orchestration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any

from .errors import (
    ActionCountMismatch,
    InvalidLogits,
    InvalidProbabilityDistribution,
    InvalidRandomValue,
)
from .protocols import RandomGenerator


def _materialize(
    values: Iterable[Any], *, name: str, error_type: type[Exception]
) -> tuple[Any, ...]:
    try:
        return tuple(values)
    except (TypeError, ValueError) as error:
        raise error_type(f"{name} must be a finite sequence of real numbers") from error


def coerce_logits(
    logits: Iterable[Any], *, expected_count: int | None = None
) -> tuple[float, ...]:
    """Freeze and validate a logit collection."""

    raw = _materialize(logits, name="logits", error_type=InvalidLogits)
    if expected_count is not None and len(raw) != expected_count:
        raise ActionCountMismatch(expected_count, len(raw), source="logits")
    if not raw:
        raise InvalidLogits("logits must contain at least one value")

    result: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidLogits(
                f"logit at index {index} must be a finite real number, got {value!r}"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidLogits(
                f"logit at index {index} must be a finite real number, got {value!r}"
            ) from error
        if not math.isfinite(converted):
            raise InvalidLogits(
                f"logit at index {index} must be finite, got {converted!r}"
            )
        result.append(converted)
    return tuple(result)


def coerce_probabilities(
    probabilities: Iterable[Any], *, expected_count: int | None = None
) -> tuple[float, ...]:
    """Freeze and validate a categorical probability distribution."""

    raw = _materialize(
        probabilities,
        name="probabilities",
        error_type=InvalidProbabilityDistribution,
    )
    if expected_count is not None and len(raw) != expected_count:
        raise ActionCountMismatch(expected_count, len(raw), source="probabilities")
    if not raw:
        raise InvalidProbabilityDistribution(
            "probabilities must contain at least one value"
        )

    result: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidProbabilityDistribution(
                f"probability at index {index} must be a finite real number, "
                f"got {value!r}"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InvalidProbabilityDistribution(
                f"probability at index {index} must be a finite real number, "
                f"got {value!r}"
            ) from error
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise InvalidProbabilityDistribution(
                f"probability at index {index} must be finite and in [0, 1], "
                f"got {converted!r}"
            )
        result.append(converted)

    total = math.fsum(result)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise InvalidProbabilityDistribution(
            f"probabilities must sum to 1 (within tolerance), got {total!r}"
        )
    return tuple(result)


def random_unit_interval(rng: RandomGenerator) -> float:
    """Get and validate one uniform variate from an explicit RNG."""

    try:
        raw = rng.random()
    except AttributeError as error:
        raise InvalidRandomValue(rng) from error
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise InvalidRandomValue(raw)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidRandomValue(raw) from error
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise InvalidRandomValue(raw)
    return value
