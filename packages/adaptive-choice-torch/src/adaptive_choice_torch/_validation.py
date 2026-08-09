"""Shared validation for tensors crossing the adapter boundary."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .errors import InvalidTensorValue


def require_tensor(value: Any, *, source: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise InvalidTensorValue(source, "expected a torch.Tensor")
    return value


def require_floating(value: Tensor, *, source: str) -> None:
    if not value.is_floating_point():
        raise InvalidTensorValue(source, "expected a real floating-point tensor")


def require_finite(value: Tensor, *, source: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise InvalidTensorValue(source, "all values must be finite")


__all__ = ["require_finite", "require_floating", "require_tensor"]
