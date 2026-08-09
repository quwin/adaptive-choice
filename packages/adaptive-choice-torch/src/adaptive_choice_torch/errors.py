"""Errors specific to the optional PyTorch adapter boundary."""

from __future__ import annotations


class AdaptiveChoiceTorchError(ValueError):
    """Base class for malformed tensor inputs and outputs."""


class InvalidTensorValue(AdaptiveChoiceTorchError):
    """Raised when a tensor boundary value violates its documented contract."""

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"invalid {source}: {reason}")


__all__ = ["AdaptiveChoiceTorchError", "InvalidTensorValue"]
