"""Mask-preserving helpers for dynamically sized tensor candidate batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import require_finite, require_floating, require_tensor
from .errors import InvalidTensorValue


def masked_logits_to_rows(
    logits: Tensor,
    candidate_mask: Tensor,
) -> tuple[tuple[float, ...], ...]:
    """Convert padded logits into ordered scalar rows, excluding padding."""

    logits = require_tensor(logits, source="logits")
    candidate_mask = require_tensor(candidate_mask, source="candidate_mask")
    if logits.ndim != 2:
        raise InvalidTensorValue("logits", "expected a [batch, candidates] matrix")
    if candidate_mask.ndim != 2 or candidate_mask.shape != logits.shape:
        raise InvalidTensorValue(
            "candidate_mask", "must have the same two-dimensional shape as logits"
        )
    if candidate_mask.dtype != torch.bool:
        raise InvalidTensorValue("candidate_mask", "expected dtype torch.bool")
    if candidate_mask.device != logits.device:
        raise InvalidTensorValue("candidate_mask", "must share the logits device")
    require_floating(logits, source="logits")

    rows: list[tuple[float, ...]] = []
    for row_index in range(logits.shape[0]):
        selected = logits[row_index][candidate_mask[row_index]]
        if selected.numel() == 0:
            raise InvalidTensorValue(
                "candidate_mask", f"row {row_index} selects no candidates"
            )
        require_finite(selected, source=f"logits row {row_index}")
        values = selected.detach().to(device="cpu", dtype=torch.float64).tolist()
        rows.append(tuple(float(value) for value in values))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class PaddedCandidates:
    """A padded candidate tensor and its authoritative boolean mask."""

    values: Tensor
    mask: Tensor
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        values = require_tensor(self.values, source="padded candidate values")
        mask = require_tensor(self.mask, source="padded candidate mask")
        if values.ndim < 2:
            raise InvalidTensorValue(
                "padded candidate values", "expected batch and candidate dimensions"
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise InvalidTensorValue(
                "padded candidate values", 
                "batch and candidate dimensions must be non-empty"
            )
        if mask.ndim != 2 or mask.shape != values.shape[:2]:
            raise InvalidTensorValue(
                "padded candidate mask",
                "must match the values batch and candidate dimensions",
            )
        if mask.dtype != torch.bool:
            raise InvalidTensorValue(
                "padded candidate mask", "expected dtype torch.bool"
            )
        if mask.device != values.device:
            raise InvalidTensorValue(
                "padded candidate mask", "must share the values device"
            )
        if len(self.lengths) != values.shape[0]:
            raise InvalidTensorValue(
                "candidate lengths", "must contain one length per batch row"
            )
        for index, length in enumerate(self.lengths):
            selected = int(mask[index].sum().item())
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise InvalidTensorValue(
                    "candidate lengths", f"row {index} must have a positive length"
                )
            if length != selected:
                raise InvalidTensorValue(
                    "candidate lengths",
                    f"row {index} length {length} does not match "
                    f"{selected} mask values",
                )

    def logits_to_rows(self, logits: Tensor) -> tuple[tuple[float, ...], ...]:
        """Remove this batch's padding from a matching logit matrix."""

        return masked_logits_to_rows(logits, self.mask)


def pad_candidates(
    rows: Sequence[Tensor],
    *,
    padding_value: float = 0.0,
) -> PaddedCandidates:
    """Pad non-empty candidate tensors without changing row order."""

    materialized = tuple(rows)
    if not materialized:
        raise InvalidTensorValue("candidate rows", "batch must not be empty")
    first = require_tensor(materialized[0], source="candidate row 0")
    if first.ndim == 0:
        raise InvalidTensorValue("candidate row 0", "needs a candidate dimension")
    trailing_shape = first.shape[1:]
    lengths: list[int] = []
    for index, row_value in enumerate(materialized):
        row = require_tensor(row_value, source=f"candidate row {index}")
        if row.ndim == 0 or row.shape[0] == 0:
            raise InvalidTensorValue(
                f"candidate row {index}", "must contain at least one candidate"
            )
        if row.shape[1:] != trailing_shape:
            raise InvalidTensorValue(
                f"candidate row {index}", "feature shapes must match across the batch"
            )
        if row.dtype != first.dtype:
            raise InvalidTensorValue(
                f"candidate row {index}", "dtype must match the first row"
            )
        if row.device != first.device:
            raise InvalidTensorValue(
                f"candidate row {index}", "device must match the first row"
            )
        lengths.append(row.shape[0])

    max_candidates = max(lengths)
    values = first.new_full(
        (len(materialized), max_candidates, *trailing_shape),
        padding_value,
    )
    mask = torch.zeros(
        (len(materialized), max_candidates),
        dtype=torch.bool,
        device=first.device,
    )
    for index, row in enumerate(materialized):
        length = row.shape[0]
        values[index, :length] = row
        mask[index, :length] = True
    return PaddedCandidates(values=values, mask=mask, lengths=tuple(lengths))


__all__ = ["PaddedCandidates", "masked_logits_to_rows", "pad_candidates"]
