"""PyTorch-backed implementation of the scalar ChoiceModel contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor

from ._validation import require_finite, require_floating, require_tensor
from .errors import InvalidTensorValue

ObservationT = TypeVar("ObservationT")
AgentStateT = TypeVar("AgentStateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True, slots=True)
class TorchChoiceModel(Generic[ObservationT, AgentStateT, ActionT]):
    """Compose application encoders and a tensor scorer as a choice model.

    ``logits`` is the Adaptive Choice inference boundary. It optionally disables
    gradient recording, validates the tensor output, detaches it, and returns
    host-side Python floats. ``tensor_logits`` performs the same encoding and
    validation without changing gradient mode or detaching the returned tensor.
    """

    context_encoder: Callable[[ObservationT, AgentStateT], Tensor]
    action_encoder: Callable[[Sequence[ActionT]], Tensor]
    scorer: Callable[[Tensor, Tensor], Tensor]
    use_inference_mode: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_inference_mode, bool):
            raise TypeError("use_inference_mode must be a bool")

    def tensor_logits(
        self,
        observation: ObservationT,
        agent: AgentStateT,
        actions: Sequence[ActionT],
    ) -> Tensor:
        """Return validated tensor logits while preserving their graph."""

        action_count = len(actions)
        if action_count == 0:
            raise InvalidTensorValue("actions", "at least one candidate is required")

        context = require_tensor(
            self.context_encoder(observation, agent),
            source="context_encoder output",
        )
        candidates = require_tensor(
            self.action_encoder(actions),
            source="action_encoder output",
        )
        if candidates.ndim == 0:
            raise InvalidTensorValue(
                "action_encoder output", "must have a candidate dimension"
            )
        if candidates.shape[0] != action_count:
            raise InvalidTensorValue(
                "action_encoder output",
                f"expected {action_count} candidate rows, got {candidates.shape[0]}",
            )

        logits = require_tensor(
            self.scorer(context, candidates),
            source="scorer output",
        )
        if logits.ndim != 1:
            raise InvalidTensorValue(
                "scorer output",
                f"expected shape ({action_count},), got {tuple(logits.shape)}",
            )
        if logits.shape[0] != action_count:
            raise InvalidTensorValue(
                "scorer output",
                f"expected {action_count} logits, got {logits.shape[0]}",
            )
        require_floating(logits, source="scorer output")
        require_finite(logits, source="scorer output")
        return logits

    def logits(
        self,
        observation: ObservationT,
        agent: AgentStateT,
        actions: Sequence[ActionT],
    ) -> tuple[float, ...]:
        """Return finite host-side logits for the scalar core runtime."""

        context: Any
        if self.use_inference_mode:
            context = torch.inference_mode()
        else:
            context = nullcontext()
        with context:
            logits = self.tensor_logits(observation, agent, actions)
        values = logits.detach().to(device="cpu", dtype=torch.float64).tolist()
        return tuple(float(value) for value in values)


__all__ = ["TorchChoiceModel"]
