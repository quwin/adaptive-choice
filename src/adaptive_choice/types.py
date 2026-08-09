"""Immutable result values produced by adaptive-choice decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ObservationT = TypeVar("ObservationT")
AgentStateT = TypeVar("AgentStateT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True, slots=True)
class Choice(Generic[ActionT]):
    """A selected action together with its complete decision distribution.

    ``index`` addresses the legal action sequence supplied to the model.
    ``logits`` and ``probabilities`` are immutable snapshots in that same
    order.
    """

    action: ActionT
    index: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StepResult(Generic[ObservationT, ActionT, OutcomeT, AgentStateT]):
    """The observable record and updated agent state from one decision step."""

    observation: ObservationT
    choice: Choice[ActionT]
    outcome: OutcomeT
    agent: AgentStateT


__all__ = ["Choice", "StepResult"]
