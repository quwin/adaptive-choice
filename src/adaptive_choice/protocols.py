"""Structural interfaces for adaptive decision components.

Applications implement these protocols with ordinary Python classes. No
inheritance from the library is required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

StateT_co = TypeVar("StateT_co", covariant=True)
StateT_contra = TypeVar("StateT_contra", contravariant=True)
ObservationT_co = TypeVar("ObservationT_co", covariant=True)
ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
AgentStateT = TypeVar("AgentStateT")
AgentStateT_contra = TypeVar("AgentStateT_contra", contravariant=True)
ActionT = TypeVar("ActionT")
ActionT_contra = TypeVar("ActionT_contra", contravariant=True)
OutcomeT_co = TypeVar("OutcomeT_co", covariant=True)
OutcomeT_contra = TypeVar("OutcomeT_contra", contravariant=True)


@runtime_checkable
class RandomGenerator(Protocol):
    """Minimal explicit random-number-generator interface.

    :class:`random.Random` and compatible generators satisfy this protocol.
    Implementations must return a finite real number in the half-open interval
    ``[0, 1)``.
    """

    def random(self) -> float:
        """Return the next uniform variate in the interval ``[0, 1)``."""
        ...


@runtime_checkable
class Environment(
    Protocol[StateT_co, AgentStateT_contra, ActionT, OutcomeT_co]
):
    """Authoritative state, action legality, and transition mechanics."""

    @property
    def state(self) -> StateT_co:
        """Return the current authoritative environment state."""
        ...

    def legal_actions(self, agent: AgentStateT_contra) -> Sequence[ActionT]:
        """Return the actions currently legal for ``agent`` in stable order."""
        ...

    def step(self, action: ActionT) -> OutcomeT_co:
        """Execute ``action``, update the environment, and return its outcome."""
        ...


@runtime_checkable
class Observer(
    Protocol[StateT_contra, AgentStateT_contra, ObservationT_co]
):
    """Construct agent-visible information from world and agent state."""

    def observe(
        self, state: StateT_contra, agent: AgentStateT_contra
    ) -> ObservationT_co:
        """Return what ``agent`` can observe about ``state``."""
        ...


@runtime_checkable
class ChoiceModel(
    Protocol[ObservationT_contra, AgentStateT_contra, ActionT_contra]
):
    """Score an ordered, dynamically sized collection of candidate actions."""

    def logits(
        self,
        observation: ObservationT_contra,
        agent: AgentStateT_contra,
        actions: Sequence[ActionT_contra],
    ) -> Sequence[float]:
        """Return one finite relative-utility score per candidate action."""
        ...


@runtime_checkable
class Sampler(Protocol):
    """Convert logits to probabilities and select a candidate index."""

    def probabilities(self, logits: Sequence[float]) -> Sequence[float]:
        """Return a probability corresponding to every input logit."""
        ...

    def sample(self, logits: Sequence[float], rng: RandomGenerator) -> int:
        """Return the index of one candidate, using ``rng`` if stochastic."""
        ...


@runtime_checkable
class AgentUpdater(
    Protocol[
        AgentStateT,
        ObservationT_contra,
        ActionT_contra,
        OutcomeT_contra,
    ]
):
    """Produce the next agent state from one decision experience."""

    def update(
        self,
        agent: AgentStateT,
        observation: ObservationT_contra,
        action: ActionT_contra,
        outcome: OutcomeT_contra,
    ) -> AgentStateT:
        """Return the agent state that follows this experience."""
        ...


__all__ = [
    "AgentUpdater",
    "ChoiceModel",
    "Environment",
    "Observer",
    "RandomGenerator",
    "Sampler",
]
