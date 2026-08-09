"""Composition-oriented orchestration for one adaptive decision step."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ._validation import coerce_logits, coerce_probabilities
from .errors import InvalidChoiceIndex, NoLegalActions
from .protocols import (
    AgentUpdater,
    ChoiceModel,
    Environment,
    Observer,
    RandomGenerator,
    Sampler,
)
from .types import Choice, StepResult


StateT = TypeVar("StateT")
ObservationT = TypeVar("ObservationT")
AgentStateT = TypeVar("AgentStateT")
ActionT = TypeVar("ActionT")
OutcomeT = TypeVar("OutcomeT")


def _choice_index(value: Any, *, action_count: int) -> int:
    if isinstance(value, bool):
        raise InvalidChoiceIndex(value, action_count)
    try:
        index = operator.index(value)
    except TypeError as error:
        raise InvalidChoiceIndex(value, action_count) from error
    if not 0 <= index < action_count:
        raise InvalidChoiceIndex(value, action_count)
    return index


def simulate_step(
    environment: Environment[StateT, AgentStateT, ActionT, OutcomeT],
    agent: AgentStateT,
    observer: Observer[StateT, AgentStateT, ObservationT],
    choice_model: ChoiceModel[ObservationT, AgentStateT, ActionT],
    sampler: Sampler,
    updater: AgentUpdater[AgentStateT, ObservationT, ActionT, OutcomeT],
    rng: RandomGenerator,
) -> StepResult[ObservationT, ActionT, OutcomeT, AgentStateT]:
    """Run exactly one observe-score-sample-step-update decision cycle.

    Candidate actions, logits, and probabilities are materialized immediately
    so later mutation by user components cannot alter the recorded decision.
    Protocol violations are detected before the environment executes an action.

    Args:
        environment: Authoritative simulation state and transition mechanics.
        agent: Current application-owned agent state.
        observer: Maps world state and agent state to visible information.
        choice_model: Produces one logit for each legal action.
        sampler: Produces probabilities and a selected action index.
        updater: Produces the next agent state after the outcome.
        rng: Explicit random source used by stochastic samplers.

    Returns:
        An immutable record containing the observation, complete choice data,
        outcome, and updated agent state.

    Raises:
        NoLegalActions: If the environment returns no candidates.
        ActionCountMismatch: If logits or probabilities do not align with the
            action sequence.
        InvalidLogits: If a logit is not finite and real.
        InvalidProbabilityDistribution: If sampler probabilities are invalid.
        InvalidChoiceIndex: If the sampled index is not an in-range integer.
    """

    observation = observer.observe(environment.state, agent)

    legal_actions = environment.legal_actions(agent)
    try:
        actions = tuple(legal_actions)
    except TypeError as error:
        raise TypeError(
            "environment.legal_actions() must return an iterable"
        ) from error
    if not actions:
        raise NoLegalActions()

    logits = coerce_logits(
        choice_model.logits(observation, agent, actions),
        expected_count=len(actions),
    )
    probabilities = coerce_probabilities(
        sampler.probabilities(logits),
        expected_count=len(actions),
    )
    index = _choice_index(sampler.sample(logits, rng), action_count=len(actions))
    action = actions[index]
    outcome = environment.step(action)
    updated_agent = updater.update(agent, observation, action, outcome)

    choice = Choice(
        action=action,
        index=index,
        logits=logits,
        probabilities=probabilities,
    )
    return StepResult(
        observation=observation,
        choice=choice,
        outcome=outcome,
        agent=updated_agent,
    )


@dataclass(frozen=True, slots=True)
class DecisionSystem(
    Generic[StateT, ObservationT, AgentStateT, ActionT, OutcomeT]
):
    """A lightweight, reusable bundle of decision components."""

    observer: Observer[StateT, AgentStateT, ObservationT]
    choice_model: ChoiceModel[ObservationT, AgentStateT, ActionT]
    sampler: Sampler
    updater: AgentUpdater[AgentStateT, ObservationT, ActionT, OutcomeT]

    def step(
        self,
        environment: Environment[StateT, AgentStateT, ActionT, OutcomeT],
        agent: AgentStateT,
        rng: RandomGenerator,
    ) -> StepResult[ObservationT, ActionT, OutcomeT, AgentStateT]:
        """Run one decision step using this system's components."""

        return simulate_step(
            environment=environment,
            agent=agent,
            observer=self.observer,
            choice_model=self.choice_model,
            sampler=self.sampler,
            updater=self.updater,
            rng=rng,
        )


__all__ = ["DecisionSystem", "simulate_step"]
