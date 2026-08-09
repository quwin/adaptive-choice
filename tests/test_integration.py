"""End-to-end tests using only application-owned domain objects."""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass
from itertools import pairwise

from adaptive_choice import DecisionSystem, SoftmaxSampler


@dataclass(frozen=True)
class CounterWorld:
    turn: int
    total: int


@dataclass(frozen=True)
class CounterAgent:
    preference: float
    history: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterAction:
    name: str
    change: int


@dataclass(frozen=True)
class CounterObservation:
    turn: int
    visible_total: int


@dataclass(frozen=True)
class CounterOutcome:
    change: int
    new_total: int


class CounterEnvironment:
    """A deterministic world with a dynamically sized action set."""

    def __init__(self) -> None:
        self._turn = 0
        self._total = 0

    @property
    def state(self) -> CounterWorld:
        return CounterWorld(self._turn, self._total)

    def legal_actions(self, agent: CounterAgent) -> tuple[CounterAction, ...]:
        del agent
        forward = (
            CounterAction("small step", 1),
            CounterAction("large step", 2),
        )
        if self._turn % 2:
            return (CounterAction("step back", -1),) + forward
        return forward

    def step(self, action: CounterAction) -> CounterOutcome:
        self._total += action.change
        self._turn += 1
        return CounterOutcome(action.change, self._total)


class CounterObserver:
    def observe(
        self, state: CounterWorld, agent: CounterAgent
    ) -> CounterObservation:
        # This arbitrary observer hides negative totals from the agent.
        del agent
        return CounterObservation(state.turn, max(0, state.total))


class AdaptiveCounterModel:
    def logits(
        self,
        observation: CounterObservation,
        agent: CounterAgent,
        actions: tuple[CounterAction, ...],
    ) -> tuple[float, ...]:
        del observation
        return tuple(agent.preference * action.change for action in actions)


class CounterUpdater:
    def update(
        self,
        agent: CounterAgent,
        observation: CounterObservation,
        action: CounterAction,
        outcome: CounterOutcome,
    ) -> CounterAgent:
        del observation
        learned_preference = agent.preference + 0.15 * outcome.change
        return CounterAgent(learned_preference, agent.history + (action.name,))


def run_counter_trajectory(seed: int, steps: int = 8) -> tuple[object, ...]:
    environment = CounterEnvironment()
    system = DecisionSystem(
        observer=CounterObserver(),
        choice_model=AdaptiveCounterModel(),
        sampler=SoftmaxSampler(temperature=0.9),
        updater=CounterUpdater(),
    )
    rng = random.Random(seed)
    agent = CounterAgent(preference=0.0)
    results: list[object] = []

    for _ in range(steps):
        result = system.step(environment=environment, agent=agent, rng=rng)
        results.append(result)
        agent = result.agent

    return tuple(results)


class IntegrationTests(unittest.TestCase):
    def test_unrelated_generic_domain_types_work_without_inheritance(self) -> None:
        results = run_counter_trajectory(seed=11, steps=5)

        self.assertEqual(len(results), 5)
        self.assertTrue(
            all(isinstance(result.choice.action, CounterAction) for result in results)
        )
        self.assertTrue(
            all(
                isinstance(result.observation, CounterObservation)
                for result in results
            )
        )
        self.assertTrue(
            all(isinstance(result.outcome, CounterOutcome) for result in results)
        )
        self.assertEqual(
            results[-1].agent.history,
            tuple(result.choice.action.name for result in results),
        )

    def test_dynamic_action_counts_are_preserved_in_each_choice(self) -> None:
        results = run_counter_trajectory(seed=7, steps=6)

        self.assertEqual(
            [len(result.choice.logits) for result in results],
            [2, 3, 2, 3, 2, 3],
        )
        self.assertEqual(
            [len(result.choice.probabilities) for result in results],
            [2, 3, 2, 3, 2, 3],
        )

    def test_seed_reproduces_the_full_trajectory(self) -> None:
        first = run_counter_trajectory(seed=2026)
        second = run_counter_trajectory(seed=2026)

        self.assertEqual(first, second)

    def test_different_seed_can_change_the_adaptive_trajectory(self) -> None:
        # random.Random(1).random() begins below 0.5 while seed 2 begins above
        # 0.5, so the initial uniform choice is guaranteed to differ.
        first = run_counter_trajectory(seed=1)
        second = run_counter_trajectory(seed=2)

        self.assertNotEqual(first[0].choice.action, second[0].choice.action)
        self.assertNotEqual(first, second)

    def test_each_update_becomes_the_next_step_agent_state(self) -> None:
        results = run_counter_trajectory(seed=19, steps=6)

        for previous, following in pairwise(results):
            expected_preference = previous.agent.preference
            # The next choice probabilities must be based on the agent returned
            # by the preceding updater, not the initial agent.
            actions = (
                (-1, 1, 2) if following.observation.turn % 2 else (1, 2)
            )
            expected_logits = tuple(expected_preference * value for value in actions)
            self.assertEqual(following.choice.logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
