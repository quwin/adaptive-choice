"""Tests for the scalar decision loop and its validation boundaries."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from tests._support import NeverRandom, SequenceRandom

from adaptive_choice import (
    ActionCountMismatch,
    ArgmaxSampler,
    DecisionSystem,
    InvalidChoiceIndex,
    InvalidLogits,
    InvalidProbabilityDistribution,
    NoLegalActions,
    simulate_step,
)


class RecordingEnvironment:
    def __init__(self, trace: list[str], actions: list[str] | tuple[str, ...]) -> None:
        self.trace = trace
        self.actions = actions
        self.world = {"turn": 3, "secret": "world truth"}
        self.outcome = {"reward": 7}
        self.agent_seen = None
        self.action_seen = None

    @property
    def state(self) -> object:
        self.trace.append("environment.state")
        return self.world

    def legal_actions(self, agent: object) -> list[str] | tuple[str, ...]:
        self.trace.append("environment.legal_actions")
        self.agent_seen = agent
        return self.actions

    def step(self, action: str) -> object:
        self.trace.append("environment.step")
        self.action_seen = action
        return self.outcome


class RecordingObserver:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.arguments = None
        self.observation = {"visible_turn": 3}

    def observe(self, state: object, agent: object) -> object:
        self.trace.append("observer.observe")
        self.arguments = (state, agent)
        return self.observation


class RecordingModel:
    def __init__(self, trace: list[str], logits: object) -> None:
        self.trace = trace
        self.logits_value = logits
        self.arguments = None

    def logits(
        self, observation: object, agent: object, actions: tuple[str, ...]
    ) -> object:
        self.trace.append("choice_model.logits")
        self.arguments = (observation, agent, actions)
        return self.logits_value


class RecordingSampler:
    def __init__(
        self,
        trace: list[str],
        probabilities: object = (0.25, 0.75),
        index: object = 1,
    ) -> None:
        self.trace = trace
        self.probabilities_value = probabilities
        self.index_value = index
        self.probability_logits = None
        self.sample_arguments = None

    def probabilities(self, logits: tuple[float, ...]) -> object:
        self.trace.append("sampler.probabilities")
        self.probability_logits = logits
        return self.probabilities_value

    def sample(self, logits: tuple[float, ...], rng: object) -> object:
        self.trace.append("sampler.sample")
        self.sample_arguments = (logits, rng)
        return self.index_value


class RecordingUpdater:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.arguments = None
        self.updated_agent = {"name": "Ada", "experience": 1}

    def update(
        self,
        agent: object,
        observation: object,
        action: object,
        outcome: object,
    ) -> object:
        self.trace.append("updater.update")
        self.arguments = (agent, observation, action, outcome)
        return self.updated_agent


class RunnerTests(unittest.TestCase):
    def make_components(
        self,
        *,
        actions: list[str] | tuple[str, ...] = ("stay", "explore"),
        logits: object = (0.0, 1.0),
        probabilities: object = (0.25, 0.75),
        index: object = 1,
    ) -> tuple[
        list[str],
        RecordingEnvironment,
        RecordingObserver,
        RecordingModel,
        RecordingSampler,
        RecordingUpdater,
    ]:
        trace: list[str] = []
        return (
            trace,
            RecordingEnvironment(trace, actions),
            RecordingObserver(trace),
            RecordingModel(trace, logits),
            RecordingSampler(trace, probabilities, index),
            RecordingUpdater(trace),
        )

    def run_components(
        self,
        components: tuple[
            list[str],
            RecordingEnvironment,
            RecordingObserver,
            RecordingModel,
            RecordingSampler,
            RecordingUpdater,
        ],
        *,
        agent: object | None = None,
        rng: object | None = None,
    ) -> object:
        _, environment, observer, model, sampler, updater = components
        return simulate_step(
            environment=environment,
            agent={"name": "Ada"} if agent is None else agent,
            observer=observer,
            choice_model=model,
            sampler=sampler,
            updater=updater,
            rng=NeverRandom() if rng is None else rng,
        )

    def test_simulate_step_preserves_canonical_call_order_and_data_flow(self) -> None:
        components = self.make_components()
        trace, environment, observer, model, sampler, updater = components
        agent = {"name": "Ada"}
        rng = NeverRandom()

        result = self.run_components(components, agent=agent, rng=rng)

        self.assertEqual(
            trace,
            [
                "environment.state",
                "observer.observe",
                "environment.legal_actions",
                "choice_model.logits",
                "sampler.probabilities",
                "sampler.sample",
                "environment.step",
                "updater.update",
            ],
        )
        self.assertEqual(observer.arguments, (environment.world, agent))
        self.assertIs(environment.agent_seen, agent)
        self.assertEqual(
            model.arguments, (observer.observation, agent, ("stay", "explore"))
        )
        self.assertEqual(sampler.probability_logits, (0.0, 1.0))
        self.assertEqual(sampler.sample_arguments, ((0.0, 1.0), rng))
        self.assertEqual(environment.action_seen, "explore")
        self.assertEqual(
            updater.arguments,
            (agent, observer.observation, "explore", environment.outcome),
        )
        self.assertIs(result.observation, observer.observation)
        self.assertEqual(result.choice.action, "explore")
        self.assertEqual(result.choice.index, 1)
        self.assertEqual(result.choice.logits, (0.0, 1.0))
        self.assertEqual(result.choice.probabilities, (0.25, 0.75))
        self.assertIs(result.outcome, environment.outcome)
        self.assertIs(result.agent, updater.updated_agent)

    def test_actions_logits_and_probabilities_are_snapshotted(self) -> None:
        actions = ["stay", "explore"]
        logits = [0.0, 1.0]
        probabilities = [0.25, 0.75]
        components = self.make_components(
            actions=actions, logits=logits, probabilities=probabilities
        )

        result = self.run_components(components)
        _, _, _, model, sampler, _ = components
        actions[:] = ["changed"]
        logits[:] = [99.0]
        probabilities[:] = [1.0]

        self.assertIsInstance(model.arguments[2], tuple)
        self.assertIsInstance(sampler.probability_logits, tuple)
        self.assertEqual(result.choice.action, "explore")
        self.assertEqual(result.choice.logits, (0.0, 1.0))
        self.assertEqual(result.choice.probabilities, (0.25, 0.75))

    def test_choice_and_step_result_are_frozen_and_slotted(self) -> None:
        result = self.run_components(self.make_components())

        with self.assertRaises(FrozenInstanceError):
            result.outcome = object()
        with self.assertRaises(FrozenInstanceError):
            result.choice.index = 0
        # CPython versions differ on whether a frozen slotted dataclass reports
        # an unknown assignment as AttributeError, FrozenInstanceError, or the
        # TypeError produced by its generated __setattr__ implementation.
        with self.assertRaises((AttributeError, FrozenInstanceError, TypeError)):
            result.extra_field = "not allowed"
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertFalse(hasattr(result.choice, "__dict__"))

    def test_no_legal_actions_short_circuits_before_model_and_sampler(self) -> None:
        components = self.make_components(actions=[])
        trace = components[0]

        with self.assertRaises(NoLegalActions):
            self.run_components(components)

        self.assertEqual(
            trace,
            [
                "environment.state",
                "observer.observe",
                "environment.legal_actions",
            ],
        )

    def test_logit_count_must_match_action_count(self) -> None:
        components = self.make_components(logits=[1.0])

        with self.assertRaises(ActionCountMismatch):
            self.run_components(components)

        self.assertEqual(
            components[0],
            [
                "environment.state",
                "observer.observe",
                "environment.legal_actions",
                "choice_model.logits",
            ],
        )

    def test_invalid_model_logits_short_circuit_before_sampler(self) -> None:
        components = self.make_components(logits=[0.0, math.nan])

        with self.assertRaises(InvalidLogits):
            self.run_components(components)

        self.assertEqual(components[0][-1], "choice_model.logits")
        self.assertNotIn("sampler.probabilities", components[0])

    def test_probability_count_must_match_action_count(self) -> None:
        components = self.make_components(probabilities=[1.0])

        with self.assertRaises(ActionCountMismatch):
            self.run_components(components)

        self.assertEqual(components[0][-1], "sampler.probabilities")
        self.assertNotIn("sampler.sample", components[0])

    def test_invalid_custom_probability_distributions_are_rejected(self) -> None:
        invalid_distributions = (
            (-0.1, 1.1),
            (0.2, 0.2),
            (math.nan, 0.0),
            (math.inf, 0.0),
            ("likely", "unlikely"),
            ((0.5,), (0.5,)),
        )

        for probabilities in invalid_distributions:
            with self.subTest(probabilities=probabilities):
                components = self.make_components(probabilities=probabilities)
                with self.assertRaises(InvalidProbabilityDistribution):
                    self.run_components(components)
                self.assertEqual(components[0][-1], "sampler.probabilities")
                self.assertNotIn("sampler.sample", components[0])

    def test_invalid_custom_sampler_indexes_are_rejected(self) -> None:
        invalid_indexes = (-1, 2, 1.0, True, "1", None)

        for index in invalid_indexes:
            with self.subTest(index=index):
                components = self.make_components(index=index)
                with self.assertRaises(InvalidChoiceIndex):
                    self.run_components(components)
                self.assertEqual(components[0][-1], "sampler.sample")
                self.assertNotIn("environment.step", components[0])

    def test_decision_system_delegates_to_the_same_loop(self) -> None:
        components = self.make_components(index=0)
        trace, environment, observer, model, sampler, updater = components
        system = DecisionSystem(observer, model, sampler, updater)
        agent = {"name": "Ada"}

        result = system.step(environment, agent, NeverRandom())

        self.assertEqual(result.choice.action, "stay")
        self.assertEqual(trace[-2:], ["environment.step", "updater.update"])
        self.assertIs(system.observer, observer)
        self.assertIs(system.choice_model, model)
        self.assertIs(system.sampler, sampler)
        self.assertIs(system.updater, updater)

    def test_decision_system_is_a_frozen_slotted_configuration(self) -> None:
        components = self.make_components()
        _, _, observer, model, sampler, updater = components
        system = DecisionSystem(
            observer=observer,
            choice_model=model,
            sampler=sampler,
            updater=updater,
        )

        with self.assertRaises(FrozenInstanceError):
            system.sampler = ArgmaxSampler()
        self.assertFalse(hasattr(system, "__dict__"))

    def test_simulate_step_accepts_the_documented_positional_form(self) -> None:
        components = self.make_components(index=0)
        _, environment, observer, model, sampler, updater = components

        result = simulate_step(
            environment,
            {"name": "Ada"},
            observer,
            model,
            sampler,
            updater,
            NeverRandom(),
        )

        self.assertEqual(result.choice.action, "stay")


class FailureBoundaryTests(unittest.TestCase):
    def test_environment_failure_is_propagated_without_updating_agent(self) -> None:
        trace: list[str] = []

        class BrokenEnvironment(RecordingEnvironment):
            def step(self, action: str) -> object:
                self.trace.append("environment.step")
                raise LookupError("domain failure")

        environment = BrokenEnvironment(trace, ["act"])
        observer = RecordingObserver(trace)
        model = RecordingModel(trace, [0.0])
        sampler = RecordingSampler(trace, [1.0], 0)
        updater = RecordingUpdater(trace)

        with self.assertRaisesRegex(LookupError, "domain failure"):
            simulate_step(
                environment,
                object(),
                observer,
                model,
                sampler,
                updater,
                NeverRandom(),
            )

        self.assertEqual(trace[-1], "environment.step")
        self.assertNotIn("updater.update", trace)

    def test_sampler_failure_is_propagated_without_stepping_environment(self) -> None:
        trace: list[str] = []

        class BrokenSampler(RecordingSampler):
            def sample(self, logits: tuple[float, ...], rng: object) -> int:
                self.trace.append("sampler.sample")
                raise RuntimeError("sampler failure")

        environment = RecordingEnvironment(trace, ["act"])
        observer = RecordingObserver(trace)
        model = RecordingModel(trace, [0.0])
        sampler = BrokenSampler(trace, [1.0], 0)
        updater = RecordingUpdater(trace)

        with self.assertRaisesRegex(RuntimeError, "sampler failure"):
            simulate_step(
                environment,
                object(),
                observer,
                model,
                sampler,
                updater,
                NeverRandom(),
            )

        self.assertEqual(trace[-1], "sampler.sample")
        self.assertNotIn("environment.step", trace)

    def test_updater_failure_is_propagated_after_environment_step(self) -> None:
        trace: list[str] = []

        class BrokenUpdater(RecordingUpdater):
            def update(
                self,
                agent: object,
                observation: object,
                action: object,
                outcome: object,
            ) -> object:
                self.trace.append("updater.update")
                raise RuntimeError("update failure")

        environment = RecordingEnvironment(trace, ["act"])
        observer = RecordingObserver(trace)
        model = RecordingModel(trace, [0.0])
        sampler = RecordingSampler(trace, [1.0], 0)
        updater = BrokenUpdater(trace)

        with self.assertRaisesRegex(RuntimeError, "update failure"):
            simulate_step(
                environment,
                object(),
                observer,
                model,
                sampler,
                updater,
                NeverRandom(),
            )

        self.assertEqual(trace[-2:], ["environment.step", "updater.update"])

    def test_non_iterable_legal_action_result_fails_before_model(self) -> None:
        trace: list[str] = []

        class InvalidEnvironment(RecordingEnvironment):
            def legal_actions(self, agent: object) -> object:
                self.trace.append("environment.legal_actions")
                return None

        environment = InvalidEnvironment(trace, [])
        observer = RecordingObserver(trace)
        model = RecordingModel(trace, [0.0])
        sampler = RecordingSampler(trace, [1.0], 0)
        updater = RecordingUpdater(trace)

        with self.assertRaisesRegex(TypeError, "legal_actions"):
            simulate_step(
                environment,
                object(),
                observer,
                model,
                sampler,
                updater,
                NeverRandom(),
            )

        self.assertEqual(trace[-1], "environment.legal_actions")
        self.assertNotIn("choice_model.logits", trace)

    def test_type_error_raised_inside_legal_actions_is_not_rewritten(self) -> None:
        trace: list[str] = []
        domain_error = TypeError("agent is missing a domain capability")

        class BrokenEnvironment(RecordingEnvironment):
            def legal_actions(self, agent: object) -> tuple[str, ...]:
                self.trace.append("environment.legal_actions")
                raise domain_error

        environment = BrokenEnvironment(trace, [])
        observer = RecordingObserver(trace)
        model = RecordingModel(trace, [0.0])
        sampler = RecordingSampler(trace, [1.0], 0)
        updater = RecordingUpdater(trace)

        with self.assertRaises(TypeError) as raised:
            simulate_step(
                environment,
                object(),
                observer,
                model,
                sampler,
                updater,
                NeverRandom(),
            )

        self.assertIs(raised.exception, domain_error)
        self.assertEqual(trace[-1], "environment.legal_actions")
        self.assertNotIn("choice_model.logits", trace)


if __name__ == "__main__":
    unittest.main()
