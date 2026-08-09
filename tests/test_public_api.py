"""Public-surface and structural-protocol tests."""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

import adaptive_choice
from adaptive_choice import (
    ActionCountMismatch,
    AdaptiveChoiceError,
    AgentUpdater,
    ArgmaxSampler,
    Choice,
    ChoiceModel,
    DecisionExperience,
    DecisionSystem,
    Environment,
    InvalidChoiceIndex,
    InvalidLogits,
    InvalidProbabilityDistribution,
    InvalidRandomValue,
    NoLegalActions,
    Observer,
    RandomGenerator,
    Sampler,
    SoftmaxSampler,
    StepResult,
    simulate_step,
)


class TinyEnvironment:
    @property
    def state(self) -> int:
        return 0

    def legal_actions(self, agent: object) -> tuple[str, ...]:
        return ("act",)

    def step(self, action: str) -> str:
        return "done"


class TinyObserver:
    def observe(self, state: int, agent: object) -> int:
        return state


class TinyModel:
    def logits(
        self, observation: int, agent: object, actions: tuple[str, ...]
    ) -> tuple[float, ...]:
        return (0.0,)


class TinyUpdater:
    def update(self, agent: object, experience: object) -> object:
        return agent


@dataclass(frozen=True)
class ExperienceAgent:
    experiences: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservedEvent:
    description: str


@dataclass(frozen=True)
class ReceivedInformation:
    message: str


PerceivedExperience = (
    DecisionExperience[int, str, str] | ObservedEvent | ReceivedInformation
)


class GeneralExperienceUpdater:
    def update(
        self,
        agent: ExperienceAgent,
        experience: PerceivedExperience,
    ) -> ExperienceAgent:
        if isinstance(experience, DecisionExperience):
            description = f"outcome:{experience.outcome}"
        elif isinstance(experience, ObservedEvent):
            description = f"event:{experience.description}"
        else:
            description = f"information:{experience.message}"
        return ExperienceAgent(agent.experiences + (description,))


class PublicApiTests(unittest.TestCase):
    def test_documented_public_names_are_exported(self) -> None:
        expected = {
            "ActionCountMismatch",
            "AdaptiveChoiceError",
            "AgentUpdater",
            "ArgmaxSampler",
            "Choice",
            "ChoiceModel",
            "DecisionExperience",
            "DecisionSystem",
            "Environment",
            "InvalidChoiceIndex",
            "InvalidLogits",
            "InvalidProbabilityDistribution",
            "InvalidRandomValue",
            "NoLegalActions",
            "Observer",
            "RandomGenerator",
            "Sampler",
            "SoftmaxSampler",
            "StepResult",
            "__version__",
            "simulate_step",
        }

        self.assertEqual(set(adaptive_choice.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(adaptive_choice, name), name)

    def test_version_identifies_the_v0_2_release(self) -> None:
        self.assertRegex(adaptive_choice.__version__, r"^0\.2\.\d+$")

    def test_domain_errors_share_a_single_public_base_class(self) -> None:
        error_types = (
            NoLegalActions,
            InvalidLogits,
            ActionCountMismatch,
            InvalidProbabilityDistribution,
            InvalidChoiceIndex,
            InvalidRandomValue,
        )

        for error_type in error_types:
            with self.subTest(error_type=error_type.__name__):
                self.assertTrue(issubclass(error_type, AdaptiveChoiceError))

    def test_diagnostic_errors_preserve_relevant_values(self) -> None:
        mismatch = ActionCountMismatch(3, 2, source="probabilities")
        bad_index = InvalidChoiceIndex(-1, 3)
        bad_random = InvalidRandomValue(1.0)

        self.assertEqual(
            (mismatch.expected, mismatch.actual, mismatch.source),
            (3, 2, "probabilities"),
        )
        self.assertIn("expected 3, got 2", str(mismatch))
        self.assertEqual((bad_index.index, bad_index.action_count), (-1, 3))
        self.assertIn("-1", str(bad_index))
        self.assertEqual(bad_random.value, 1.0)
        self.assertIn("[0, 1)", str(bad_random))

    def test_protocols_accept_ordinary_structural_implementations(self) -> None:
        environment = TinyEnvironment()
        observer = TinyObserver()
        model = TinyModel()
        sampler = SoftmaxSampler()
        updater = TinyUpdater()
        rng = random.Random(1)

        self.assertIsInstance(environment, Environment)
        self.assertIsInstance(observer, Observer)
        self.assertIsInstance(model, ChoiceModel)
        self.assertIsInstance(sampler, Sampler)
        self.assertIsInstance(updater, AgentUpdater)
        self.assertIsInstance(rng, RandomGenerator)

    def test_standard_samplers_satisfy_sampler_protocol(self) -> None:
        self.assertIsInstance(SoftmaxSampler(), Sampler)
        self.assertIsInstance(ArgmaxSampler(), Sampler)

    def test_agent_updater_accepts_any_application_perceived_experience(self) -> None:
        updater = GeneralExperienceUpdater()
        agent = ExperienceAgent()

        experiences: tuple[PerceivedExperience, ...] = (
            DecisionExperience(observation=1, action="act", outcome="done"),
            ObservedEvent("another agent arrived"),
            ReceivedInformation("the bridge is closed"),
        )
        for experience in experiences:
            agent = updater.update(agent, experience)

        self.assertEqual(
            agent.experiences,
            (
                "outcome:done",
                "event:another agent arrived",
                "information:the bridge is closed",
            ),
        )

    def test_public_values_are_constructible_for_external_storage(self) -> None:
        choice = Choice(
            action="act",
            index=0,
            logits=(0.0,),
            probabilities=(1.0,),
        )
        result = StepResult(
            observation={"visible": True},
            choice=choice,
            outcome="done",
            agent={"experience": 1},
        )

        self.assertIs(result.choice, choice)
        self.assertEqual(result.choice.action, "act")

    def test_imported_runner_symbols_are_the_public_implementations(self) -> None:
        self.assertIs(simulate_step, adaptive_choice.simulate_step)
        self.assertIs(DecisionSystem, adaptive_choice.DecisionSystem)


if __name__ == "__main__":
    unittest.main()
