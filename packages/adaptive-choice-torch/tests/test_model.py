"""Tests for the tensor-to-scalar choice-model boundary."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch
from adaptive_choice import (
    ArgmaxSampler,
    ChoiceModel,
    DecisionExperience,
    DecisionSystem,
)
from torch import Tensor

from adaptive_choice_torch import (
    DotProductScorer,
    InvalidTensorValue,
    TorchChoiceModel,
)


@dataclass(frozen=True)
class Action:
    x: float
    y: float


def encode_context(observation: float, agent: float) -> Tensor:
    return torch.tensor((observation, agent), dtype=torch.float32)


def encode_actions(actions: tuple[Action, ...]) -> Tensor:
    return torch.tensor([(action.x, action.y) for action in actions])


class StaticEnvironment:
    @property
    def state(self) -> float:
        return 2.0

    def legal_actions(self, agent: float) -> tuple[Action, ...]:
        del agent
        return (Action(1.0, 0.0), Action(0.0, 3.0))

    def step(self, action: Action) -> Action:
        return action


class IdentityObserver:
    def observe(self, state: float, agent: float) -> float:
        del agent
        return state


class IdentityUpdater:
    def update(
        self,
        agent: float,
        experience: DecisionExperience[float, Action, Action],
    ) -> float:
        del experience
        return agent


class NeverRandom:
    def random(self) -> float:
        raise AssertionError("argmax must not use randomness")


class TorchChoiceModelTests(unittest.TestCase):
    def make_model(self, **changes: object) -> TorchChoiceModel[float, float, Action]:
        arguments: dict[str, object] = {
            "context_encoder": encode_context,
            "action_encoder": encode_actions,
            "scorer": DotProductScorer(),
        }
        arguments.update(changes)
        return TorchChoiceModel(**arguments)  # type: ignore[arg-type]

    def test_logits_preserve_candidate_order_as_python_floats(self) -> None:
        actions = (Action(1.0, 0.0), Action(0.0, 3.0))

        logits = self.make_model().logits(2.0, 0.5, actions)

        self.assertEqual(logits, (2.0, 1.5))
        self.assertTrue(all(isinstance(value, float) for value in logits))

    def test_model_satisfies_core_protocol_and_runs_in_decision_system(self) -> None:
        model = self.make_model()
        system = DecisionSystem(
            observer=IdentityObserver(),
            choice_model=model,
            sampler=ArgmaxSampler(),
            updater=IdentityUpdater(),
        )

        result = system.step(StaticEnvironment(), 0.5, NeverRandom())

        self.assertIsInstance(model, ChoiceModel)
        self.assertEqual(result.choice.index, 0)
        self.assertEqual(result.choice.logits, (2.0, 1.5))

    def test_tensor_logits_preserves_gradients(self) -> None:
        context = torch.tensor((2.0, 0.5), requires_grad=True)
        candidates = torch.tensor(((1.0, 0.0), (0.0, 3.0)))
        model = TorchChoiceModel(
            context_encoder=lambda observation, agent: context,
            action_encoder=lambda actions: candidates,
            scorer=DotProductScorer(),
        )

        logits = model.tensor_logits(0.0, 0.0, (Action(0, 0), Action(0, 0)))
        logits.sum().backward()

        self.assertEqual(context.grad.tolist(), [1.0, 3.0])

    def test_scalar_logits_use_inference_mode_by_default(self) -> None:
        grad_modes: list[bool] = []

        def scorer(context: Tensor, candidates: Tensor) -> Tensor:
            grad_modes.append(torch.is_grad_enabled())
            return torch.mv(candidates, context)

        self.make_model(scorer=scorer).logits(
            2.0, 0.5, (Action(1.0, 0.0), Action(0.0, 1.0))
        )
        self.make_model(scorer=scorer, use_inference_mode=False).logits(
            2.0, 0.5, (Action(1.0, 0.0), Action(0.0, 1.0))
        )

        self.assertEqual(grad_modes, [False, True])

    def test_use_inference_mode_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "bool"):
            self.make_model(use_inference_mode=1)

    def test_empty_actions_are_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidTensorValue, "at least one"):
            self.make_model().logits(2.0, 0.5, ())

    def test_action_encoder_must_return_one_row_per_action(self) -> None:
        model = self.make_model(
            action_encoder=lambda actions: torch.zeros((1, 2))
        )

        with self.assertRaisesRegex(InvalidTensorValue, "expected 2 candidate rows"):
            model.logits(2.0, 0.5, (Action(1, 0), Action(0, 1)))

    def test_scorer_output_must_be_a_flat_float_per_action(self) -> None:
        invalid_outputs = (
            torch.zeros((2, 1)),
            torch.zeros(1),
            torch.tensor((1, 2)),
            torch.tensor((0.0, float("nan"))),
        )
        actions = (Action(1, 0), Action(0, 1))

        for output in invalid_outputs:
            with self.subTest(output=output):
                model = self.make_model(
                    scorer=lambda context, candidates, result=output: result
                )
                with self.assertRaises(InvalidTensorValue):
                    model.logits(2.0, 0.5, actions)

    def test_encoder_and_scorer_outputs_must_be_tensors(self) -> None:
        actions = (Action(1, 0),)
        invalid_models = (
            self.make_model(context_encoder=lambda observation, agent: [1.0]),
            self.make_model(action_encoder=lambda candidate_actions: [[1.0, 2.0]]),
            self.make_model(scorer=lambda context, candidates: [1.0]),
        )

        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaisesRegex(InvalidTensorValue, "torch.Tensor"):
                    model.logits(2.0, 0.5, actions)


if __name__ == "__main__":
    unittest.main()
