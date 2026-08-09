"""Unit tests for the standard samplers."""

from __future__ import annotations

import math
import random
import unittest

from adaptive_choice import (
    ArgmaxSampler,
    InvalidLogits,
    InvalidRandomValue,
    SoftmaxSampler,
)
from tests._support import NeverRandom, SequenceRandom


class SoftmaxSamplerTests(unittest.TestCase):
    def assertDistribution(self, probabilities: tuple[float, ...]) -> None:
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=14)

    def test_default_temperature_matches_softmax_definition(self) -> None:
        sampler = SoftmaxSampler()

        probabilities = sampler.probabilities([1.0, 2.0, 3.0])
        denominator = sum(math.exp(value) for value in (1.0, 2.0, 3.0))
        expected = tuple(
            math.exp(value) / denominator for value in (1.0, 2.0, 3.0)
        )

        self.assertEqual(sampler.temperature, 1.0)
        self.assertIsInstance(probabilities, tuple)
        for actual, wanted in zip(probabilities, expected, strict=True):
            self.assertAlmostEqual(actual, wanted, places=14)
        self.assertDistribution(probabilities)

    def test_softmax_is_stable_for_extreme_positive_logits(self) -> None:
        sampler = SoftmaxSampler()

        extreme = sampler.probabilities([10_000.0, 9_999.0, 9_998.0])
        shifted = sampler.probabilities([0.0, -1.0, -2.0])

        for actual, wanted in zip(extreme, shifted, strict=True):
            self.assertAlmostEqual(actual, wanted, places=14)
        self.assertDistribution(extreme)

    def test_softmax_is_stable_for_extreme_negative_logits(self) -> None:
        sampler = SoftmaxSampler()

        probabilities = sampler.probabilities([-10_000.0, -10_001.0, -10_002.0])

        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])
        self.assertDistribution(probabilities)

    def test_equal_logits_produce_a_uniform_distribution(self) -> None:
        probabilities = SoftmaxSampler(temperature=0.125).probabilities(
            [42.0, 42.0, 42.0, 42.0]
        )

        self.assertEqual(probabilities, (0.25, 0.25, 0.25, 0.25))

    def test_temperature_controls_distribution_sharpness(self) -> None:
        logits = [-1.0, 0.0, 1.0]

        cold = SoftmaxSampler(temperature=0.25).probabilities(logits)
        ordinary = SoftmaxSampler(temperature=1.0).probabilities(logits)
        hot = SoftmaxSampler(temperature=8.0).probabilities(logits)

        self.assertGreater(cold[-1], ordinary[-1])
        self.assertGreater(ordinary[-1], hot[-1])
        self.assertLess(cold[0], ordinary[0])
        self.assertLess(ordinary[0], hot[0])
        self.assertLess(max(hot) - min(hot), max(ordinary) - min(ordinary))
        self.assertDistribution(cold)
        self.assertDistribution(hot)

    def test_invalid_temperatures_are_rejected_at_construction(self) -> None:
        invalid_temperatures = (
            0.0,
            -1.0,
            math.nan,
            math.inf,
            -math.inf,
            True,
            "1.0",
            None,
            object(),
        )
        for temperature in invalid_temperatures:
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    SoftmaxSampler(temperature=temperature)

    def test_invalid_logits_are_rejected(self) -> None:
        invalid_cases = (
            [],
            [1.0, math.nan],
            [1.0, math.inf],
            [1.0, -math.inf],
            [1.0, "high"],
            [1.0, [2.0]],
            [True],
            [object()],
        )
        sampler = SoftmaxSampler()

        for logits in invalid_cases:
            with self.subTest(logits=logits):
                with self.assertRaises(InvalidLogits):
                    sampler.probabilities(logits)

    def test_probabilities_are_an_immutable_snapshot(self) -> None:
        logits = [0.0, 1.0]

        probabilities = SoftmaxSampler().probabilities(logits)
        logits[:] = [100.0, -100.0]

        self.assertIsInstance(probabilities, tuple)
        self.assertAlmostEqual(probabilities[0], 1.0 / (1.0 + math.e))

    def test_sample_uses_one_explicit_rng_draw(self) -> None:
        cases = ((0.10, 0), (0.50, 1), (0.90, 2))
        sampler = SoftmaxSampler()

        for draw, expected_index in cases:
            with self.subTest(draw=draw):
                rng = SequenceRandom([draw])
                index = sampler.sample([0.0, 0.0, 0.0], rng)
                self.assertEqual(index, expected_index)
                self.assertEqual(rng.calls, 1)

    def test_single_candidate_handles_the_largest_valid_rng_draw(self) -> None:
        rng = SequenceRandom([math.nextafter(1.0, 0.0)])

        index = SoftmaxSampler().sample([123.0], rng)

        self.assertEqual(index, 0)
        self.assertEqual(rng.calls, 1)

    def test_sample_is_reproducible_with_a_seeded_rng(self) -> None:
        sampler = SoftmaxSampler(temperature=0.8)

        first_rng = random.Random(913)
        second_rng = random.Random(913)
        first = [sampler.sample([-1.0, 0.2, 0.7], first_rng) for _ in range(50)]
        second = [sampler.sample([-1.0, 0.2, 0.7], second_rng) for _ in range(50)]

        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_rng_draw_must_be_finite_and_in_the_half_open_unit_interval(self) -> None:
        invalid_draws = (
            -0.001,
            1.0,
            2.0,
            math.nan,
            math.inf,
            -math.inf,
            "0.5",
            True,
            None,
            object(),
        )
        sampler = SoftmaxSampler()

        for draw in invalid_draws:
            with self.subTest(draw=draw):
                with self.assertRaises(InvalidRandomValue):
                    sampler.sample([0.0, 0.0], SequenceRandom([draw]))

    def test_rng_without_random_method_is_rejected(self) -> None:
        with self.assertRaises(InvalidRandomValue):
            SoftmaxSampler().sample([0.0], object())

    def test_invalid_logits_short_circuit_before_rng_use(self) -> None:
        with self.assertRaises(InvalidLogits):
            SoftmaxSampler().sample([0.0, math.nan], NeverRandom())


class ArgmaxSamplerTests(unittest.TestCase):
    def test_probabilities_are_one_hot_at_the_maximum(self) -> None:
        sampler = ArgmaxSampler()

        probabilities = sampler.probabilities([-2.0, 5.0, 1.0])

        self.assertEqual(probabilities, (0.0, 1.0, 0.0))
        self.assertIsInstance(probabilities, tuple)

    def test_first_maximum_wins_a_tie(self) -> None:
        sampler = ArgmaxSampler()

        self.assertEqual(sampler.probabilities([4.0, 4.0, 2.0]), (1.0, 0.0, 0.0))
        self.assertEqual(sampler.sample([4.0, 4.0, 2.0], NeverRandom()), 0)

    def test_sample_is_deterministic_and_does_not_consult_rng(self) -> None:
        index = ArgmaxSampler().sample([-3.0, -1.0, -2.0], NeverRandom())

        self.assertEqual(index, 1)

    def test_invalid_logits_are_rejected_by_both_operations(self) -> None:
        invalid_cases = ([], [math.nan], [math.inf], [[1.0]], ["best"], [True])
        sampler = ArgmaxSampler()

        for logits in invalid_cases:
            with self.subTest(logits=logits, operation="probabilities"):
                with self.assertRaises(InvalidLogits):
                    sampler.probabilities(logits)
            with self.subTest(logits=logits, operation="sample"):
                with self.assertRaises(InvalidLogits):
                    sampler.sample(logits, NeverRandom())


if __name__ == "__main__":
    unittest.main()
