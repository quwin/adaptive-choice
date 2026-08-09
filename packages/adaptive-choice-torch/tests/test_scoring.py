"""Tests for reusable tensor scorers."""

from __future__ import annotations

import unittest

import torch

from adaptive_choice_torch import DotProductScorer, InvalidTensorValue, MLPScorer


class DotProductScorerTests(unittest.TestCase):
    def test_scores_each_candidate(self) -> None:
        logits = DotProductScorer()(
            torch.tensor((2.0, -1.0)),
            torch.tensor(((1.0, 0.0), (0.5, 3.0))),
        )

        self.assertEqual(logits.tolist(), [2.0, -2.0])

    def test_requires_matching_feature_counts(self) -> None:
        with self.assertRaisesRegex(InvalidTensorValue, "expected 2 features"):
            DotProductScorer()(torch.zeros(2), torch.zeros((3, 4)))

    def test_does_not_silently_move_or_cast_inputs(self) -> None:
        with self.assertRaisesRegex(InvalidTensorValue, "dtype"):
            DotProductScorer()(
                torch.zeros(2, dtype=torch.float32),
                torch.zeros((1, 2), dtype=torch.float64),
            )


class MLPScorerTests(unittest.TestCase):
    def test_returns_one_logit_per_candidate(self) -> None:
        scorer = MLPScorer(
            context_features=2,
            action_features=3,
            hidden_features=4,
        )

        logits = scorer(torch.zeros(2), torch.zeros((5, 3)))

        self.assertEqual(tuple(logits.shape), (5,))

    def test_constructor_requires_positive_feature_counts(self) -> None:
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    MLPScorer(invalid, 2, 3)  # type: ignore[arg-type]

    def test_forward_validates_configured_shapes(self) -> None:
        scorer = MLPScorer(2, 3, 4)

        with self.assertRaisesRegex(InvalidTensorValue, "expected 3 features"):
            scorer(torch.zeros(2), torch.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
