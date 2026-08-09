"""Tests for the separately installed package surface."""

from __future__ import annotations

import unittest

import adaptive_choice_torch


class PublicApiTests(unittest.TestCase):
    def test_documented_names_are_exported(self) -> None:
        expected = {
            "AdaptiveChoiceTorchError",
            "DotProductScorer",
            "InvalidTensorValue",
            "MLPScorer",
            "PaddedCandidates",
            "TorchChoiceModel",
            "__version__",
            "masked_logits_to_rows",
            "pad_candidates",
        }

        self.assertEqual(set(adaptive_choice_torch.__all__), expected)
        self.assertEqual(adaptive_choice_torch.__version__, "0.2.0")
        for name in expected:
            self.assertTrue(hasattr(adaptive_choice_torch, name), name)


if __name__ == "__main__":
    unittest.main()
