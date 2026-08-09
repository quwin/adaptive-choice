"""Tests for ordered masked candidate batching."""

from __future__ import annotations

import unittest

import torch

from adaptive_choice_torch import (
    DotProductScorer,
    InvalidTensorValue,
    PaddedCandidates,
    masked_logits_to_rows,
    pad_candidates,
)


class CandidateBatchingTests(unittest.TestCase):
    def test_pad_candidates_preserves_values_order_and_lengths(self) -> None:
        first = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        second = torch.tensor(((5.0, 6.0),))

        batch = pad_candidates((first, second), padding_value=-9.0)

        self.assertEqual(tuple(batch.values.shape), (2, 2, 2))
        self.assertEqual(batch.values[0].tolist(), first.tolist())
        self.assertEqual(batch.values[1].tolist(), [[5.0, 6.0], [-9.0, -9.0]])
        self.assertEqual(batch.mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(batch.lengths, (2, 1))

    def test_logits_to_rows_removes_padding_in_original_order(self) -> None:
        batch = pad_candidates((torch.zeros((2, 1)), torch.zeros((1, 1))))
        logits = torch.tensor(((0.1, 0.2), (0.3, float("nan"))))

        rows = batch.logits_to_rows(logits)

        self.assertEqual(tuple(map(len, rows)), (2, 1))
        self.assertAlmostEqual(rows[0][0], 0.1)
        self.assertAlmostEqual(rows[0][1], 0.2)
        self.assertAlmostEqual(rows[1][0], 0.3)

    def test_direct_mask_conversion_preserves_non_prefix_masks(self) -> None:
        rows = masked_logits_to_rows(
            torch.tensor(((1.0, 99.0, 2.0),)),
            torch.tensor(((True, False, True),)),
        )

        self.assertEqual(rows, ((1.0, 2.0),))

    def test_masked_batch_scoring_matches_scalar_dot_products(self) -> None:
        contexts = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        candidate_rows = (
            torch.tensor(((1.0, 0.0), (0.0, 1.0))),
            torch.tensor(((2.0, 1.0),)),
        )
        batch = pad_candidates(candidate_rows)
        batched_logits = torch.einsum("bkf,bf->bk", batch.values, contexts)

        rows = batch.logits_to_rows(batched_logits)
        scalar_rows = tuple(
            tuple(DotProductScorer()(context, candidates).tolist())
            for context, candidates in zip(contexts, candidate_rows, strict=True)
        )

        self.assertEqual(rows, scalar_rows)

    def test_direct_batch_value_construction_validates_lengths(self) -> None:
        with self.assertRaisesRegex(InvalidTensorValue, "does not match"):
            PaddedCandidates(
                values=torch.zeros((1, 2, 3)),
                mask=torch.tensor(((True, False),)),
                lengths=(2,),
            )

    def test_direct_batch_value_construction_rejects_empty_batches(self) -> None:
        with self.assertRaisesRegex(InvalidTensorValue, "non-empty"):
            PaddedCandidates(
                values=torch.zeros((0, 2, 3)),
                mask=torch.zeros((0, 2), dtype=torch.bool),
                lengths=(),
            )

    def test_padding_rejects_empty_or_incompatible_rows(self) -> None:
        invalid_batches = (
            (),
            (torch.zeros((0, 2)),),
            (torch.zeros((1, 2)), torch.zeros((1, 3))),
            (
                torch.zeros((1, 2), dtype=torch.float32),
                torch.zeros((1, 2), dtype=torch.float64),
            ),
        )

        for rows in invalid_batches:
            with self.subTest(rows=rows):
                with self.assertRaises(InvalidTensorValue):
                    pad_candidates(rows)

    def test_mask_must_be_boolean_matching_and_nonempty_per_row(self) -> None:
        logits = torch.zeros((2, 2))
        invalid_masks = (
            torch.ones((2, 2)),
            torch.ones((2, 1), dtype=torch.bool),
            torch.tensor(((True, False), (False, False))),
        )

        for mask in invalid_masks:
            with self.subTest(mask=mask):
                with self.assertRaises(InvalidTensorValue):
                    masked_logits_to_rows(logits, mask)

    def test_selected_logits_must_be_finite_floating_values(self) -> None:
        with self.assertRaises(InvalidTensorValue):
            masked_logits_to_rows(
                torch.tensor(((1, 2),)), torch.tensor(((True, True),))
            )
        with self.assertRaises(InvalidTensorValue):
            masked_logits_to_rows(
                torch.tensor(((1.0, float("inf")),)),
                torch.tensor(((True, True),)),
            )


if __name__ == "__main__":
    unittest.main()
