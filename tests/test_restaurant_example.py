"""Executable-example tests."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from examples.restaurant import main, run_simulation


class RestaurantExampleTests(unittest.TestCase):
    def test_example_is_reproducible_and_adaptive(self) -> None:
        first = run_simulation(seed=23, steps=8)
        second = run_simulation(seed=23, steps=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first.results), 8)
        self.assertEqual(len(first.final_agent.visits), 8)
        self.assertEqual(
            first.final_agent.visits,
            tuple(result.choice.action.restaurant.name for result in first.results),
        )
        self.assertTrue(first.final_agent.restaurant_affinities)
        self.assertTrue(
            any(
                abs(value) > 0.0
                for value in first.final_agent.restaurant_affinities.values()
            )
        )

    def test_example_uses_dynamic_legal_action_sets(self) -> None:
        run = run_simulation(seed=7, steps=5)

        self.assertEqual(
            [len(result.choice.logits) for result in run.results],
            [2, 2, 2, 3, 2],
        )
        for result in run.results:
            visible_names = {view.name for view in result.observation.options}
            self.assertIn(result.choice.action.restaurant.name, visible_names)

    def test_zero_steps_returns_the_initial_agent(self) -> None:
        run = run_simulation(seed=1, steps=0)

        self.assertEqual(run.results, ())
        self.assertEqual(run.final_agent.visits, ())
        self.assertEqual(run.final_agent.restaurant_affinities, {})

    def test_negative_step_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            run_simulation(steps=-1)

    def test_cli_entry_point_prints_decisions_and_learned_state(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--seed", "5", "--steps", "2"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("seed=5, steps=2", rendered)
        self.assertIn("day  1:", rendered)
        self.assertIn("Learned restaurant affinities:", rendered)


if __name__ == "__main__":
    unittest.main()
