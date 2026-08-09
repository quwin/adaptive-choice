# Testing

The component boundaries are designed for small dependency-free tests. Test each
transformation in isolation, then add a seeded trajectory test for their
composition.

The repository uses `unittest`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Environment tests

Given a known world state and action, assert both the returned outcome and next
world state. Test illegal domain actions directly against the environment; the
Adaptive Choice core does not define those errors.

```python
class EnvironmentTests(unittest.TestCase):
    def test_step_advances_round_and_returns_reward(self) -> None:
        game = Game()
        action = Action("safe", risk=0.1, reward=1)

        outcome = game.step(action)

        self.assertEqual(outcome, 1)
        self.assertEqual(game.state, 1)
```

## Observer tests

Construct world state containing both visible and hidden information, then assert
that the observation exposes only the permitted view. Include two agents when
visibility is agent-specific.

## Choice-model tests

Assert length, ordering, values, and finiteness independently of sampling:

```python
class ChoiceModelTests(unittest.TestCase):
    def test_scores_candidates_in_input_order(self) -> None:
        actions = (
            Action("safe", risk=0.1, reward=1),
            Action("bold", risk=0.8, reward=4),
        )

        logits = RiskModel().logits(0, Agent(boldness=0.5), actions)

        self.assertEqual(len(logits), len(actions))
        self.assertLess(logits[0], logits[1])
```

When adapting a framework model, also test empty candidates at the adapter level,
flat shape conversion, device-to-host conversion, and rejection of NaN or
infinite output by the runtime.

## Sampler tests

Separate probability tests from selection tests:

- equal logits yield uniform softmax probabilities;
- adding a common logit offset preserves probabilities;
- lower temperature sharpens and higher temperature flattens;
- extreme finite logits remain valid;
- argmax ties choose the first maximum;
- empty and non-finite logits raise `InvalidLogits`;
- seeded or fixed RNGs yield reproducible selection.

Use a controlled RNG to test categorical boundaries and call counts:

```python
class SequenceRandom:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return next(self._values)


class SamplerTests(unittest.TestCase):
    def test_uniform_draw_uses_explicit_rng(self) -> None:
        rng = SequenceRandom(0.9)

        index = SoftmaxSampler().sample((0.0, 0.0), rng)

        self.assertEqual(index, 1)
        self.assertEqual(rng.calls, 1)
```

For statistical frequency tests, fix a seed, use enough draws, and assert a
generous confidence interval. Do not require an exact stream unless stream
compatibility is specifically part of the contract.

## Updater tests

Given agent, observation, action, and outcome, assert the complete next agent.
If agent state is intended to be immutable, also assert that the original value
is unchanged.

## Integration tests

Use simple spy components to assert call order, arguments, and the complete
result. Fix every input and the RNG:

```python
class IntegrationTests(unittest.TestCase):
    def test_seeded_step_is_reproducible(self) -> None:
        first = make_system().step(make_game(), make_agent(), Random(73))
        second = make_system().step(make_game(), make_agent(), Random(73))

        self.assertEqual(first, second)
```

For a multi-step trajectory, instantiate two independent environments and RNGs
from the same fixtures, carry each returned agent forward, and compare the full
result sequences.

## Validation and short-circuit tests

For every invalid model or sampler output, use an environment spy whose `step`
method fails the test if called. Verify the specific exception class rather than
only the base class. Important cases include:

- no legal actions;
- too few or too many logits;
- NaN and infinite logits;
- wrong probability count or total;
- negative probability;
- non-integer, negative, and out-of-range choice index;
- RNG values below 0, equal to 1, or non-finite.

Also document the application's behavior if `AgentUpdater.update` raises after a
stateful environment has stepped.

## Behavioral evaluation

Accuracy alone rarely characterizes a stochastic adaptive simulation. External
evaluation can derive the following from collected results and application
trajectories:

- choice entropy and calibration;
- behavioral consistency and diversity;
- context, knowledge, and goal sensitivity;
- differences between agents using a shared model;
- outcome-dependent preference adaptation;
- complete trajectory reproducibility.

These metrics belong in the application or an evaluation package, not the core
runtime. `Choice.probabilities` provides the necessary distribution without
prescribing a metric library.

See [Reproducibility](reproducibility.md) for experiment metadata and stream
design.
