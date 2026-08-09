# Custom components

Adaptive Choice uses `typing.Protocol`: an object is compatible when it provides
the required attributes and methods. There is no inheritance, decorator,
registry, or framework-owned agent class.

This page describes the behavioral contract of each component. The exact public
signatures are collected in the [API reference](api.md).

## Environment contract

An environment exposes current state, returns an ordered sequence of legal
actions, and applies a selected action:

```python
class MyEnvironment:
    @property
    def state(self) -> WorldState:
        ...

    def legal_actions(self, agent: MyAgent) -> Sequence[MyAction]:
        ...

    def step(self, action: MyAction) -> MyOutcome:
        ...
```

Requirements:

- `legal_actions` must describe legality at the instant of the decision.
- Return a stable, finite `Sequence`, not a one-shot iterator.
- Treat its order as meaningful for the duration of the step.
- `step` should reject domain-invalid actions itself; Adaptive Choice validates
  only the candidate index it selects.
- Do not use `step` to update the agent. Return an outcome for the updater.

Adaptive Choice snapshots legal actions to preserve ordering. It calls
`legal_actions` only once per step.

## Observer contract

An observer maps authoritative world state and the current agent to the
agent-visible observation:

```python
class MyObserver:
    def observe(
        self,
        state: WorldState,
        agent: MyAgent,
    ) -> MyObservation:
        ...
```

Keep information access here. Passing the entire world state through and asking
the model to pretend some fields are hidden defeats the separation and makes
information leaks difficult to test.

If perception is stochastic, store or inject a dedicated RNG in the observer.
Document that stream separately from the selection RNG supplied to
`DecisionSystem.step`.

## Choice model contract

A choice model scores the ordered candidate sequence:

```python
class MyChoiceModel:
    def logits(
        self,
        observation: MyObservation,
        agent: MyAgent,
        actions: Sequence[MyAction],
    ) -> Sequence[float]:
        ...
```

Return exactly one finite real number for each action. The association is by
position:

```text
actions[0] <-> logits[0]
actions[1] <-> logits[1]
...
```

The values are relative utilities; they do not need to be normalized or bounded.
Do not sample in this method. Keeping scoring deterministic for fixed inputs
isolates behavior randomness in the sampler.

### Adapting array and tensor models

The core accepts a sequence of values convertible to finite Python floats. Keep
framework code behind the component and return a flat host-side sequence:

```python
class TensorBackedModel:
    def __init__(self, model: object) -> None:
        self._model = model

    def logits(self, observation, agent, actions) -> Sequence[float]:
        tensor = score_candidates(self._model, observation, agent, actions)
        return tuple(float(value) for value in tensor.detach().cpu().reshape(-1))
```

This is an application adapter, not a core dependency. Ensure gradient-sensitive
training code does not expect the runtime's immutable float tuple to retain a
computation graph.

## Sampler contract

A sampler exposes both its distribution and selection behavior:

```python
class MySampler:
    def probabilities(self, logits: Sequence[float]) -> Sequence[float]:
        ...

    def sample(
        self,
        logits: Sequence[float],
        rng: RandomGenerator,
    ) -> int:
        ...
```

`probabilities` must return exactly one finite, non-negative value per logit and
the values must sum to 1 within floating-point tolerance. `sample` must return a
zero-based integer index into the same sequence. Use only the supplied RNG for
stochastic behavior.

The runner validates custom sampler output. Keep `probabilities` and `sample`
consistent: the retained distribution is what users will analyze, even though a
malicious or incorrect sampler could return an index using different logic.

### Example categorical sampler

This illustrative sampler delegates distribution construction to softmax but
implements its own draw. In normal code, use `SoftmaxSampler` directly.

```python
from adaptive_choice import RandomGenerator, SoftmaxSampler


class WrappedSoftmax:
    def __init__(self, temperature: float = 1.0) -> None:
        self._softmax = SoftmaxSampler(temperature)

    def probabilities(self, logits: Sequence[float]) -> Sequence[float]:
        return self._softmax.probabilities(logits)

    def sample(
        self,
        logits: Sequence[float],
        rng: RandomGenerator,
    ) -> int:
        probabilities = self.probabilities(logits)
        threshold = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                return index
        return len(probabilities) - 1
```

A production custom sampler must also validate its RNG values and handle edge
cases. Study [Sampling and numerical behavior](sampling.md) and use the built-in
samplers unless different behavior is necessary.

## Agent updater contract

The updater converts any application-defined perceived experience into the next
agent value:

```python
from dataclasses import dataclass

from adaptive_choice import DecisionExperience


@dataclass(frozen=True)
class ObservedEvent:
    description: str


@dataclass(frozen=True)
class ReceivedInformation:
    message: str


Experience = (
    DecisionExperience[MyObservation, MyAction, MyOutcome]
    | ObservedEvent
    | ReceivedInformation
)


class MyUpdater:
    def update(
        self,
        agent: MyAgent,
        experience: Experience,
    ) -> MyAgent:
        ...
```

`DecisionSystem.step` constructs `DecisionExperience` after the environment
returns an outcome. Applications can invoke the same updater independently when
the agent observes an event or receives information. The core does not prescribe
a universal experience base class, event bus, or update schedule.

Return the next value even if the update is an identity operation. Avoid mutating
the input agent before all update logic succeeds; otherwise a failed call can
leave the application with a partially updated object.

Two agents may interpret the same perceived experience differently. Objective
mechanics belong in the environment; subjective adaptation belongs here.

## Explicit protocol checks

The public protocols are runtime-checkable for simple diagnostics:

```python
from adaptive_choice import ChoiceModel, Environment

assert isinstance(MyEnvironment(), Environment)
assert isinstance(MyChoiceModel(), ChoiceModel)
```

Runtime protocol checks verify only that required attributes exist. They do not
validate signatures, return types, or semantics. Use a static type checker and
component tests for those guarantees. See [Integration and typing](integration.md)
and [Testing](testing.md).
