# API reference

This page documents every public symbol in Adaptive Choice 0.1.0. All are
importable from the package root:

```python
from adaptive_choice import DecisionSystem, SoftmaxSampler, simulate_step
```

Module paths are included for readers maintaining precise imports, but the root
imports are the supported convenience API.

## Protocols

Protocols live in `adaptive_choice.protocols`. They are structural and decorated
with `typing.runtime_checkable`. Runtime `isinstance` checks test attribute
presence only; static checking verifies compatible signatures.

### `RandomGenerator`

```python
class RandomGenerator(Protocol):
    def random(self) -> float: ...
```

The returned value must be a finite real number in `[0.0, 1.0)`. A
`random.Random` instance satisfies this protocol. Built-in stochastic samplers
raise `InvalidRandomValue` for a boolean, non-real, non-finite, or out-of-range
draw.

### `Environment[StateT, AgentStateT, ActionT, OutcomeT]`

```python
class Environment(Protocol[StateT, AgentStateT, ActionT, OutcomeT]):
    @property
    def state(self) -> StateT: ...

    def legal_actions(
        self,
        agent: AgentStateT,
    ) -> Sequence[ActionT]: ...

    def step(self, action: ActionT) -> OutcomeT: ...
```

`state` and outcome are covariant; the agent input is contravariant. `ActionT`
is invariant because actions are both returned and consumed. `legal_actions`
defines candidate order for the step.

### `Observer[StateT, AgentStateT, ObservationT]`

```python
class Observer(Protocol[StateT, AgentStateT, ObservationT]):
    def observe(
        self,
        state: StateT,
        agent: AgentStateT,
    ) -> ObservationT: ...
```

World and agent inputs are contravariant; observation output is covariant.

### `ChoiceModel[ObservationT, AgentStateT, ActionT]`

```python
class ChoiceModel(Protocol[ObservationT, AgentStateT, ActionT]):
    def logits(
        self,
        observation: ObservationT,
        agent: AgentStateT,
        actions: Sequence[ActionT],
    ) -> Sequence[float]: ...
```

All generic parameters are input types and therefore contravariant. Return one
finite real logit per action, preserving order. The runner freezes values as
Python floats before sampling.

### `Sampler`

```python
class Sampler(Protocol):
    def probabilities(
        self,
        logits: Sequence[float],
    ) -> Sequence[float]: ...

    def sample(
        self,
        logits: Sequence[float],
        rng: RandomGenerator,
    ) -> int: ...
```

`probabilities` must return one finite value in `[0, 1]` per logit and the total
must be approximately 1. `sample` returns a zero-based index. The runner validates
both operations for custom samplers before stepping the environment.

### `AgentUpdater[AgentStateT, ObservationT, ActionT, OutcomeT]`

```python
class AgentUpdater(
    Protocol[AgentStateT, ObservationT, ActionT, OutcomeT]
):
    def update(
        self,
        agent: AgentStateT,
        observation: ObservationT,
        action: ActionT,
        outcome: OutcomeT,
    ) -> AgentStateT: ...
```

The agent type is invariant because it is both input and output. Observation,
action, and outcome are contravariant inputs.

## Result types

Result records live in `adaptive_choice.types`. Both are frozen, slotted data
classes: attributes cannot be reassigned and instances have value-based equality.
Contained domain objects are not deep-copied or made immutable.

### `Choice[ActionT]`

```python
@dataclass(frozen=True, slots=True)
class Choice(Generic[ActionT]):
    action: ActionT
    index: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
```

- `action`: selected domain action.
- `index`: its zero-based position in the legal-action snapshot.
- `logits`: validated immutable model-output snapshot.
- `probabilities`: validated immutable sampler-distribution snapshot.

### `StepResult[ObservationT, ActionT, OutcomeT, AgentStateT]`

```python
@dataclass(frozen=True, slots=True)
class StepResult(Generic[ObservationT, ActionT, OutcomeT, AgentStateT]):
    observation: ObservationT
    choice: Choice[ActionT]
    outcome: OutcomeT
    agent: AgentStateT
```

`agent` is the value returned by `AgentUpdater.update`, not necessarily the same
object passed into the step.

## Built-in samplers

Samplers live in `adaptive_choice.sampling`.

### `SoftmaxSampler`

```python
@dataclass(frozen=True, slots=True)
class SoftmaxSampler:
    temperature: float = 1.0

    def probabilities(
        self,
        logits: Sequence[float],
    ) -> tuple[float, ...]: ...

    def sample(
        self,
        logits: Sequence[float],
        rng: RandomGenerator,
    ) -> int: ...
```

Constructs a temperature-scaled, numerically stable softmax distribution and
performs one categorical draw. Temperature is normalized to a Python float and
must be a non-boolean finite real number greater than zero; otherwise construction
raises `ValueError`.

`probabilities` and `sample` raise `InvalidLogits` for invalid or empty input.
`sample` additionally raises `InvalidRandomValue` for a bad RNG or draw. It
consults the RNG exactly once after validating logits.

### `ArgmaxSampler`

```python
@dataclass(frozen=True, slots=True)
class ArgmaxSampler:
    def probabilities(
        self,
        logits: Sequence[float],
    ) -> tuple[float, ...]: ...

    def sample(
        self,
        logits: Sequence[float],
        rng: RandomGenerator,
    ) -> int: ...
```

Returns a one-hot distribution at, and selects, the first maximum. The RNG is
not consulted. Both methods raise `InvalidLogits` for invalid or empty input.

See [Sampling and numerical behavior](sampling.md) for policy details.

## Orchestration

Orchestration lives in `adaptive_choice.runner`.

### `simulate_step`

```python
def simulate_step(
    environment: Environment[StateT, AgentStateT, ActionT, OutcomeT],
    agent: AgentStateT,
    observer: Observer[StateT, AgentStateT, ObservationT],
    choice_model: ChoiceModel[ObservationT, AgentStateT, ActionT],
    sampler: Sampler,
    updater: AgentUpdater[AgentStateT, ObservationT, ActionT, OutcomeT],
    rng: RandomGenerator,
) -> StepResult[ObservationT, ActionT, OutcomeT, AgentStateT]: ...
```

Runs one observe, candidate generation, scoring, distribution, sampling,
environment step, and agent update cycle. Arguments may be positional or keyword.
Candidate actions, logits, and probabilities are materialized before action
execution.

Raises:

- `NoLegalActions` for an empty candidate sequence;
- `ActionCountMismatch` for logit or probability length mismatch;
- `InvalidLogits` for malformed model output;
- `InvalidProbabilityDistribution` for malformed sampler output;
- `InvalidChoiceIndex` for a non-integral or out-of-range sampled index;
- sampler-specific errors such as `InvalidRandomValue`.

If `legal_actions` is not iterable, a `TypeError` with a boundary-specific
message is raised. Exceptions from other component methods propagate unchanged.

### `DecisionSystem[StateT, ObservationT, AgentStateT, ActionT, OutcomeT]`

```python
@dataclass(frozen=True, slots=True)
class DecisionSystem(
    Generic[StateT, ObservationT, AgentStateT, ActionT, OutcomeT]
):
    observer: Observer[StateT, AgentStateT, ObservationT]
    choice_model: ChoiceModel[ObservationT, AgentStateT, ActionT]
    sampler: Sampler
    updater: AgentUpdater[AgentStateT, ObservationT, ActionT, OutcomeT]

    def step(
        self,
        environment: Environment[StateT, AgentStateT, ActionT, OutcomeT],
        agent: AgentStateT,
        rng: RandomGenerator,
    ) -> StepResult[ObservationT, ActionT, OutcomeT, AgentStateT]: ...
```

A reusable immutable component bundle. `step` delegates to `simulate_step` and
has identical behavior. The components referenced by the bundle may themselves
be mutable or stateful.

## Exceptions

Exceptions live in `adaptive_choice.errors`.

### `AdaptiveChoiceError`

Base class for decision-protocol violations.

### `NoLegalActions`

Raised when the environment returns no candidates.

### `InvalidLogits`

Raised for an empty logit iterable or any boolean, non-real, or non-finite logit.

### `ActionCountMismatch`

```python
class ActionCountMismatch(AdaptiveChoiceError):
    expected: int
    actual: int
    source: str
```

Raised when logits or probabilities do not match the legal action count.
`source` is `"logits"` or `"probabilities"`.

### `InvalidProbabilityDistribution`

Raised for an empty probability iterable, a boolean/non-real/non-finite value, a
value outside `[0, 1]`, or a total not close to 1. The total uses `math.fsum`; the
acceptance check uses `math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12)`.

### `InvalidChoiceIndex`

```python
class InvalidChoiceIndex(AdaptiveChoiceError):
    index: object
    action_count: int
```

Raised when a custom sampler returns a boolean, non-indexable value, negative
index, or index outside the legal-action snapshot. Objects implementing
`__index__` are accepted and normalized to `int`.

### `InvalidRandomValue`

```python
class InvalidRandomValue(AdaptiveChoiceError):
    value: object
```

Raised when an RNG lacks `random`, or the draw is boolean, non-real, non-finite,
negative, or greater than or equal to 1.

More recovery guidance is in [Errors and validation](errors.md).

## Version

### `__version__`

```python
from adaptive_choice import __version__

assert __version__ == "0.1.0"
```

A PEP 440-compatible package version string. For installed distribution metadata,
applications may alternatively use
`importlib.metadata.version("adaptive-choice")`.
