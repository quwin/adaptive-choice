# Integration and typing

Adaptive Choice is a typed boundary around application-owned types. It ships a
`py.typed` marker, so static analyzers inspect its inline annotations when the
package is installed.

## Annotate the complete composition

Protocol assignments make boundary mistakes visible without requiring component
inheritance. Using the domain types from [Getting started](getting-started.md):

```python
from random import Random

from adaptive_choice import (
    AgentUpdater,
    ChoiceModel,
    DecisionExperience,
    DecisionSystem,
    Environment,
    Observer,
    RandomGenerator,
    Sampler,
    SoftmaxSampler,
)

environment: Environment[int, Agent, Action, int] = Game()
observer: Observer[int, Agent, int] = RoundObserver()
model: ChoiceModel[int, Agent, Action] = RiskModel()
sampler: Sampler = SoftmaxSampler()
updater: AgentUpdater[Agent, DecisionExperience[int, Action, int]] = ScoreUpdater()
rng: RandomGenerator = Random(42)

system: DecisionSystem[int, int, Agent, Action, int] = DecisionSystem(
    observer=observer,
    choice_model=model,
    sampler=sampler,
    updater=updater,
)
```

The `DecisionSystem` generic order is world state, observation, agent state,
action, then outcome. `StepResult` uses observation, action, outcome, then agent:

```python
from adaptive_choice import StepResult

result: StepResult[int, Action, int, Agent]
result = system.step(environment, Agent(boldness=0.6), rng)
```

Run the project's strict type check with:

```bash
python -m mypy src
```

Downstream projects can check their own package and tests using their preferred
analyzer.

## Runtime protocol checks

All component protocols are `runtime_checkable`:

```python
from adaptive_choice import ChoiceModel, RandomGenerator

assert isinstance(RiskModel(), ChoiceModel)
assert isinstance(Random(42), RandomGenerator)
```

These checks inspect only whether named attributes exist. They do not verify
generic arguments, full signatures, numerical output, or semantics. Treat them
as diagnostics rather than validation; rely on static analysis and tests for the
actual contract.

## Adapting numerical frameworks

Keep third-party dependencies in downstream components. Convert the final
candidate scores to a flat sequence of finite host-side real numbers:

```python
class FrameworkChoiceModel:
    def __init__(self, model: object) -> None:
        self.model = model

    def logits(self, observation, agent, actions) -> tuple[float, ...]:
        output = run_model(self.model, observation, agent, actions)
        return tuple(float(item) for item in to_host_vector(output))
```

The core then snapshots those values as Python floats. Device, dtype, gradient,
and model-mode ownership remain in the adapter. Runtime result tuples do not
retain a computation graph. If training requires original tensors, record them
in a training-specific trajectory outside `StepResult`.

For PyTorch, the separately installed `adaptive-choice-torch` package provides
this boundary directly. See the [PyTorch adapter guide](torch-adapter.md).

For padded batches with shapes such as `[batch, max_candidates, features]`, mask
padded logits inside the adapter and expose each real candidate sequence with the
same scalar ordering. Version 0.2 does not define a core batch-execution API; the
optional PyTorch adapter provides mask-preserving tensor batching helpers.

## Adapting RNGs

An external generator need only expose a scalar `random()` result. An explicit
adapter can normalize a library scalar to Python `float`:

```python
class GeneratorAdapter:
    def __init__(self, generator: object) -> None:
        self.generator = generator

    def random(self) -> float:
        return float(draw_scalar_uniform(self.generator))
```

The draw must be finite and in `[0, 1)`. Keep this adapter instance with the
simulation rather than constructing it per call.

## Multi-agent systems

One `DecisionSystem` can be shared across agents when its components support
sharing:

```python
for agent_id in scheduled_agent_ids:
    result = system.step(environment, agents[agent_id], rngs[agent_id])
    agents[agent_id] = result.agent
```

Individuality normally belongs in each agent value while a choice model shares
parameters. Give each agent or trajectory a deliberate RNG stream if scheduling
order must not change its choice sequence.

`DecisionSystem` being frozen does not make referenced components thread-safe.
Models with caches, mutable framework state, or internal RNGs still require
application-level synchronization or per-worker instances.

## Persistence and serialization

`Choice` and `StepResult` are ordinary data classes, but serializability depends
on domain values. The library does not choose JSON, pickle, database, or schema
versions. Persist a domain-specific transition record containing stable action
identifiers and enough candidate information for replay.

Avoid treating arbitrary pickle input as trusted merely because the top-level
object is a `StepResult`.

## Async and service boundaries

The core runtime is synchronous. Async applications can call it directly when
components are CPU-local and non-blocking, or place blocking model/environment
operations behind their own scheduling boundary. Do not split a stateful step
across concurrent tasks without defining environment consistency and rollback.

Across a service boundary, serialize domain values in the application, validate
remote model output, and implement timeout/retry semantics outside the core. A
retry after `Environment.step` may duplicate an action unless the application
uses idempotency or transactions.

## Compatibility boundaries

The core promises ordered scalar semantics, not a particular framework array
type. Adapters should test:

- candidate and logit order;
- exact length after masking or filtering;
- finite scalar conversion;
- deterministic model mode where expected;
- RNG ownership;
- parity with one scalar `simulate_step`.

Review [Design boundaries](design-boundaries.md) before promoting adapter-specific
concepts into shared code.
