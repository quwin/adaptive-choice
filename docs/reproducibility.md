# Reproducibility

Adaptive Choice never uses process-global randomness. Every stochastic sampler
receives an explicit object satisfying `RandomGenerator`:

```python
class RandomGenerator(Protocol):
    def random(self) -> float:
        ...
```

Python's `random.Random`, many small test doubles, and application-specific RNG
adapters satisfy this structural contract.

## Seed once and retain the stream

Create the RNG at the simulation boundary and reuse it:

```python
from random import Random

rng = Random(20260808)
agent = initial_agent

for _ in range(100):
    result = system.step(environment, agent, rng)
    agent = result.agent
```

Each softmax selection consumes one value from this stream. Constructing a new
RNG with the same seed inside the loop resets the stream and can repeat the same
relative draw at every step.

`ArgmaxSampler` consumes no values. Switching between argmax and softmax therefore
changes later stream position unless choice and other stochastic operations use
separate streams.

## Separate independent sources

A simulation may contain randomness in perception, world transitions, agent
adaptation, and choice. Give each concern its own RNG derived from a recorded
master seed:

```python
from random import Random

master = Random(314159)
choice_rng = Random(master.getrandbits(128))
environment_rng = Random(master.getrandbits(128))
observer_rng = Random(master.getrandbits(128))
updater_rng = Random(master.getrandbits(128))
```

Adaptive Choice receives only `choice_rng`; inject the other streams into their
own components. This prevents an added perception draw from silently changing
all later choices.

For parallel workers, derive and record a distinct seed per worker or trajectory.
Do not share one mutable RNG concurrently unless the RNG implementation and
ordering semantics explicitly support it.

## Record enough to replay

A reproducibility record should include:

- Adaptive Choice and Python versions;
- initial environment and agent state;
- component implementations and configuration;
- model parameters or content hash;
- master and derived seeds;
- action ordering at every decision if candidate generation can vary;
- external library versions and platform details when adapters affect logits.

`StepResult` preserves observation, logits, probabilities, choice, outcome, and
updated agent, which makes divergence localization straightforward. If every
recorded value matches through probabilities but the selected index differs,
inspect the RNG stream. If logits first diverge, inspect observation, agent, and
model state.

## Scope of the guarantee

With the same component behavior, ordered candidates, initial values, and RNG
stream, Adaptive Choice performs the same sequence of decisions. The library
cannot guarantee cross-platform bit-for-bit behavior for downstream models,
third-party numerical kernels, concurrent scheduling, unordered application
collections, or RNG implementations with different algorithms.

For long-lived experiments, replay against stored trajectories in addition to
storing a seed. A seed alone cannot compensate for application or dependency
changes that alter the number of draws or logits.

## Test RNGs

For precise unit tests, a tiny deterministic RNG is clearer than relying on a
seeded algorithm:

```python
class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


rng = FixedRandom(0.25)
index = sampler.sample((0.0, 0.0), rng)
assert rng.calls == 1
```

See [Testing](testing.md) for complete component and trajectory strategies.
