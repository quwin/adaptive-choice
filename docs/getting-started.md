# Getting started

This guide builds a complete decision step using ordinary Python objects. No
component inherits from Adaptive Choice; matching the typed method signatures is
enough.

## Requirements and installation

Install this checkout in editable mode:

```bash
python -m pip install -e .
```

After the package is published to a package index, the intended release command
will be:

```bash
python -m pip install adaptive-choice
```

Do not assume that command is available before publication. Adaptive Choice 0.1
has no mandatory third-party runtime dependencies and requires Python 3.10 or
newer.

## 1. Define domain values

The library does not prescribe agent, action, or outcome classes. Frozen data
classes make state transitions especially easy to inspect.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    boldness: float
    score: int = 0


@dataclass(frozen=True)
class Action:
    name: str
    risk: float
    reward: int
```

## 2. Implement the environment and observer

The environment owns world truth and action execution. The observer decides what
the agent can see.

```python
from typing import Sequence


class Game:
    def __init__(self) -> None:
        self._round = 0

    @property
    def state(self) -> int:
        return self._round

    def legal_actions(self, agent: Agent) -> Sequence[Action]:
        del agent
        return (
            Action("safe", risk=0.1, reward=1),
            Action("bold", risk=0.8, reward=4),
        )

    def step(self, action: Action) -> int:
        self._round += 1
        return action.reward


class RoundObserver:
    def observe(self, state: int, agent: Agent) -> int:
        del agent
        return state
```

Returning a `Sequence` of legal actions is important: its order establishes the
index mapping used throughout the decision.

## 3. Implement scoring and adaptation

The choice model returns one finite logit for every candidate. It does not select
an action. The updater receives the exact observation and action used by the
step, then returns the next agent value.

```python
from dataclasses import replace


class RiskModel:
    def logits(
        self,
        observation: int,
        agent: Agent,
        actions: Sequence[Action],
    ) -> Sequence[float]:
        del observation
        return tuple(
            action.reward * agent.boldness - action.risk
            for action in actions
        )


class ScoreUpdater:
    def update(
        self,
        agent: Agent,
        observation: int,
        action: Action,
        outcome: int,
    ) -> Agent:
        del observation, action
        return replace(agent, score=agent.score + outcome)
```

## 4. Compose and run

Use a seeded RNG instance. `random.Random` satisfies the `RandomGenerator`
protocol because it provides `random() -> float`.

```python
from random import Random

from adaptive_choice import DecisionSystem, SoftmaxSampler


game = Game()
agent = Agent(boldness=0.6)

system = DecisionSystem(
    observer=RoundObserver(),
    choice_model=RiskModel(),
    sampler=SoftmaxSampler(temperature=0.75),
    updater=ScoreUpdater(),
)

result = system.step(
    environment=game,
    agent=agent,
    rng=Random(42),
)
```

The returned object contains every important stage after candidate generation:

```python
assert result.observation == 0
assert len(result.choice.logits) == 2
assert len(result.choice.probabilities) == 2
assert sum(result.choice.probabilities) == 1.0

chosen_action = result.choice.action
chosen_index = result.choice.index
outcome = result.outcome
next_agent = result.agent
```

Probabilities can contain ordinary floating-point rounding, so production tests
should normally use `math.isclose` rather than exact equality for their sum.

## Run multiple steps

`DecisionSystem` bundles behavior components; it does not retain the environment,
agent, or RNG. Keep those values in the application loop and carry the returned
agent forward explicitly.

```python
rng = Random(42)
game = Game()
agent = Agent(boldness=0.6)

trajectory = []
for _ in range(5):
    result = system.step(game, agent, rng)
    trajectory.append(result)
    agent = result.agent
```

Reusing the same RNG advances one deliberate choice stream. Recreating
`Random(42)` inside the loop would reset that stream on every step.

## Use the functional API

`DecisionSystem.step` delegates to `simulate_step`. Use the function directly
when the components are already managed separately:

```python
from adaptive_choice import simulate_step


result = simulate_step(
    environment=game,
    agent=agent,
    observer=RoundObserver(),
    choice_model=RiskModel(),
    sampler=SoftmaxSampler(),
    updater=ScoreUpdater(),
    rng=rng,
)
```

Both entry points have the same semantics.

## Choose deterministic behavior

Swap in `ArgmaxSampler()` to select the first candidate with the highest logit:

```python
from adaptive_choice import ArgmaxSampler

deterministic_system = DecisionSystem(
    observer=RoundObserver(),
    choice_model=RiskModel(),
    sampler=ArgmaxSampler(),
    updater=ScoreUpdater(),
)
```

The RNG argument remains part of the common sampler interface but is not consumed
by argmax.

## Run the packaged example

The repository includes a longer restaurant simulation with configurable seed
and step count:

```bash
python examples/restaurant.py --seed 7 --steps 5
```

Continue with [Concepts and architecture](concepts.md), or use
[Custom components](custom-components.md) as an integration checklist.
