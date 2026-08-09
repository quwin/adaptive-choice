# Adaptive Choice

Adaptive Choice is a small, typed Python library for adaptive, stochastic,
preference-conditioned decisions in structured simulations.

Given an agent-visible observation, user-owned agent state, and the actions that
are legal *right now*, it scores each action, forms a probability distribution,
selects an action, executes it in the environment, and updates the agent. The
library coordinates that boundary without owning your world model, action
schema, learning algorithm, or simulation loop.

The 0.2 core remains dependency-free and framework-independent. It
provides protocols, result records, robust sampling, validation, and one-step
orchestration. PyTorch support ships as a separate optional distribution rather
than becoming a core dependency.

## Install

Install this checkout directly:

```bash
python -m pip install -e .
```

After the project is published to a package index, the intended release command
is `python -m pip install adaptive-choice`. Do not assume an unpublished package
name resolves to this source tree.

Install the optional PyTorch adapter separately:

```bash
python -m pip install adaptive-choice-torch
```

From this checkout, install both distributions in editable mode:

```bash
python -m pip install -e . -e ./packages/adaptive-choice-torch
```

## Quick start

Components use structural typing: implement the documented methods; no base
class or registration is required.

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Sequence

from adaptive_choice import DecisionSystem, SoftmaxSampler


@dataclass(frozen=True)
class Agent:
    preferred_flavor: str
    meals: int = 0


@dataclass(frozen=True)
class Action:
    restaurant: str
    flavor: str
    price: float


class RestaurantEnvironment:
    def __init__(self) -> None:
        self._state = {"open": ("Noodle Bar", "Cafe")}

    @property
    def state(self) -> dict[str, tuple[str, ...]]:
        return self._state

    def legal_actions(self, agent: Agent) -> Sequence[Action]:
        del agent
        return (
            Action("Noodle Bar", "spicy", 14.0),
            Action("Cafe", "mild", 10.0),
        )

    def step(self, action: Action) -> float:
        return 1.0 if action.flavor == "spicy" else 0.5


class MenuObserver:
    def observe(
        self,
        state: dict[str, tuple[str, ...]],
        agent: Agent,
    ) -> tuple[str, ...]:
        del agent
        return state["open"]


class PreferenceModel:
    def logits(
        self,
        observation: tuple[str, ...],
        agent: Agent,
        actions: Sequence[Action],
    ) -> Sequence[float]:
        del observation
        return tuple(
            (2.0 if action.flavor == agent.preferred_flavor else 0.0)
            - 0.05 * action.price
            for action in actions
        )


class MealUpdater:
    def update(
        self,
        agent: Agent,
        observation: tuple[str, ...],
        action: Action,
        outcome: float,
    ) -> Agent:
        del observation, action, outcome
        return replace(agent, meals=agent.meals + 1)


system = DecisionSystem(
    observer=MenuObserver(),
    choice_model=PreferenceModel(),
    sampler=SoftmaxSampler(temperature=0.8),
    updater=MealUpdater(),
)

result = system.step(
    environment=RestaurantEnvironment(),
    agent=Agent(preferred_flavor="spicy"),
    rng=Random(7),
)

print(result.choice.action)
print(result.choice.probabilities)
print(result.outcome)
print(result.agent)
```

`result.choice.logits` and `result.choice.probabilities` retain the distribution
in the same order as the legal actions. `result.agent` is the value returned by
the updater; Adaptive Choice does not mutate agent state itself.

For a deterministic policy, substitute `ArgmaxSampler()`. For lower-level
composition, call `simulate_step(...)` directly.

## Optional PyTorch models

`adaptive-choice-torch` adapts tensor encoders and scorers to the same scalar
`ChoiceModel` contract:

```python
from adaptive_choice_torch import DotProductScorer, TorchChoiceModel

torch_model = TorchChoiceModel(
    context_encoder=encode_observation_and_agent,
    action_encoder=encode_actions_in_order,
    scorer=DotProductScorer(),
)
```

The adapter also includes an MLP scorer and mask-preserving helpers for padded
candidate tensors. See the [PyTorch adapter guide](docs/torch-adapter.md).

## The decision boundary

The canonical step is:

```text
world state -> observation -> legal actions -> logits -> probabilities
            -> sampled action -> outcome -> updated agent
```

Responsibility is deliberately split:

| Component | Owns | Does not own |
| --- | --- | --- |
| `Environment` | authoritative state, legality, action execution | preferences |
| `Observer` | agent-visible information | world truth |
| `ChoiceModel` | one logit per ordered legal action | selection or learning |
| `Sampler` | probabilities and candidate index | action meaning |
| `AgentUpdater` | experience-to-agent transition | environment mechanics |

This separation supports imperfect information, changing candidate sets,
hand-written or learned utility models, and individualized state with shared
model parameters.

## Guarantees and constraints

- Legal actions may change identity and count at every step.
- `actions[i]`, `logits[i]`, and `probabilities[i]` always refer to the same
  candidate.
- Every stochastic operation receives an explicit RNG.
- Softmax is computed with numerical stabilization and rejects non-finite input.
- An empty action set and mismatched logit count fail before the environment is
  stepped.
- Public protocols and the `py.typed` marker support static type checking.
- The core has no required NumPy, PyTorch, JAX, or TensorFlow dependency.

## Documentation

- [Getting started](docs/getting-started.md)
- [Concepts and architecture](docs/concepts.md)
- [Complete API reference](docs/api.md)
- [Custom components](docs/custom-components.md)
- [Sampling and numerical behavior](docs/sampling.md)
- [Reproducibility](docs/reproducibility.md)
- [Errors and validation](docs/errors.md)
- [Testing components and trajectories](docs/testing.md)
- [Integration and typing](docs/integration.md)
- [Design boundaries](docs/design-boundaries.md)
- [FAQ](docs/faq.md)
- [Roadmap](docs/roadmap.md)

## Project status

Version 0.2 preserves the scalar core contract and adds the separately packaged
PyTorch adapter. Public APIs may evolve before 1.0, with changes documented in
[CHANGELOG.md](CHANGELOG.md). Training systems remain downstream of the runtime.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and design guidance. To
report a vulnerability, follow [SECURITY.md](SECURITY.md). Community
participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Adaptive Choice is distributed under the [MIT License](LICENSE).
