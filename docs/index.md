# Adaptive Choice

Adaptive Choice is a minimal, typed Python kernel for adaptive and stochastic
choice in structured simulations. It coordinates one decision at a time:

```text
(observation, agent state, legal actions)
        -> logits -> probabilities -> action
        -> outcome -> updated agent state
```

The simulation retains ownership of its world, actions, outcomes, and agent
schema. Adaptive Choice supplies structural interfaces, robust samplers,
validation, result records, and orchestration. There are no mandatory
machine-learning or numerical-computing dependencies.

## Where to begin

- Follow [Getting started](getting-started.md) to implement and run one complete
  step.
- Read [Concepts and architecture](concepts.md) for component responsibilities
  and data flow.
- Use the [API reference](api.md) for every public symbol in version 0.1.0.
- Consult [Sampling and numerical behavior](sampling.md) before selecting a
  temperature or supplying custom logits.
- Use [Custom components](custom-components.md) to integrate a domain model.

## Why a decision kernel?

Simulations frequently mix world mechanics, perception, preferences, random
selection, and learning into one object. That makes it difficult to substitute a
model, test a transition, or determine which source of randomness caused an
outcome. Adaptive Choice gives each concern one boundary:

| Question | Owner |
| --- | --- |
| What is true, legal, and caused by an action? | `Environment` |
| What can this agent see? | `Observer` |
| How valuable is each current candidate? | `ChoiceModel` |
| How are utilities converted into behavior? | `Sampler` |
| How does experience change the agent? | `AgentUpdater` |

The orchestration layer performs no domain reasoning. As a result, a hand-written
scorer can be exchanged for a learned model, or stochastic softmax selection for
argmax selection, without changing environment mechanics.

## Version 0.1 scope

Version 0.1.0 includes the scalar runtime contract, explicit RNG handling,
softmax and argmax samplers, immutable results, and domain-independent
validation. It deliberately excludes training, framework adapters, batching,
and a prescribed model of memory or preference. See [Design boundaries](design-boundaries.md)
and the evidence-driven [Roadmap](roadmap.md).
