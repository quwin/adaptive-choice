# Frequently asked questions

## Is Adaptive Choice a simulation framework?

No. Your environment owns world state, legal actions, and action execution. The
library coordinates a single decision and agent update. Scheduling, time,
persistence, events, and multi-agent interaction remain in the simulation.

## Is this a reinforcement-learning library?

No. A policy trained with reinforcement learning can implement `ChoiceModel`,
and trajectories can feed external training code, but algorithms, replay
buffers, rewards, optimizers, and training loops are outside the core.

## Do components need to inherit base classes?

No. They satisfy typed protocols structurally. Implement the required methods
with compatible signatures. See [Custom components](custom-components.md).

## Does agent state need a particular schema?

No. It can be a frozen data class, mapping, domain entity, array, or other value.
Adaptive Choice does not inspect it. Prefer an explicit immutable value when
that works for the application's lifecycle.

## Why return logits instead of probabilities?

Logits separate learned or hand-written utility from behavioral policy. The same
model can use softmax at different temperatures or deterministic argmax without
being rewritten. Samplers also centralize numerical stabilization and validation.

## Can the legal action set change every step?

Yes. Dynamic action sets are a primary design requirement. A model receives the
current ordered candidates and returns exactly one logit for each; it does not
need a fixed global output head.

## What should happen when there are no legal actions?

The step raises `NoLegalActions`. If waiting or stopping is valid behavior, add
an explicit action with environment-defined semantics. The core cannot infer a
domain-appropriate fallback.

## Why is action order important?

Order establishes the mapping among actions, logits, probabilities, and the
sampled index. Argmax also uses it to break ties by choosing the first maximum.
Generate a deterministic order when replayability matters.

## Can I use NumPy, PyTorch, or JAX?

Yes, behind custom components. Convert final scalar logits to a flat host-side
sequence. The core does not import these frameworks or retain gradients,
dtypes, or devices. See [Integration and typing](integration.md).

## Can temperature depend on the agent or context?

Yes, but the built-in sampler stores one fixed temperature. Select or construct
the appropriate sampler at the application boundary, or implement a custom
sampler whose configuration is supplied explicitly. Avoid hiding mutable
context inside a shared sampler in concurrent simulations.

## Does `DecisionSystem` retain state?

It retains references to its observer, choice model, sampler, and updater. It
does not retain the environment, agent, or RNG passed to `step`. Components may
themselves be stateful, so thread safety depends on their implementations.

## Does the runner mutate my agent?

No library code mutates agent state. It passes the object to user components and
stores whatever the updater returns. A custom component can still mutate it, so
use immutable values or component discipline if this distinction matters.

## Is a step transactional?

No. The environment is stepped before the updater runs. If the updater fails, a
stateful environment may already have changed. Implement transaction or rollback
semantics in the application when required.

## How do I make a run reproducible?

Seed and retain an explicit RNG, deterministically order legal actions, and
record initial state plus component and model versions. Use separate RNG streams
for choice, environment, perception, and adaptation. Read
[Reproducibility](reproducibility.md) for the limits of seed-only replay.

## Why preserve the full distribution?

The selected action alone cannot reveal uncertainty, calibration, or near ties.
`Choice.logits` and `Choice.probabilities` enable debugging, evaluation,
trajectory collection, and downstream training without making those systems
core dependencies.

## Does the core support batching or multi-agent execution?

The core provides canonical scalar semantics. Applications can run multiple
agents with a shared `DecisionSystem` and distinct agent states. Optimized masked
batch adapters are potential future work and must preserve scalar semantics.

## Is there a built-in trainable preference model?

No. The initial release validates the interoperability boundary. A hand-written
model remains a permanent first-class use case, while optional trainable
adapters are evaluated on the [Roadmap](roadmap.md).
