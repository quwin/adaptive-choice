# Concepts and architecture

Adaptive Choice owns the decision boundary, not the world. Its generic types are
opaque: the library does not inspect the fields of a world state, observation,
agent, action, or outcome.

## Data flow

One call to `simulate_step` or `DecisionSystem.step` performs these operations in
order:

1. `observer.observe(environment.state, agent)` creates the agent-visible
   observation.
2. `environment.legal_actions(agent)` returns the ordered current candidates.
3. `choice_model.logits(observation, agent, actions)` returns one utility per
   candidate.
4. `sampler.probabilities(logits)` creates the distribution retained in the
   result.
5. `sampler.sample(logits, rng)` chooses a candidate index.
6. `environment.step(action)` applies the selected action and returns an outcome.
7. `updater.update(agent, observation, action, outcome)` returns the next agent
   state.
8. A `StepResult` captures the observation, complete `Choice`, outcome, and next
   agent.

Validation prevents the environment from being stepped when the action set,
logits, probabilities, random value, or selected index violates the contract.
See [Errors and validation](errors.md).

## Environment

The environment is authoritative for world state and mechanics. It exposes its
current `state`, determines which actions are legal for an agent, and applies one
selected action. It must not rank legal actions on the agent's behalf.

A deterministic mapping from `(state, action)` to `(next state, outcome)` is the
recommended baseline. A stochastic environment is valid, but it should own a
separate explicit RNG stream; the decision RNG passed to Adaptive Choice is not
forwarded to `Environment.step`.

## Observer

The observer translates world truth into what a particular agent can perceive.
This supports hidden state, stale beliefs, misinformation, uncertainty, and
agent-specific perception without forcing irrational behavior into the choice
model.

If observation is identity in a fully observable simulation, use a small
pass-through component. Keeping that component explicit still makes the
information boundary auditable.

## Agent state

Agent state is ordinary user-owned data, not a framework base class. It may
contain stable traits, learned preferences, temporary state, beliefs, memory,
goals, or none of these. The library passes it through unchanged until the
updater returns the next value.

Prefer immutable agent values where practical. Immutability makes the
`agent -> updated agent` transition visible and prevents a failed step from
partially modifying application state.

Memory and preference should not be conflated. "The restaurant was closed" is a
remembered fact; "I now prefer reliable restaurants" is an adaptation. A custom
updater may change both, but they remain domain concepts.

## Choice model

The model produces relative utilities—not actions and not probabilities. For
ordered actions `(a0, ..., aK-1)`, it must return exactly `K` finite, real logits
in the same order.

Only relative differences affect softmax. Adding a constant to every logit does
not change its distribution. A model can therefore be a short rule, a linear
scorer, a neural module hidden behind a dependency-free adapter, or any object
with the required `logits` method.

Dynamic candidate scoring is central. The model scores the legal set for this
step rather than emitting a fixed vector over a global action universe. Runtime
work can therefore scale with the current candidate count.

## Sampler

The sampler owns the translation from utility to behavior. It has two related
operations:

- `probabilities(logits)` returns the distribution for inspection and recording;
- `sample(logits, rng)` returns one candidate index.

`SoftmaxSampler` performs temperature-scaled stochastic choice.
`ArgmaxSampler` assigns probability 1 to the first maximal logit and ignores its
RNG argument. Because sampling is separate from scoring, policy behavior can be
changed without retraining or rewriting a model. Details are in
[Sampling and numerical behavior](sampling.md).

## Agent updater

The updater interprets experience. It receives the prior agent, the actual
observation used for the decision, the selected action, and the environment's
outcome, then returns the next agent state.

It may be a pure rule, Bayesian update, state machine, learned recurrent
transition, or identity operation. It does not modify world state; environment
mechanics have already completed.

## Results and observability

`Choice` preserves action, index, logits, and probabilities. `StepResult`
preserves that choice alongside the observation, outcome, and updated agent.
These records support debugging, entropy calculations, calibration, trajectory
collection, and external training without expanding the runtime abstraction.

The retained candidate list itself is not part of `Choice`; if downstream
analysis needs every action object, record the legal action sequence in the
application's trajectory schema.

## Scalar semantics first

The v0.1 API executes one agent and one dynamic candidate set at a time. Future
batch implementations must preserve the same ordering, validation, selection,
and update semantics. Padding and masks are adapter implementation details, not
different behavioral rules.

Next: implement each boundary in [Custom components](custom-components.md), or
see all signatures in the [API reference](api.md).
