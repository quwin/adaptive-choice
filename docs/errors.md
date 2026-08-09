# Errors and validation

Adaptive Choice validates only domain-independent decision invariants. All
library exceptions inherit from `AdaptiveChoiceError`, except invalid
`SoftmaxSampler` construction, which raises `ValueError` as a configuration
error.

```python
from adaptive_choice import AdaptiveChoiceError

try:
    result = system.step(environment, agent, rng)
except AdaptiveChoiceError as error:
    log_invalid_decision(error)
```

Avoid catching the base class unless the application has one sensible response
to every protocol violation. Specific exceptions provide better diagnostics.

## Exception reference

| Exception | Trigger | Environment stepped? |
| --- | --- | --- |
| `NoLegalActions` | `legal_actions` returns an empty sequence | No |
| `InvalidLogits` | logits are empty, non-real, non-finite, malformed, or not a usable sequence | No |
| `ActionCountMismatch` | number of logits or probabilities differs from number of legal actions | No |
| `InvalidProbabilityDistribution` | custom probabilities are non-real, non-finite, negative, or do not sum to 1 | No |
| `InvalidChoiceIndex` | a custom sampler returns a non-integer or out-of-range index | No |
| `InvalidRandomValue` | softmax receives an RNG value outside `[0, 1)` or a non-finite/non-real value | No |

The “Environment stepped?” column refers to `Environment.step`. The observer,
legal-action method, model, and sampler may already have run. Components should
avoid side effects in read/scoring operations so validation failures remain
cheap and safe.

## No legal actions

An empty legal-action sequence has no universal meaning, so Adaptive Choice does
not invent a no-op:

```python
from adaptive_choice import NoLegalActions

try:
    result = system.step(environment, agent, rng)
except NoLegalActions:
    mark_agent_idle(agent)
```

If “wait,” “stop,” or “do nothing” is legitimate, model it as an explicit legal
action and let the environment define its outcome.

## Invalid logits and count mismatches

Choice models must return one finite real value per legal action. Keep shape
errors distinct from numerical errors:

- `ActionCountMismatch` means the sequence lengths disagree.
- `InvalidLogits` means the logit values or container cannot be used safely.

NaN and either infinity are rejected rather than silently producing an invalid
softmax. Validate adapter output at its framework boundary if device arrays,
traced values, or nested shapes might escape.

## Invalid distributions

Built-in sampler methods generate valid distributions for valid input. This
exception primarily protects the runner from custom `Sampler.probabilities`
implementations. A distribution must contain finite real values, contain no
negative values, match the candidate count, and total approximately 1.

A custom sampler is responsible for producing values at ordinary floating-point
precision. Do not depend on validation tolerance to normalize a materially
incorrect distribution.

## Invalid indices

`Sampler.sample` returns a zero-based integer into the action snapshot. A custom
sampler that returns `-1`, `len(actions)`, a float, or another invalid value raises
`InvalidChoiceIndex` before `Environment.step` is called. Negative Python indexing
is deliberately not accepted.

## Invalid RNG values

`RandomGenerator.random()` must follow the conventional contract: one finite
real value greater than or equal to 0 and strictly less than 1. Returning 1 is
invalid even if a custom RNG treats that endpoint as possible. This strict
half-open interval keeps categorical boundaries unambiguous.

## Invalid temperature

`SoftmaxSampler(temperature=...)` raises `ValueError` when temperature is zero,
negative, NaN, infinite, or not a supported real number. Construction errors are
not decision-protocol errors and therefore do not inherit from
`AdaptiveChoiceError`.

## Domain exceptions and partial effects

Exceptions raised by user components propagate unchanged. The library does not
wrap an environment's domain error, an observer failure, or an updater exception.
This preserves the application's exception semantics.

The runner is not transactional. `Environment.step` completes before
`AgentUpdater.update`; if updating then fails, an environment that mutates in
place has already advanced. Applications requiring atomicity should provide a
transactional environment, execute on a copy, or add recovery around the whole
step.

See [Testing](testing.md) for verifying short-circuit behavior.
