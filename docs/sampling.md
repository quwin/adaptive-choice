# Sampling and numerical behavior

Choice models return relative utilities. Samplers expose the corresponding
probability distribution and select a zero-based candidate index.

```python
from random import Random

from adaptive_choice import SoftmaxSampler

sampler = SoftmaxSampler(temperature=0.8)
logits = (2.7, 2.5, 0.8, 1.3)

probabilities = sampler.probabilities(logits)
index = sampler.sample(logits, Random(7))
```

Both built-in samplers return probabilities as immutable `tuple[float, ...]`.
They reject empty, non-numeric, or non-finite logit sequences.

## Softmax sampler

For logit `z[i]` and positive temperature `temperature`, softmax is:

```text
p[i] = exp(z[i] / temperature) / sum(exp(z[j] / temperature) for j)
```

`SoftmaxSampler` computes the equivalent stabilized form:

```text
scaled[i] = z[i] / temperature
shifted[i] = scaled[i] - max(scaled)
p[i] = exp(shifted[i]) / sum(exp(shifted[j]) for j)
```

Subtracting the maximum does not change the distribution and prevents overflow
for large finite logits. Extremely unlikely candidates may underflow to exactly
zero, which is valid. The returned values are finite, non-negative, and sum to 1
within floating-point tolerance.

### Temperature

`SoftmaxSampler()` uses `temperature=1.0`.

- Below 1, differences are amplified and the distribution becomes sharper.
- Above 1, differences are compressed and the distribution becomes flatter.
- As a positive temperature approaches zero, behavior approaches argmax, but
  finite-precision effects become more pronounced.

Temperature must be a finite number greater than zero. Zero, negative values,
NaN, and infinities raise `ValueError` when the sampler is constructed. Use
`ArgmaxSampler` when deterministic maximum selection is the intended semantic;
do not emulate it with an extremely small temperature.

Equal logits produce an exactly uniform distribution in ordinary cases. Adding
the same constant to all logits leaves probabilities unchanged.

### Categorical selection

`sample` calculates probabilities, requests exactly one `rng.random()` value,
and maps that draw through the cumulative distribution. The value must be a
finite real number in the half-open interval `[0.0, 1.0)`. Any other value raises
`InvalidRandomValue`.

Pass a long-lived seeded `random.Random` instance or another object implementing
`RandomGenerator`. Never pass the `random` module's hidden global state if exact
stream ownership matters.

## Argmax sampler

`ArgmaxSampler` creates a one-hot distribution and selects the index of the
largest logit:

```python
from adaptive_choice import ArgmaxSampler

sampler = ArgmaxSampler()
assert sampler.probabilities((4.0, 4.0, 2.0)) == (1.0, 0.0, 0.0)
```

Ties resolve to the first maximal candidate. This policy makes ordering an
explicit tie-breaker. The RNG argument to `sample` is accepted for protocol
compatibility but never consulted.

Argmax is deterministic selection, not zero-temperature softmax. Its one-hot
probabilities document the actual policy used by `sample`.

## Candidate ordering

Candidate order is semantic:

```text
actions[i] <-> logits[i] <-> probabilities[i]
```

The runner snapshots model logits and sampler probabilities as tuples. Mutating
the original sequences afterward cannot rewrite a completed result. If action
objects themselves are mutable, however, the library does not deep-copy them.

## Custom sampler validation

When a custom sampler is used, the runner verifies that `probabilities`:

- has the same length as the legal action sequence;
- contains only finite real numbers;
- contains no negative values;
- has a total sufficiently close to 1.

It also verifies that `sample` returns a valid integer candidate index. Failures
raise the exceptions described in [Errors and validation](errors.md).

Validation establishes shape and numerical invariants; it cannot prove that a
custom sampler actually draws from its advertised distribution. Test that
behavior independently.

## Precision expectations

Logits and probabilities are converted to ordinary Python floats. The core does
not preserve decimal, arbitrary-precision, array dtype, device placement, or
autodifferentiation metadata. Convert at an adapter boundary and retain original
framework values separately when training or high-precision analysis needs
them.

Do not compare general softmax probabilities with exact equality. Use
`math.isclose`, appropriate tolerances, and invariant checks:

```python
import math

probabilities = SoftmaxSampler().probabilities((1.0, 2.0, 3.0))
assert all(math.isfinite(value) and value >= 0.0 for value in probabilities)
assert math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
```

Continue with [Reproducibility](reproducibility.md) for RNG stream design.
