# Roadmap

The roadmap is evidence-driven. Items beyond the current release describe
possible directions, not promised APIs or release dates. New abstractions should
be supported by at least two concrete implementations and must preserve the
boundaries in [Design boundaries](design-boundaries.md).

## 0.1: minimal runtime

Version 0.1.0 establishes:

- typed structural protocols;
- immutable `Choice` and `StepResult` records;
- numerically stable `SoftmaxSampler`;
- deterministic first-maximum `ArgmaxSampler`;
- `simulate_step` and compositional `DecisionSystem` entry points;
- explicit RNG handling and domain-independent validation;
- dependency-free unit tests and a small example simulation.

The release deliberately contains no trainable preference model.

## 0.2: optional PyTorch integration

Version 0.2 preserves the dependency-free scalar core and adds the separately
distributed `adaptive-choice-torch` package:

- `TorchChoiceModel` composes application context/action encoders with a tensor
  scorer;
- `DotProductScorer` and `MLPScorer` provide candidate-local scoring blocks;
- dynamic candidate padding retains an authoritative boolean mask and original
  row ordering;
- scalar conversion validates shape and finiteness before detaching logits;
- model mode, devices, dtypes, and training remain application-owned.

Batching covers tensor scoring, not environment execution. Scalar orchestration
remains the reference semantics.

## Candidate 0.3 themes

Trajectory and training interoperability may include:

- an application-neutral transition record or collection helper;
- imitation and preference-learning adapter examples;
- candidate-interaction patterns;
- adaptive preference-state examples;
- integration guidance for bandit and RL systems.

Training will remain downstream of the runtime. The core will not own optimizers,
replay buffers, loss functions, or experiment tracking.

## Longer-term extension points

Potential research and scaling directions include permutation-aware candidate
interaction, probabilistic preference state, hierarchical memory, shared models
across large agent populations, RL environment bridges, and rare semantic
fallbacks implemented by external systems.

None requires expanding the core agent schema or changing its ownership
model. In particular, individuality should normally live in agent state rather
than one model instance per agent.

## Evaluation criteria

A proposed roadmap item should answer:

1. Which observed user problems does it solve?
2. Which multiple implementations need the shared abstraction?
3. Can it remain optional and dependency-isolated?
4. How does it preserve action/logit/probability ordering?
5. How are randomness and reproducibility controlled?
6. Can its optimized behavior be tested against scalar semantics?
7. Does it keep domain meaning and training outside the core?

Track shipped work in the repository [changelog](../CHANGELOG.md); do not infer
availability from this page.
