# Design boundaries

Adaptive Choice stays useful by remaining small. It is an interoperability layer
between a simulation, a utility model, a selection policy, and agent adaptation.
It is not the owner of those systems.

## Dependency direction

```text
application simulation
        |
        v
adaptive_choice core
        ^
        |
optional framework or domain adapters

training and evaluation consume results; the core does not import them
```

The core depends only on Python's standard library. Applications may place
NumPy, PyTorch, JAX, database, service, or distributed-systems code behind the
protocols without passing those dependencies to every Adaptive Choice user.
The official PyTorch integration follows this rule as the separate
`adaptive-choice-torch` distribution.

## Core responsibilities

The core is responsible for:

- structural contracts for one decision step;
- one-to-one candidate/logit/probability ordering;
- softmax and argmax selection;
- explicit choice RNG input;
- domain-independent validation;
- immutable decision results;
- small orchestration functions.

## Application responsibilities

The application is responsible for:

- world and agent schemas;
- legal-action generation and action execution;
- observation and information-access policy;
- utility semantics and model lifecycle;
- interpretation of outcomes and adaptation;
- persistence, concurrency, transactions, and retries;
- extra randomness in environments, observation, or updates;
- trajectory storage, evaluation, and training.

The orchestrator does not provide rollback. If `Environment.step` mutates world
state and a later updater call fails, the application must define its own
transaction or recovery strategy.

## Non-goals

The core does not provide a simulation or physics engine, universal memory or
goal representation, reinforcement-learning algorithms, natural-language or LLM
integration, model serving, experiment tracking, distributed execution, a
database, or a fixed action schema.

Natural-language values are allowed when they genuinely are the application's
structured domain values. The boundary only rejects a requirement to serialize
already structured data into language as an intermediate modeling format.

## Design rules

1. Prefer protocols and composition over inheritance.
2. Add an abstraction only after multiple implementations demonstrate a shared
   need.
3. Keep domain semantics outside the core.
4. Models return logits; samplers return candidate indices.
5. Environments own legality and world transitions.
6. Observers own visibility.
7. Updaters own agent-state transitions.
8. Make every source of randomness explicit.
9. Keep training separate from inference.
10. Preserve scalar semantics in optimized or batched adapters.
11. Scale against current candidates rather than a global action universe.
12. Prefer structured representations when structured data exists.
13. Keep distributed infrastructure out of the core.
14. Keep hand-written choice models first-class.
15. Preserve intermediate decision information.

## When to build an adapter

Build an application or framework adapter when several components share an
external convention—for example, conversion between tensor logits and Python
sequences, masked padded batches, or a Gymnasium environment wrapper. Keep the
adapter dependency optional, translate at the boundary, and test it against the
same scalar semantics described in [Concepts and architecture](concepts.md).

See [Integration and typing](integration.md) for concrete patterns and
[Roadmap](roadmap.md) for potential future work.
