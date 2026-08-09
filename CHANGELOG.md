# Changelog

All notable changes to Adaptive Choice are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Runtime-checkable protocols for environments, observers, choice models,
  samplers, agent updaters, and random generators.
- Generic `Choice` and `StepResult` records that preserve the full decision
  trace.
- Numerically stable `SoftmaxSampler` and deterministic `ArgmaxSampler`.
- `simulate_step` orchestration and the compositional `DecisionSystem` facade.
- Domain-independent validation errors for empty action sets, invalid logits,
  invalid probability distributions, and action-count mismatches.
- Type information via the `py.typed` marker.
- Unit tests, an end-to-end example, and production documentation.

### Fixed

- Package metadata now uses an SPDX license expression accepted by current
  build backends.
- Source distributions include the documentation, example, test support files,
  citation metadata, and contributor guidance needed for a complete checkout.

### Planned

- Gather evidence for optional framework adapters and batched execution without
  expanding the core runtime contract prematurely.
