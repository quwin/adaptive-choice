# Changelog

All notable changes to Adaptive Choice are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-09

### Added

- Generic `DecisionExperience` values for the observation, action, and outcome
  produced by a decision step.
- The separately distributed `adaptive-choice-torch` package, keeping PyTorch
  out of the core dependency graph.
- `TorchChoiceModel` for composing application tensor encoders and scorers under
  the existing scalar `ChoiceModel` contract.
- `DotProductScorer` and `MLPScorer` tensor modules.
- Mask-preserving candidate padding and logit unpadding utilities for dynamic
  candidate batches.
- Tensor validation and documentation covering shapes, ordering, devices,
  dtypes, gradients, model mode, and padded candidates.

### Changed

- `AgentUpdater` now consumes one application-typed perceived experience rather
  than four decision-specific arguments, allowing adaptation from observed
  events, received information, and directly experienced outcomes.
- Package and citation versions identify the 0.2.0 release.
- Roadmap and integration documentation now distinguish tensor batching from
  scalar environment execution.

## [0.1.0] - 2026-08-09

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
