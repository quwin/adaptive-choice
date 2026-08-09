# Contributing to Adaptive Choice

Thank you for helping improve Adaptive Choice. Contributions of code,
documentation, tests, examples, and focused design discussion are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
For vulnerabilities, use the private process in [SECURITY.md](SECURITY.md)
instead of opening a public issue.

## Before proposing a change

Adaptive Choice intentionally owns only the decision boundary between a
simulation and an adaptive chooser. Before adding an abstraction, ask:

1. Is it domain-independent?
2. Does it preserve dynamic legal-action sets?
3. Can it be expressed through composition and structural typing?
4. Does it keep randomness explicit?
5. Are at least two concrete implementations likely to need it?

Training systems, simulation engines, domain schemas, and framework-specific
models normally belong in adapters or downstream projects. Start a design issue
before making a public-API change or adding a dependency.

## Development setup

Adaptive Choice requires a supported Python 3 release and has no mandatory
runtime dependency outside the standard library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,docs]'
```

Run the standard-library test suite:

```bash
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy src
```

Build and inspect distributions before submitting packaging changes:

```bash
python -m pip install twine
python -m build
python -m twine check dist/*
```

To preview the documentation:

```bash
python -m pip install mkdocs
mkdocs serve
```

## Change requirements

- Add or update tests for observable behavior.
- Keep public functions and protocols fully typed.
- Document every public API addition and user-visible error.
- Preserve candidate ordering: `actions[i]` must correspond to `logits[i]` and
  `probabilities[i]`.
- Never introduce hidden process-global randomness.
- Prefer immutable return values and avoid mutating user-owned agent objects.
- Do not make a machine-learning framework a core dependency.
- Add an entry under `Unreleased` in `CHANGELOG.md` for user-visible changes.

## Testing guidance

Use small, deterministic component tests. An integration test should fix the
initial world state, agent state, component implementations, and RNG seed, then
assert the complete `StepResult`. Statistical sampler tests should use generous
confidence bounds and fixed seeds; avoid snapshots tied to a specific Python
minor version's random stream unless exact stream compatibility is the subject
of the test.

## Pull requests

Keep each pull request focused. Include:

- the problem and intended behavior;
- design-boundary implications;
- tests performed;
- typing and documentation impact;
- compatibility or migration notes, when applicable.

Maintainers may ask to split unrelated changes. Approval and passing CI are
required before merge.

## Releases

Maintainers release from a clean main branch:

1. Confirm the version in package metadata and `CITATION.cff`, and add the
   actual `date-released` to `CITATION.cff`.
2. Move changelog entries from `Unreleased` to a dated release heading.
3. Run tests and build checks on every supported Python version.
4. Inspect wheel and source-distribution contents.
5. Tag `vX.Y.Z`, publish release notes, and publish the distributions through a
   trusted release workflow.

Do not commit credentials or publish from an unreviewed local tree.
