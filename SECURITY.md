# Security Policy

## Supported versions

Security fixes are provided for the latest released minor version.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| Earlier or unreleased versions | No |

Pre-1.0 releases may contain breaking API changes, but security fixes will be
called out explicitly in release notes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the private
security-reporting channel published by the source host or software distributor.
If this copy has no published private channel, contact the person or organization
that supplied it and request a private route to the maintainers before sharing
technical details. A canonical private contact must be configured before a
public release. Include:

- affected versions and environment;
- a minimal reproducer or proof of concept;
- expected and observed impact;
- suggested mitigation, if known;
- whether the issue is already public.

Maintainers aim to acknowledge a report within three business days and provide
an initial assessment within seven business days. Timelines may vary with
severity and complexity. Reporters will receive updates when the assessment,
fix, and coordinated disclosure plan change.

## Scope

Relevant issues include distribution or build compromise, unsafe deserialization
introduced by this package, validation bypasses that violate documented runtime
invariants, and disclosure of information by library code. Domain-model errors,
unsafe downstream component implementations, and adversarial model behavior are
normally outside the core package's security boundary, though maintainers still
welcome private reports when scope is uncertain.

Please allow a reasonable remediation period before public disclosure.
