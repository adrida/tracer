# Security Policy

## Supported versions

Tracer (`tracer-llm`) is under active development. Security fixes are applied to
the latest released version on PyPI. Please upgrade to the most recent release
before reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public
issue, pull request, or discussion for anything security-sensitive.

Preferred: use GitHub's private vulnerability reporting via the
[**Security → Report a vulnerability**](https://github.com/adrida/tracer/security/advisories/new)
tab on this repository.

Alternatively, email **adam@tracerml.ai** with:

- a description of the issue and the potential impact,
- steps to reproduce or a proof of concept,
- affected version(s) and environment, and
- any suggested mitigation, if you have one.

## What to expect

- **Acknowledgement** within 3 business days.
- **Initial assessment** (severity and next steps) within 7 business days.
- We will keep you updated as we work on a fix and coordinate a disclosure
  timeline with you. We aim to release a patch for confirmed vulnerabilities
  within 90 days of the report.
- With your permission, we are happy to credit you in the release notes once a
  fix is published.

## Scope

This policy covers the code in this repository (the `tracer-llm` SDK and its
documentation). Tracer runs your own models on your own infrastructure with your
own API keys; issues in third-party dependencies, model providers, or your own
deployment configuration are out of scope here — please report those to the
relevant vendor.

Thank you for helping keep Tracer and its users safe.
