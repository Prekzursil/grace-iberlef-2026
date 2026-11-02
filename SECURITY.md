# Security Policy

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-reviewing/privately-reporting-a-security-vulnerability)
on this repository (**Security → Report a vulnerability**). Do not open a
public issue for anything security-sensitive.

## Scope

The maintained research code (`grace/`, `scripts/`, `tests/`) is in scope.
`FINAL/` is the frozen record of the submitted IberLEF 2026 runs — it is
excluded by name from lint/type/coverage/CodeQL scopes (see
`pyproject.toml [tool.ruff]` and `.github/codeql/codeql-config.yml`) and is
never rewritten; a report concerning those bytes is still welcome here and
will be handled as a follow-up on the maintained code.

## Handling

- Dependabot security updates are enabled; findings are fixed by version
  bumps, never suppressed.
- CodeQL (security-and-quality over the declared scope) and the lean quality
  gate (gitleaks, opengrep, osv-scanner at zero tolerance) run on every
  pull request against `master`.
