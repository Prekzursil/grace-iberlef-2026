# Contributing

## Lean quality gates (CI, required on every PR)

`.github/workflows/quality.yml` calls quality-zero-platform's reusable lean
6-gate workflow, pinned to a full commit SHA. For the languages in this
repository (Python):

1. **Lint + format** — ruff (`ruff check` + `ruff format --check`)
2. **Types** — basedpyright at `typeCheckingMode: standard` (config in
   `pyproject.toml [tool.basedpyright]`)
3. **Tests + coverage** — pytest + pytest-cov, **STRICT 100% line AND branch**
   on the declared scope (`grace/` and `scripts/`; `FINAL/` is the frozen
   record of the submitted runs and is excluded by name from every quality
   surface — see `pyproject.toml`)
4. **SAST** — opengrep (always on)
5. **Secrets** — gitleaks (always on; `.gitleaks.toml` is a reviewed
   allowlist only)
6. **Deps** — osv-scanner + Dependabot at `deps-severity-floor: any`
   (every finding blocks; `osv-scanner.toml` carries only reviewed, dated,
   reasoned ignores)

CodeQL (`security-and-quality`) runs alongside via the thin caller in
`.github/workflows/codeql.yml`, scoped by `.github/codeql/codeql-config.yml`
(paths: `grace`, `tests`).

## Local commands

```bash
python -m pip install -r requirements.txt
python -m pytest                 # full suite
bash scripts/verify              # the lean gate, exactly as CI runs it
```

## Conventions

- The default branch is **`master`**. Every change reaches it through a pull
  request; never push to the default branch.
- `FINAL/` is frozen: byte-exact record of the submitted runs. Never edit it;
  fixes land in the maintained trees (`grace/`, `scripts/`, `tests/`).
- New behavior ships with the tests that pin it; keep coverage at 100/100 on
  the declared scope.
- Security findings are fixed, never muted; an ignore entry needs a written
  reason and a date.
