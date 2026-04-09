# Contributing Guide

Thanks for contributing to the Hiring Agent OpenEnv project.

## What To Contribute

- New hiring tasks with deterministic fixtures and expected outcomes.
- Grader improvements that increase interpretability and stability.
- Bias audit improvements with documented rationale.
- API, validator, and deployment reliability fixes.
- Docs and examples that improve reproducibility.

## Local Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate Before PR

```bash
python -m pytest -q tests
python validator.py
python inference.py
```

Optional full smoke run:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all_checks.ps1 -VenvPath .venv_clean -SkipInstall
```

## Coding Standards

- Keep runtime lightweight and deterministic.
- Preserve `step/reset/state` OpenEnv contract.
- Keep grader outputs in `[0.0, 1.0]`.
- Keep inference stdout format stable: `[START]`, `[STEP]`, `[END]`.
- Use typed models for new domain objects.
- Add or update tests for every behavior change.

## Pull Request Checklist

- [ ] Task, grader, or infra behavior is documented.
- [ ] Tests are updated and passing.
- [ ] `validator.py` passes.
- [ ] No secrets in committed files.
- [ ] Backward compatibility of API and logging format is maintained.

## Reporting Security Issues

Do not open public issues for sensitive security vulnerabilities. Share a private report with maintainers first.

