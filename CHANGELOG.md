# Changelog

All notable changes to this project are documented in this file.

## [1.1.0] - 2026-04-07

### Added

- Dedicated bias auditor module with typed outputs and disparity metrics.
- Dedicated LLM scorer module with deterministic fallback.
- Bias audit and LLM scoring integration in final grader outputs.
- Additional tests for bias audit and LLM scoring behavior.
- OSS contribution pack:
  - `LICENSE`
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - issue templates
  - PR template

### Changed

- Grader dimension list expanded with `bias_audit` and `llm_decision_quality`.
- Requirements adjusted for Python 3.14 compatibility (`pydantic==2.12.5`).

## [1.0.0] - 2026-04-07

### Added

- Initial release of Hiring Agent OpenEnv environment.
- Easy/medium/hard task registry with deterministic fixtures.
- FastAPI service with `/health`, `/tasks`, `/reset`, `/step`, `/state`.
- Deterministic baseline `inference.py` with strict structured stdout format.
- Pre-submission validator and automated tests.
- Docker and HF Spaces deployment support.

