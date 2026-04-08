---
title: Hiring Agent OpenEnv
emoji: "📋"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Hiring Agent OpenEnv (Round-1 Submission Ready)

A complete, real-world OpenEnv environment for training and evaluating AI hiring agents.

## Problem Framing

This environment models a realistic hiring pipeline:

- requisition intake and constraints
- candidate screening against hard requirements
- shortlist and interview progression
- interviewer assignment
- fit summarization and evidence capture
- compensation-band recommendation
- final hire/no-hire decision with justification

The objective is to let an agent learn structured hiring actions with interpretable reward signals across process quality and outcome correctness.

## OpenEnv API

Implemented in `app/env.py`:

- `reset(task_id: Optional[str]) -> Observation`
- `step(action: Action) -> tuple[Observation, RewardOutput]`
- `state() -> EnvironmentState`

Service endpoints in `app/main.py`:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## Typed Models

Defined in `app/models.py`:

- `Action`, `Observation`, `EnvironmentState`
- `TaskDefinition`, `GraderResult`, `RewardOutput`
- Domain entities:
  - `CandidateProfile`
  - `JobRequisition`
  - `HiringConstraintSet`
  - `InterviewFeedback`
  - `HiringDecision`
  - `PipelineStage`
  - `CompensationBand`

## Task Set (Easy -> Medium -> Hard)

1. `task_easy_screen_backend`
2. `task_medium_tradeoff_ml`
3. `task_hard_hiring_manager_e2e`

Each task has:

- deterministic fixtures
- automated grading
- normalized task-level final score in strict `(0.0, 1.0)`
- partial progress reward signals

## Action Space

- `screen_candidate`
- `shortlist_candidates`
- `advance_stage`
- `assign_interviewer`
- `request_interview`
- `reject_candidate`
- `hold_candidate`
- `choose_compensation_band`
- `summarize_fit`
- `finalize_decision`

## Observation / State Space

Observation includes:

- task metadata (id, name, objective, difficulty)
- step index and done status
- candidate-level stage and weighted fit score
- pipeline overview
- progress score and environment message

State includes:

- active task definition
- candidate map
- stage map
- shortlist and interview advances
- interviewer assignments
- fit summaries
- compensation choice
- final decision
- action history

## Grading and Reward Logic

Graders are in `app/graders.py` with normalized subscores:

- hard requirement compliance
- shortlist quality
- progression quality
- final decision quality
- consistency and justification
- feedback alignment
- fairness and process guardrails
- bias audit
- llm decision quality
- efficiency

Bias auditing is implemented in `app/bias_auditor.py` and is exposed via typed models in environment state and grader outputs.
LLM scoring is implemented in `app/llm_scorer.py` and uses deterministic fallback scoring when disabled or unavailable.

Reward in `step()`:

- `progress_score` from grading over current state
- `step_reward` as positive incremental progress
- `final_score` when task is done

## Required Environment Variables

Set these in `.env` or platform secrets:

- `API_BASE_URL` (for OpenAI client base URL)
- `MODEL_NAME` (model id for optional LLM justification)
- `API_KEY` (required API key used by OpenAI client)

Optional:

- `HF_TOKEN` (legacy fallback only if `API_KEY` is not set)
- `OPENENV_SEED` (default `42`)
- `PORT` (default `7860`)
- `USE_LLM_JUSTIFICATION` (`0`/`1`)
- `USE_LLM_SCORING` (`0`/`1`)
- `OPENENV_LLM_PROXY_PROBE` (`1` by default; sends one lightweight proxy call in `inference.py`)

All LLM calls use the OpenAI Client (`openai` package), configured from these env vars.

## Hackathon Prerequisites (from Scaler Dashboard)

- Python `3.10`, `3.11`, or `3.12` recommended for submission compatibility.
- Git + GitHub access for source submission.
- Docker installed and running (for container build checks).
- Hugging Face CLI configured (for Space deployment).
- OpenEnv CLI available (`pip install openenv-core`) for `openenv validate`.

## Local Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Sanity check:

```bash
curl http://localhost:7860/health
```

## Baseline Inference

`inference.py`:

- deterministic task ordering
- fixed seed (configurable)
- transparent heuristic policy
- end-to-end action flow for all tasks
- strict structured stdout with:
  - `[START]`
  - `[STEP]`
  - `[END]`

Run:

```bash
python inference.py
python inference.py --task-id task_medium_tradeoff_ml
```

## Validator (Pre-Submission)

Run:

```bash
python validator.py
```

Official-style precheck (matches the hackathon pre-validation flow):

```bash
python validator.py --strict-precheck --hf-space-url https://your-space.hf.space
```

Alternative shell script (Linux/macOS):

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space.hf.space .
```

Equivalent granular flags:

```bash
python validator.py --hf-space-url https://your-space.hf.space --check-docker-build --check-openenv-validate
```

Validator checks:

- required file presence
- `openenv.yaml` structure
- endpoint behavior (`/health`, `/reset`, `/step`, `/state`)
- task registry has at least 3 tasks with easy/medium/hard
- task-level grader outputs in strict `(0.0, 1.0)`
- inference script exists and produces strict `[START]/[STEP]/[END]` format
- env var references for `API_BASE_URL`, `MODEL_NAME`, `API_KEY`
- Dockerfile essentials
- optional HF Space `/reset` ping check
- optional real `docker build` check
- optional `openenv validate` check

## Tests

```bash
python -m pytest -q tests
```

## Hugging Face Spaces (Docker)

1. Create a new **Docker Space**.
2. Push this repository.
3. Set Space secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `API_KEY`
4. Space should expose:
   - `/health` (returns `200`)
   - `/reset` (works after boot)

Docker runtime is lightweight and designed for 2 vCPU / 8 GB RAM.

## Submission Notes

- Keep `inference.py` at project root.
- Keep strict stdout block/field ordering in inference logs.
- Ensure deployed Space URL responds before final submission.

## OSS Readiness

This repo includes standard open-source contribution assets:

- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `CHANGELOG.md`
- `.github/ISSUE_TEMPLATE/`
- `.github/PULL_REQUEST_TEMPLATE.md`
