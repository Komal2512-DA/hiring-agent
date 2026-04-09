#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from fastapi.testclient import TestClient

from app.env import HiringOpenEnv
from app.graders import grade_task_state
from app.main import app
from app.models import Action, ActionType
from app.policy import choose_advances, choose_offer_candidate, choose_shortlist


@dataclass(frozen=True)
class ValidationOptions:
    root: Path
    hf_space_url: str | None
    run_docker_build: bool
    run_openenv_validate: bool
    inference_timeout_seconds: int
    command_timeout_seconds: int


def _check_required_files(root: Path) -> List[str]:
    required = [
        "openenv.yaml",
        "__init__.py",
        "client.py",
        "models.py",
        "README.md",
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "inference.py",
        "validator.py",
        ".env.example",
        ".gitignore",
        "scripts/validate-submission.sh",
        "app/main.py",
        "app/config.py",
        "app/models.py",
        "app/env.py",
        "app/tasks.py",
        "app/graders.py",
        "app/bias_auditor.py",
        "app/llm_scorer.py",
        "app/data.py",
        "app/utils.py",
        "app/policy.py",
        "app/fixtures/roles.json",
        "app/fixtures/candidates.json",
        "app/fixtures/interviews.json",
        "tests/test_api.py",
        "tests/test_graders.py",
        "tests/test_tasks.py",
        "tests/test_inference.py",
        "tests/test_bias_and_llm.py",
    ]
    missing = [item for item in required if not (root / item).exists()]
    return [f"Missing required file: {item}" for item in missing]


def _check_openenv_yaml(root: Path) -> List[str]:
    path = root / "openenv.yaml"
    if not path.exists():
        return ["openenv.yaml missing"]

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: List[str] = []

    for key in ("name", "entrypoint", "api", "tasks", "evaluation"):
        if key not in data:
            errors.append(f"openenv.yaml missing key: {key}")

    api = data.get("api", {})
    for key in ("reset_endpoint", "step_endpoint", "state_endpoint"):
        if key not in api:
            errors.append(f"openenv.yaml api missing key: {key}")

    return errors


def _run_baseline_once(env: HiringOpenEnv, task_id: str) -> float:
    env.reset(task_id)
    state = env.state()

    shortlist = choose_shortlist(state.task, state.candidates)
    env.step(Action(action_type=ActionType.SHORTLIST_CANDIDATES, payload={"candidate_ids": shortlist}))

    advances = choose_advances(state.task, shortlist, state.candidates)
    env.step(Action(action_type=ActionType.ADVANCE_STAGE, payload={"candidate_ids": advances, "stage": "interview"}))

    for idx, cid in enumerate(advances, start=1):
        env.step(Action(action_type=ActionType.ASSIGN_INTERVIEWER, payload={"candidate_id": cid, "interviewer_id": f"INT-{idx}"}))

    for cid in advances:
        env.step(Action(action_type=ActionType.SUMMARIZE_FIT, payload={"candidate_id": cid, "summary": "validator summary"}))

    env.step(Action(action_type=ActionType.CHOOSE_COMPENSATION_BAND, payload={"compensation_band": state.task.expected_compensation_band}))

    offer = choose_offer_candidate(state.task, advances, state.candidates)
    reject_ids = [cid for cid in state.task.candidate_ids if cid != offer]
    env.step(
        Action(
            action_type=ActionType.FINALIZE_DECISION,
            payload={
                "offer_candidate_id": offer,
                "compensation_band": state.task.expected_compensation_band,
                "reject_candidate_ids": reject_ids,
                "hold_candidate_ids": [],
                "justification": "Deterministic baseline decision.",
            },
        )
    )

    current_state = env.state()
    graded = grade_task_state(current_state.task, current_state, current_state.candidates)
    return graded.final_score


def _check_env_and_graders() -> List[str]:
    errors: List[str] = []
    env = HiringOpenEnv(seed=42)

    tasks = env.list_tasks()
    if len(tasks) < 3:
        errors.append("Task registry has fewer than 3 tasks.")

    difficulties = {task.difficulty.value for task in tasks}
    for required in {"easy", "medium", "hard"}:
        if required not in difficulties:
            errors.append(f"Missing required difficulty task: {required}")

    for task in tasks:
        score = _run_baseline_once(env, task.task_id)
        if not (0.0 < score < 1.0):
            errors.append(f"Task {task.task_id} grader score must be strictly in (0,1): {score}")

    return errors


def _check_api_endpoints() -> List[str]:
    errors: List[str] = []
    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        errors.append("/health did not return 200.")

    reset = client.post("/reset", json={"task_id": "task_easy_screen_backend"})
    if reset.status_code != 200:
        errors.append("/reset did not return 200.")

    state = client.get("/state")
    if state.status_code != 200:
        errors.append("/state did not return 200.")

    step = client.post(
        "/step",
        json={
            "action_type": "shortlist_candidates",
            "payload": {"candidate_ids": ["C001", "C002"]},
        },
    )
    if step.status_code != 200:
        errors.append("/step did not return 200.")

    return errors


def _check_env_var_references(root: Path) -> List[str]:
    required_vars = ["API_BASE_URL", "MODEL_NAME", "API_KEY", "USE_LLM_SCORING"]
    files = [root / "app/config.py", root / "inference.py", root / "app/utils.py", root / ".env.example"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files if path.exists())

    errors: List[str] = []
    for key in required_vars:
        if key not in text:
            errors.append(f"Environment variable not referenced in code/docs: {key}")
    return errors


def _check_dockerfile(root: Path) -> List[str]:
    path = root / "Dockerfile"
    if not path.exists():
        return ["Dockerfile missing."]

    content = path.read_text(encoding="utf-8")
    required_tokens = ["FROM", "COPY", "pip install", "EXPOSE", "CMD"]
    return [f"Dockerfile missing token: {token}" for token in required_tokens if token not in content]


def _check_hf_space_reset(hf_space_url: str) -> List[str]:
    url = f"{hf_space_url.rstrip('/')}/reset"
    req = Request(url=url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return [f"HF Space /reset returned HTTP {resp.status}, expected 200: {url}"]
    except HTTPError as exc:
        return [f"HF Space /reset returned HTTP {exc.code}: {url}"]
    except URLError as exc:
        return [f"HF Space /reset unreachable: {url} ({exc.reason})"]
    except Exception as exc:  # pragma: no cover - defensive path
        return [f"HF Space /reset check failed for {url}: {exc}"]
    return []


def _tail_text(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _check_docker_build(root: Path, timeout_seconds: int) -> List[str]:
    if shutil.which("docker") is None:
        return ["docker command not found in PATH (required for docker build precheck)."]

    proc = subprocess.run(
        ["docker", "build", str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if proc.returncode == 0:
        return []

    details = _tail_text(f"{proc.stdout}\n{proc.stderr}".strip())
    return [f"Docker build failed with code {proc.returncode}:\n{details}"]


def _check_openenv_validate(root: Path, timeout_seconds: int) -> List[str]:
    openenv_cmd: str | None = shutil.which("openenv")
    if openenv_cmd is None:
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (
            exe_dir / "openenv",
            exe_dir / "openenv.exe",
            exe_dir / "openenv.cmd",
            exe_dir / "openenv.bat",
        ):
            if candidate.exists():
                openenv_cmd = str(candidate)
                break

    if openenv_cmd is None:
        return [
            "openenv command not found in PATH or current Python Scripts dir (install openenv-core for official precheck)."
        ]

    proc = subprocess.run(
        [openenv_cmd, "validate"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if proc.returncode == 0:
        return []

    details = _tail_text(f"{proc.stdout}\n{proc.stderr}".strip())
    return [f"openenv validate failed with code {proc.returncode}:\n{details}"]


def _parse_inference_output_for_format(output: str) -> List[str]:
    errors: List[str] = []
    lines = [line.rstrip("\n") for line in output.splitlines()]

    i = 0
    start_blocks = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "[START]":
            start_blocks += 1
            expected = ["task_id=", "task_name="]
            for idx, prefix in enumerate(expected, start=1):
                if i + idx >= len(lines) or not lines[i + idx].startswith(prefix):
                    errors.append("[START] block field order mismatch.")
            i += 3
            continue

        if line == "[STEP]":
            expected = [
                "step_index=",
                "action_type=",
                "action_payload=",
                "observation_summary=",
                "reward=",
                "done=",
            ]
            for idx, prefix in enumerate(expected, start=1):
                if i + idx >= len(lines) or not lines[i + idx].startswith(prefix):
                    errors.append("[STEP] block field order mismatch.")
                    break
            i += 7
            continue

        if line == "[END]":
            expected = ["task_id=", "final_score=", "result_summary="]
            for idx, prefix in enumerate(expected, start=1):
                if i + idx >= len(lines) or not lines[i + idx].startswith(prefix):
                    errors.append("[END] block field order mismatch.")
            i += 4
            continue

        if line:
            errors.append(f"Unexpected non-structured log line: {line}")
        i += 1

    if start_blocks == 0:
        errors.append("No [START] blocks found in inference output.")

    return errors


def _check_inference_runtime_and_format(root: Path, timeout_seconds: int) -> List[str]:
    inherited = dict(os.environ)
    inherited["USE_LLM_JUSTIFICATION"] = "0"
    inherited["OPENENV_LLM_PROXY_PROBE"] = "0"

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=inherited,
    )

    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"inference.py exited with code {proc.returncode}: {proc.stderr.strip()}")
        return errors

    errors.extend(_parse_inference_output_for_format(proc.stdout))
    return errors


def run_validation(options: ValidationOptions) -> List[str]:
    root = options.root
    errors: List[str] = []
    errors.extend(_check_required_files(root))
    errors.extend(_check_openenv_yaml(root))
    errors.extend(_check_env_and_graders())
    errors.extend(_check_api_endpoints())
    errors.extend(_check_env_var_references(root))
    errors.extend(_check_dockerfile(root))
    errors.extend(_check_inference_runtime_and_format(root, timeout_seconds=options.inference_timeout_seconds))

    if options.hf_space_url:
        errors.extend(_check_hf_space_reset(options.hf_space_url))
    if options.run_docker_build:
        errors.extend(_check_docker_build(root, timeout_seconds=options.command_timeout_seconds))
    if options.run_openenv_validate:
        errors.extend(_check_openenv_validate(root, timeout_seconds=options.command_timeout_seconds))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-submission validator for Hiring Agent OpenEnv project")
    parser.add_argument("--root", default=".", help="Project root path")
    parser.add_argument("--hf-space-url", default=None, help="Optional HF Space base URL for /reset ping check")
    parser.add_argument("--check-docker-build", action="store_true", help="Run real docker build check")
    parser.add_argument("--check-openenv-validate", action="store_true", help="Run `openenv validate` check")
    parser.add_argument(
        "--strict-precheck",
        action="store_true",
        help="Run official-style 3 checks: HF /reset ping, docker build, openenv validate",
    )
    parser.add_argument(
        "--inference-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for inference.py execution",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=600,
        help="Timeout for docker/openenv command checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    run_docker_build = bool(args.check_docker_build or args.strict_precheck)
    run_openenv_validate = bool(args.check_openenv_validate or args.strict_precheck)
    hf_space_url = args.hf_space_url

    errors: List[str] = []
    if args.strict_precheck and not hf_space_url:
        errors.append("--strict-precheck requires --hf-space-url (example: https://your-space.hf.space)")

    options = ValidationOptions(
        root=root,
        hf_space_url=hf_space_url,
        run_docker_build=run_docker_build,
        run_openenv_validate=run_openenv_validate,
        inference_timeout_seconds=args.inference_timeout_seconds,
        command_timeout_seconds=args.command_timeout_seconds,
    )

    errors.extend(run_validation(options))
    if errors:
        print("VALIDATION_FAILED")
        for idx, err in enumerate(errors, start=1):
            print(f"{idx:02d}. {err}")
        return 1

    print("VALIDATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
