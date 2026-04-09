#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import traceback
from typing import Iterable, List

from openai import OpenAI

from app.config import get_settings
from app.env import HiringOpenEnv
from app.graders import grade_task_state
from app.models import Action, ActionType, RewardOutput, TaskDefinition
from app.policy import choose_advances, choose_offer_candidate, choose_shortlist
from app.utils import OpenAIJustificationHelper, compact_json

BENCHMARK_NAME = "hiring-agent-openenv"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _submission_range(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _print_start(task: TaskDefinition, model_name: str) -> None:
    print(f"[START] task={task.task_id} env={BENCHMARK_NAME} model={model_name}")


def _print_step(step_index: int, action: Action, reward: RewardOutput, error: str | None) -> float:
    displayed_reward = _submission_range(reward.step_reward)
    action_str = f"{action.action_type.value}({compact_json(action.payload)})"
    error_text = "null" if error is None else error.replace("\n", " ").strip()
    done_val = getattr(reward, "_done_value", False)
    print(
        f"[STEP] step={step_index} action={action_str} "
        f"reward={displayed_reward:.2f} done={_bool_text(done_val)} error={error_text}"
    )
    return displayed_reward


def _print_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{_submission_range(value):.2f}" for value in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} rewards={rewards_text}")


def _build_action_plan(task: TaskDefinition, env: HiringOpenEnv, helper: OpenAIJustificationHelper) -> List[Action]:
    state = env.state()
    shortlist = choose_shortlist(task, state.candidates)
    advances = choose_advances(task, shortlist, state.candidates)
    offer = choose_offer_candidate(task, advances, state.candidates)

    non_offer = [cid for cid in task.candidate_ids if cid != offer]
    hold_ids = [cid for cid in shortlist if cid != offer][:1]
    reject_ids = [cid for cid in non_offer if cid not in hold_ids]

    deterministic_summary = (
        "Selected candidate based on hard-requirement compliance, interview readiness, "
        "budget compatibility, and communication/leadership signal consistency."
    )
    llm_context = {
        "task_id": task.task_id,
        "offer_candidate_id": offer,
        "shortlist": shortlist,
        "advances": advances,
        "expected_comp_band": task.expected_compensation_band,
    }
    justification = helper.generate_or_fallback(deterministic_summary, llm_context)

    actions: List[Action] = [
        Action(action_type=ActionType.SHORTLIST_CANDIDATES, payload={"candidate_ids": shortlist}),
        Action(
            action_type=ActionType.ADVANCE_STAGE,
            payload={"candidate_ids": advances, "stage": "interview"},
        ),
    ]
    actions.append(
        Action(
            action_type=ActionType.FINALIZE_DECISION,
            payload={
                "offer_candidate_id": offer,
                "compensation_band": task.expected_compensation_band,
                "reject_candidate_ids": reject_ids,
                "hold_candidate_ids": hold_ids,
                "justification": justification,
            },
        )
    )
    return actions


def _task_order(tasks: Iterable[TaskDefinition]) -> List[TaskDefinition]:
    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    return sorted(tasks, key=lambda task: (difficulty_rank.get(task.difficulty.value, 99), task.task_id))


def _probe_litellm_proxy(settings) -> None:
    value = os.getenv("OPENENV_LLM_PROXY_PROBE", "1").strip().lower()
    probe_enabled = value in {"1", "true", "yes", "on"}
    if not probe_enabled:
        return
    if not settings.api_base_url or not settings.model_name or not settings.api_key:
        return

    try:
        client = OpenAI(base_url=settings.api_base_url, api_key=settings.api_key, timeout=8.0)
        client.responses.create(
            model=settings.model_name,
            input="Respond with the single token: OK",
            temperature=0,
            max_output_tokens=8,
        )
    except Exception:
        # Keep baseline execution robust even if the external endpoint is unavailable.
        pass


def run_task(env: HiringOpenEnv, task: TaskDefinition, helper: OpenAIJustificationHelper, model_name: str) -> float:
    env.reset(task.task_id)
    _print_start(task, model_name)

    actions = _build_action_plan(task, env, helper)
    rewards: List[float] = []
    success = False
    step_count = 0

    try:
        for step_index, action in enumerate(actions, start=1):
            step_count = step_index
            try:
                observation, reward = env.step(action)
                # Attach done flag for formatter without changing model schema.
                setattr(reward, "_done_value", observation.done)
                rewards.append(_print_step(step_index, action, reward, error=None))
                if observation.done:
                    success = True
                    break
            except Exception as action_exc:
                fallback_reward = RewardOutput(step_reward=0.01, progress_score=0.01, final_score=None)
                setattr(fallback_reward, "_done_value", False)
                rewards.append(_print_step(step_index, action, fallback_reward, error=str(action_exc)))
                success = False
                break
    except Exception:
        success = False
        if not rewards:
            rewards.append(0.01)
        _ = traceback.format_exc()
    finally:
        _print_end(success=success, steps=step_count, rewards=rewards)

    # Keep existing internal score path available for callers/tests.
    current_state = env.state()
    graded = grade_task_state(current_state.task, current_state, current_state.candidates)
    return graded.final_score


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic baseline inference for Hiring Agent OpenEnv")
    parser.add_argument("--task-id", default=None, help="Run only one task by task_id")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    settings = get_settings()

    # Required env vars are loaded from .env / runtime env.
    _ = {
        "API_BASE_URL": settings.api_base_url,
        "MODEL_NAME": settings.model_name,
        "API_KEY_present": bool(settings.api_key),
        "HF_TOKEN_present": bool(settings.hf_token),
    }
    _probe_litellm_proxy(settings)

    seed = settings.openenv_seed if args.seed is None else args.seed
    env = HiringOpenEnv(seed=seed)
    helper = OpenAIJustificationHelper()

    tasks = _task_order(env.list_tasks())
    if args.task_id:
        tasks = [task for task in tasks if task.task_id == args.task_id]
        if not tasks:
            return 1

    for task in tasks:
        run_task(env, task, helper, settings.model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
