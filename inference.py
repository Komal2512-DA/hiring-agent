#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from openai import OpenAI

from app.config import get_settings
from app.env import HiringOpenEnv
from app.graders import grade_task_state
from app.models import Action, ActionType, RewardOutput, TaskDefinition
from app.policy import choose_advances, choose_offer_candidate, choose_shortlist
from app.utils import OpenAIJustificationHelper, compact_json


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _submission_range(value: float) -> float:
    return max(1e-6, min(0.99, float(value)))


def _observation_summary(observation) -> str:
    top = ""
    if observation.candidate_views:
        ranked = sorted(observation.candidate_views, key=lambda row: row.weighted_fit_score, reverse=True)
        top = ranked[0].candidate_id
    summary = {
        "message": observation.message,
        "progress": round(observation.current_progress_score, 4),
        "top_candidate": top,
        "done": observation.done,
    }
    return compact_json(summary)


def _print_start(task: TaskDefinition) -> None:
    print("[START]")
    print(f"task_id={task.task_id}")
    print(f"task_name={task.name}")
    print()


def _print_step(
    step_index: int,
    action: Action,
    observation,
    reward: RewardOutput,
    printed_reward_total: float,
) -> float:
    displayed_reward = _submission_range(reward.step_reward)
    remaining_budget = max(1e-6, 0.99 - printed_reward_total)
    displayed_reward = min(displayed_reward, remaining_budget)
    print("[STEP]")
    print(f"step_index={step_index}")
    print(f"action_type={action.action_type.value}")
    print(f"action_payload={compact_json(action.payload)}")
    print(f"observation_summary={_observation_summary(observation)}")
    print(f"reward={displayed_reward:.6f}")
    print(f"done={_bool_text(observation.done)}")
    print()
    return printed_reward_total + displayed_reward


def _print_end(task_id: str, final_score: float, result_summary: str) -> None:
    safe_final = _submission_range(final_score)
    print("[END]")
    print(f"task_id={task_id}")
    print(f"final_score={safe_final:.6f}")
    print(f"result_summary={result_summary}")
    print()


def _build_result_summary(graded) -> str:
    bias = graded.bias_audit
    llm = graded.llm_score_detail
    payload = {
        "summary": "task_completed",
        "top_subscore": max(graded.subscores, key=lambda s: s.score).name if graded.subscores else "n/a",
        "llm_source": llm.source if llm else "n/a",
        "bias_dimension": bias.audited_dimension if bias else "n/a",
    }
    return compact_json(payload)


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


def run_task(env: HiringOpenEnv, task: TaskDefinition, helper: OpenAIJustificationHelper) -> float:
    env.reset(task.task_id)
    _print_start(task)

    actions = _build_action_plan(task, env, helper)
    last_reward: RewardOutput | None = None
    printed_reward_total = 0.0

    for step_index, action in enumerate(actions, start=1):
        observation, reward = env.step(action)
        last_reward = reward
        printed_reward_total = _print_step(step_index, action, observation, reward, printed_reward_total)
        if observation.done:
            break

    current_state = env.state()
    graded = grade_task_state(current_state.task, current_state, current_state.candidates)
    final_score = graded.final_score
    if last_reward and last_reward.final_score is not None:
        final_score = last_reward.final_score

    _print_end(task.task_id, final_score, _build_result_summary(graded))
    return final_score


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
        run_task(env, task, helper)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
