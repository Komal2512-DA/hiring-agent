from app.env import HiringOpenEnv
from app.graders import grade_task_state
from app.models import Action, ActionType
from app.policy import choose_advances, choose_offer_candidate, choose_shortlist


def _run_baseline_task(env: HiringOpenEnv, task_id: str) -> float:
    env.reset(task_id)
    state = env.state()

    shortlist = choose_shortlist(state.task, state.candidates)
    env.step(Action(action_type=ActionType.SHORTLIST_CANDIDATES, payload={"candidate_ids": shortlist}))

    advances = choose_advances(state.task, shortlist, state.candidates)
    env.step(Action(action_type=ActionType.ADVANCE_STAGE, payload={"candidate_ids": advances, "stage": "interview"}))

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
                "justification": "pytest baseline justification",
            },
        )
    )

    current_state = env.state()
    result = grade_task_state(current_state.task, current_state, current_state.candidates)
    return result.final_score


def test_grader_scores_within_range():
    env = HiringOpenEnv(seed=42)
    for task in env.list_tasks():
        score = _run_baseline_task(env, task.task_id)
        assert 0.0 <= score <= 1.0

