from app.bias_auditor import run_bias_audit
from app.env import HiringOpenEnv
from app.llm_scorer import DecisionLLMScorer
from app.models import Action, ActionType
from app.policy import choose_advances, choose_offer_candidate, choose_shortlist


def _build_finalized_state(task_id: str):
    env = HiringOpenEnv(seed=42)
    env.reset(task_id)
    state = env.state()

    shortlist = choose_shortlist(state.task, state.candidates)
    env.step(Action(action_type=ActionType.SHORTLIST_CANDIDATES, payload={"candidate_ids": shortlist}))

    advances = choose_advances(state.task, shortlist, state.candidates)
    env.step(Action(action_type=ActionType.ADVANCE_STAGE, payload={"candidate_ids": advances, "stage": "interview"}))

    for cid in advances:
        env.step(Action(action_type=ActionType.SUMMARIZE_FIT, payload={"candidate_id": cid, "summary": "test summary"}))

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
                "justification": "Evidence-backed final decision for test.",
            },
        )
    )
    return env.state()


def test_bias_audit_output_is_valid():
    state = _build_finalized_state("task_hard_hiring_manager_e2e")
    audit = run_bias_audit(state.task, state, state.candidates)
    assert 0.0 <= audit.overall_score <= 1.0
    assert 0.0 <= audit.adverse_impact_ratio <= 1.0
    assert audit.metrics
    assert audit.audited_dimension


def test_llm_scorer_fallback_or_disabled_is_valid():
    state = _build_finalized_state("task_medium_tradeoff_ml")
    scorer = DecisionLLMScorer()
    detail = scorer.score(state.task, state, state.candidates)
    assert 0.0 <= detail.score <= 1.0
    assert detail.source in {"disabled", "fallback", "llm"}
    assert detail.rationale

