from __future__ import annotations

from typing import Dict, List

from app.bias_auditor import run_bias_audit
from app.llm_scorer import DecisionLLMScorer
from app.models import (
    BiasAuditResult,
    CandidateProfile,
    EnvironmentState,
    GraderResult,
    GraderSubscore,
    LLMScoreDetail,
    TaskDefinition,
)
from app.utils import (
    clamp01,
    clamp_open01,
    candidate_hard_filter,
    f1_overlap,
    location_diversity_score,
    mean_feedback_score,
)

_LLM_SCORER = DecisionLLMScorer()
SCORE_MIN = 0.1
SCORE_MAX = 0.999999
SCORE_MID = 0.5


def _hard_requirement_compliance(
    task: TaskDefinition, state: EnvironmentState, candidates: Dict[str, CandidateProfile]
) -> GraderSubscore:
    considered = set(state.shortlist) | set(state.interview_advances)
    if state.final_decision and state.final_decision.offer_candidate_id:
        considered.add(state.final_decision.offer_candidate_id)

    if not considered:
        return GraderSubscore(
            name="hard_requirement_compliance",
            score=SCORE_MIN,
            rationale="No candidates were acted on yet.",
        )

    total = len(considered)
    passed = 0
    for cid in considered:
        profile = candidates.get(cid)
        if profile and candidate_hard_filter(profile, task.job_requisition):
            passed += 1

    score = clamp_open01(passed / max(1, float(total)), epsilon=1e-6)
    return GraderSubscore(
        name="hard_requirement_compliance",
        score=score,
        rationale=f"{passed}/{total} acted-on candidates satisfy hard requirements.",
    )


def _shortlist_quality(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    score = f1_overlap(task.expected_shortlist, state.shortlist)
    return GraderSubscore(
        name="shortlist_quality",
        score=score,
        rationale=f"F1 overlap with expected shortlist is {score:.3f}.",
    )


def _progression_quality(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    score = f1_overlap(task.expected_advances, state.interview_advances)
    return GraderSubscore(
        name="progression_quality",
        score=score,
        rationale=f"F1 overlap with expected interview advances is {score:.3f}.",
    )


def _final_decision_quality(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    decision = state.final_decision
    if decision is None:
        return GraderSubscore(
            name="final_decision_quality",
            score=SCORE_MIN,
            rationale="Final decision was not submitted.",
        )

    offer_match = SCORE_MAX if decision.offer_candidate_id == task.expected_offer_candidate_id else SCORE_MIN
    band = decision.compensation_band or state.chosen_compensation_band or ""
    band_match = SCORE_MAX if band == task.expected_compensation_band else SCORE_MIN

    offer_id = decision.offer_candidate_id or ""
    rejected = set(decision.reject_candidate_ids)
    held = set(decision.hold_candidate_ids)
    exclusivity = SCORE_MAX
    if offer_id and (offer_id in rejected or offer_id in held):
        exclusivity = SCORE_MIN

    non_offer_pool = [cid for cid in task.candidate_ids if cid != offer_id]
    decided = [cid for cid in non_offer_pool if cid in rejected or cid in held]
    closure = clamp_open01(len(decided) / max(1, float(len(non_offer_pool))), epsilon=1e-6)

    score = clamp_open01((0.45 * offer_match) + (0.25 * band_match) + (0.15 * exclusivity) + (0.15 * closure), epsilon=1e-6)
    return GraderSubscore(
        name="final_decision_quality",
        score=score,
        rationale=(
            f"offer_match={offer_match:.2f}, band_match={band_match:.2f}, "
            f"exclusivity={exclusivity:.2f}, closure={closure:.2f}"
        ),
    )


def _consistency_and_justification(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    decision = state.final_decision
    if decision is None:
        return GraderSubscore(
            name="consistency_and_justification",
            score=SCORE_MIN,
            rationale="No final decision to evaluate for consistency/justification.",
        )

    justification = (decision.justification or "").strip()
    justification_score = clamp01(len(justification) / 80.0)

    offer = decision.offer_candidate_id or ""
    in_advances = SCORE_MAX if (not offer or offer in state.interview_advances) else SCORE_MIN

    band_consistency = SCORE_MAX
    if state.chosen_compensation_band and decision.compensation_band:
        band_consistency = (
            SCORE_MAX if state.chosen_compensation_band == decision.compensation_band else SCORE_MIN
        )

    summary_coverage = SCORE_MIN
    if state.interview_advances:
        covered = [cid for cid in state.interview_advances if cid in state.fit_summaries]
        summary_coverage = clamp_open01(len(covered) / float(len(state.interview_advances)), epsilon=1e-6)

    score = clamp_open01(
        (0.35 * justification_score) + (0.25 * in_advances) + (0.20 * band_consistency) + (0.20 * summary_coverage),
        epsilon=1e-6,
    )
    return GraderSubscore(
        name="consistency_and_justification",
        score=score,
        rationale=(
            f"justification={justification_score:.2f}, in_advances={in_advances:.2f}, "
            f"band_consistency={band_consistency:.2f}, summary_coverage={summary_coverage:.2f}"
        ),
    )


def _feedback_alignment(state: EnvironmentState) -> GraderSubscore:
    decision = state.final_decision
    if decision is None:
        return GraderSubscore(
            name="feedback_alignment",
            score=SCORE_MIN,
            rationale="No final decision submitted yet.",
        )

    if not state.feedback:
        return GraderSubscore(
            name="feedback_alignment",
            score=SCORE_MID,
            rationale="No interview feedback fixtures available for this task.",
        )

    by_candidate: Dict[str, float] = {}
    for cid, rows in state.feedback.items():
        normalized_rows = [
            {"technical_score": row.technical_score, "communication_score": row.communication_score} for row in rows
        ]
        by_candidate[cid] = mean_feedback_score(normalized_rows)

    if not by_candidate:
        return GraderSubscore(
            name="feedback_alignment",
            score=SCORE_MID,
            rationale="Feedback map was empty after normalization.",
        )

    best_feedback = max(by_candidate.values())
    offered = decision.offer_candidate_id or ""
    offered_feedback = by_candidate.get(offered, SCORE_MIN)
    relative = SCORE_MIN if best_feedback == 0 else clamp_open01(offered_feedback / best_feedback, epsilon=1e-6)

    rec_bonus = 0
    for row in state.feedback.get(offered, []):
        mapped = {"advance": SCORE_MAX, "hold": 0.6, "reject": SCORE_MIN}.get(row.recommendation.lower(), 0.4)
        rec_bonus += mapped
    rec_bonus = clamp_open01(rec_bonus / max(1, float(len(state.feedback.get(offered, [])))), epsilon=1e-6)

    score = clamp_open01((0.75 * relative) + (0.25 * rec_bonus), epsilon=1e-6)
    return GraderSubscore(
        name="feedback_alignment",
        score=score,
        rationale=f"offered_feedback={offered_feedback:.2f}, best_feedback={best_feedback:.2f}, rec={rec_bonus:.2f}",
    )


def _fairness_and_process_guardrails(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    shortlist = state.shortlist
    diversity = location_diversity_score(shortlist, state.candidates) if shortlist else SCORE_MIN
    shortlist_bound = SCORE_MAX if len(shortlist) <= task.constraints.max_shortlist_size else SCORE_MIN
    advance_bound = SCORE_MAX if len(state.interview_advances) <= task.constraints.max_interview_advances else SCORE_MIN

    offer_band_ok = SCORE_MAX
    if state.final_decision and state.final_decision.compensation_band:
        offer_band_ok = (
            SCORE_MAX if state.final_decision.compensation_band == task.expected_compensation_band else SCORE_MIN
        )
    elif state.chosen_compensation_band:
        offer_band_ok = SCORE_MAX if state.chosen_compensation_band == task.expected_compensation_band else SCORE_MIN

    stage_conflicts = 0
    if state.final_decision:
        offered = state.final_decision.offer_candidate_id or ""
        if offered and offered in set(state.final_decision.reject_candidate_ids + state.final_decision.hold_candidate_ids):
            stage_conflicts += 1
    stage_consistency = SCORE_MAX if stage_conflicts == 0 else SCORE_MIN

    score = clamp_open01(
        (0.22 * diversity)
        + (0.26 * shortlist_bound)
        + (0.22 * advance_bound)
        + (0.20 * offer_band_ok)
        + (0.10 * stage_consistency),
        epsilon=1e-6,
    )
    return GraderSubscore(
        name="fairness_and_process_guardrails",
        score=score,
        rationale=(
            f"diversity={diversity:.2f}, shortlist_bound={shortlist_bound:.2f}, advance_bound={advance_bound:.2f}, "
            f"offer_band_ok={offer_band_ok:.2f}, stage_consistency={stage_consistency:.2f}"
        ),
    )


def _efficiency(task: TaskDefinition, state: EnvironmentState) -> GraderSubscore:
    if task.max_steps <= 1:
        return GraderSubscore(name="efficiency", score=SCORE_MAX, rationale="Degenerate max_steps configuration.")

    used_steps = max(0, state.step_index - 3)
    usable_budget = max(1, task.max_steps - 3)
    score = clamp_open01(SCORE_MAX - (used_steps / float(usable_budget)), epsilon=1e-6)
    return GraderSubscore(
        name="efficiency",
        score=score,
        rationale=f"step_index={state.step_index}, max_steps={task.max_steps}.",
    )


def _bias_audit_score(
    task: TaskDefinition,
    state: EnvironmentState,
    candidates: Dict[str, CandidateProfile],
) -> tuple[GraderSubscore, BiasAuditResult]:
    audit = run_bias_audit(task, state, candidates)
    score = clamp_open01(audit.overall_score, epsilon=1e-6)
    return (
        GraderSubscore(
            name="bias_audit",
            score=score,
            rationale=(
                f"dimension={audit.audited_dimension}, min_air={audit.adverse_impact_ratio:.2f}, "
                f"repr={audit.representation_parity_score:.2f}, advance={audit.advancement_parity_score:.2f}, "
                f"flagged={audit.flagged_metrics or ['none']}"
            ),
        ),
        audit,
    )


def _llm_decision_quality(
    task: TaskDefinition,
    state: EnvironmentState,
    candidates: Dict[str, CandidateProfile],
) -> tuple[GraderSubscore, LLMScoreDetail]:
    detail = _LLM_SCORER.score(task, state, candidates)
    return (
        GraderSubscore(
            name="llm_decision_quality",
            score=clamp_open01(detail.score, epsilon=1e-6),
            rationale=f"source={detail.source}; {detail.rationale}",
        ),
        detail,
    )


def grade_task_state(
    task: TaskDefinition, state: EnvironmentState, candidates: Dict[str, CandidateProfile]
) -> GraderResult:
    bias_subscore, bias_audit = _bias_audit_score(task, state, candidates)
    llm_subscore, llm_detail = _llm_decision_quality(task, state, candidates)

    subscores: List[GraderSubscore] = [
        _hard_requirement_compliance(task, state, candidates),
        _shortlist_quality(task, state),
        _progression_quality(task, state),
        _final_decision_quality(task, state),
        _consistency_and_justification(task, state),
        _feedback_alignment(state),
        _fairness_and_process_guardrails(task, state),
        bias_subscore,
        llm_subscore,
        _efficiency(task, state),
    ]
    # Keep all reported grading dimensions inside strict (0, 1) as well.
    for idx, item in enumerate(subscores):
        subscores[idx] = GraderSubscore(
            name=item.name,
            score=clamp_open01(item.score, epsilon=1e-2),
            rationale=item.rationale,
        )

    llm_detail = LLMScoreDetail(
        score=clamp_open01(llm_detail.score, epsilon=1e-2),
        rationale=llm_detail.rationale,
        source=llm_detail.source,
    )

    weight = {
        "hard_requirement_compliance": 0.14,
        "shortlist_quality": 0.12,
        "progression_quality": 0.12,
        "final_decision_quality": 0.16,
        "consistency_and_justification": 0.10,
        "feedback_alignment": 0.10,
        "fairness_and_process_guardrails": 0.10,
        "bias_audit": 0.08,
        "llm_decision_quality": 0.05,
        "efficiency": 0.03,
    }

    final_score = 0
    for item in subscores:
        final_score += weight[item.name] * item.score
    # Keep task-level score comfortably inside strict (0, 1), even under coarse rounding.
    final_score = clamp_open01(final_score, epsilon=1e-2)

    summary = (
        f"Task {task.task_id} scored {final_score:.3f} with strongest signal "
        f"from {max(subscores, key=lambda s: s.score).name}."
    )
    return GraderResult(
        task_id=task.task_id,
        final_score=final_score,
        subscores=subscores,
        bias_audit=bias_audit,
        llm_score_detail=llm_detail,
        summary=summary,
    )


def grade_progress(
    task: TaskDefinition, state: EnvironmentState, candidates: Dict[str, CandidateProfile]
) -> float:
    graded = grade_task_state(task, state, candidates)
    if state.final_decision is None:
        return clamp_open01(graded.final_score * 0.85, epsilon=1e-2)
    return clamp_open01(graded.final_score, epsilon=1e-2)
