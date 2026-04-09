from __future__ import annotations

from typing import Dict, List

from app.models import CandidateProfile, Difficulty, JobRequisition, TaskDefinition
from app.utils import candidate_hard_filter, heuristic_candidate_score


def rank_candidates(candidate_ids: List[str], candidates: Dict[str, CandidateProfile], requisition: JobRequisition) -> List[str]:
    scored = []
    for cid in sorted(candidate_ids):
        candidate = candidates[cid]
        score = heuristic_candidate_score(candidate, requisition)
        hard_pass = candidate_hard_filter(candidate, requisition)
        scored.append((cid, hard_pass, score))

    scored.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [cid for cid, _, _ in scored]


def choose_shortlist(task: TaskDefinition, candidates: Dict[str, CandidateProfile]) -> List[str]:
    ranked = rank_candidates(task.candidate_ids, candidates, task.job_requisition)

    hard_pass_ids = [cid for cid in ranked if candidate_hard_filter(candidates[cid], task.job_requisition)]
    shortlist = hard_pass_ids[: task.constraints.max_shortlist_size]

    if len(shortlist) < task.constraints.max_shortlist_size:
        for cid in ranked:
            if cid not in shortlist:
                shortlist.append(cid)
            if len(shortlist) >= task.constraints.max_shortlist_size:
                break

    return shortlist


def choose_advances(task: TaskDefinition, shortlist: List[str], candidates: Dict[str, CandidateProfile]) -> List[str]:
    ranked = rank_candidates(shortlist, candidates, task.job_requisition)

    if task.difficulty == Difficulty.EASY:
        target = 1
    elif task.difficulty == Difficulty.MEDIUM:
        target = min(2, task.constraints.max_interview_advances)
    else:
        target = min(2, task.constraints.max_interview_advances)

    return ranked[:target]


def choose_offer_candidate(task: TaskDefinition, advances: List[str], candidates: Dict[str, CandidateProfile]) -> str:
    if not advances:
        return ""
    ranked = rank_candidates(advances, candidates, task.job_requisition)
    return ranked[0]


def build_fit_summary(task: TaskDefinition, candidate: CandidateProfile) -> str:
    hard_pass = candidate_hard_filter(candidate, task.job_requisition)
    score = heuristic_candidate_score(candidate, task.job_requisition)
    return (
        f"hard_pass={str(hard_pass).lower()}, fit_score={score:.2f}, "
        f"notice={candidate.notice_period_days}d, expected_comp={candidate.expected_compensation_lpa}LPA"
    )

