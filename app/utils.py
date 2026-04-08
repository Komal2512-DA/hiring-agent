from __future__ import annotations

import json
from statistics import mean
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

from app.config import get_settings
from app.models import CandidateProfile, JobRequisition


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_open01(value: float, epsilon: float = 1e-6) -> float:
    """Clamp to a strict open interval (0, 1) for evaluators that disallow boundaries."""
    lo = float(epsilon)
    hi = 1.0 - lo
    return max(lo, min(hi, float(value)))


def timezone_overlap_hours(candidate_tz: int, team_tz: int) -> float:
    delta = abs(candidate_tz - team_tz)
    wrapped_delta = min(delta, 24 - delta)
    return max(0.0, 8.0 - (wrapped_delta * 1.25))


def has_required_skills(candidate: CandidateProfile, required_skills: Iterable[str], min_skill_score: float = 0.60) -> bool:
    for skill in required_skills:
        if candidate.skill_ratings.get(skill, 0.0) < min_skill_score:
            return False
    return True


def candidate_hard_filter(candidate: CandidateProfile, requisition: JobRequisition) -> bool:
    if not has_required_skills(candidate, requisition.required_skills):
        return False
    if candidate.years_experience < requisition.min_years_experience:
        return False
    if candidate.expected_compensation_lpa > requisition.max_compensation_lpa:
        return False
    if candidate.notice_period_days > requisition.max_notice_period_days:
        return False
    if candidate.communication_score < requisition.min_communication_score:
        return False
    if candidate.leadership_score < requisition.min_leadership_score:
        return False
    if timezone_overlap_hours(candidate.timezone_offset_hours, requisition.team_timezone_offset_hours) < requisition.min_timezone_overlap_hours:
        return False
    return True


def skill_match_ratio(candidate: CandidateProfile, requisition: JobRequisition) -> float:
    required = requisition.required_skills
    preferred = requisition.preferred_skills
    if not required:
        return 1.0

    req_scores = [candidate.skill_ratings.get(skill, 0.0) for skill in required]
    pref_scores = [candidate.skill_ratings.get(skill, 0.0) for skill in preferred] or [0.0]

    req_component = sum(req_scores) / len(req_scores)
    pref_component = sum(pref_scores) / len(pref_scores)
    return clamp01((0.8 * req_component) + (0.2 * pref_component))


def compensation_alignment(candidate: CandidateProfile, requisition: JobRequisition) -> float:
    if candidate.expected_compensation_lpa <= requisition.max_compensation_lpa:
        return 1.0
    overflow = candidate.expected_compensation_lpa - requisition.max_compensation_lpa
    return clamp01(1.0 - (overflow / max(10.0, requisition.max_compensation_lpa)))


def notice_alignment(candidate: CandidateProfile, requisition: JobRequisition) -> float:
    if candidate.notice_period_days <= requisition.max_notice_period_days:
        return 1.0
    overflow = candidate.notice_period_days - requisition.max_notice_period_days
    return clamp01(1.0 - (overflow / 120.0))


def heuristic_candidate_score(candidate: CandidateProfile, requisition: JobRequisition) -> float:
    skills = skill_match_ratio(candidate, requisition)
    experience = clamp01(candidate.years_experience / max(1.0, requisition.min_years_experience + 5.0))
    timezone = clamp01(timezone_overlap_hours(candidate.timezone_offset_hours, requisition.team_timezone_offset_hours) / 8.0)
    comp = compensation_alignment(candidate, requisition)
    notice = notice_alignment(candidate, requisition)

    quality = (
        (0.36 * skills)
        + (0.10 * experience)
        + (0.10 * timezone)
        + (0.10 * comp)
        + (0.10 * notice)
        + (0.12 * candidate.communication_score)
        + (0.12 * candidate.leadership_score)
    )
    return clamp01(quality)


def compact_json(payload: Dict[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def location_diversity_score(candidate_ids: List[str], candidates: Dict[str, CandidateProfile]) -> float:
    if not candidate_ids:
        return 0.0
    locations = {candidates[cid].location.lower() for cid in candidate_ids if cid in candidates}
    return clamp01(len(locations) / max(1.0, len(candidate_ids)))


def mean_feedback_score(feedback_rows: List[Dict[str, float]]) -> float:
    if not feedback_rows:
        return 0.5
    numeric = []
    for row in feedback_rows:
        technical = float(row.get("technical_score", 0.0))
        communication = float(row.get("communication_score", 0.0))
        numeric.append(clamp01((0.7 * technical) + (0.3 * communication)))
    return clamp01(mean(numeric))


class OpenAIJustificationHelper:
    """Optional LLM helper for concise hiring rationale generation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.use_llm_justification and bool(settings.api_base_url) and bool(settings.api_key)
        self.model_name = settings.model_name
        self._base_url = settings.api_base_url
        self._api_key = settings.api_key
        self.client: Optional[OpenAI] = None

        if self.enabled:
            self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def generate_or_fallback(self, deterministic_summary: str, context: Dict[str, object]) -> str:
        if not self.enabled or self.client is None:
            return deterministic_summary

        prompt = (
            "You are a hiring assistant. Produce a one-sentence evidence-based hiring rationale. "
            "Use compact, professional wording. Context:\n"
            f"{compact_json(context)}"
        )

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0,
                max_output_tokens=120,
            )
            text = getattr(response, "output_text", "").strip()
            if text:
                return text
        except Exception:
            pass

        return deterministic_summary


def f1_overlap(expected: List[str], actual: List[str]) -> float:
    exp_set = set(expected)
    act_set = set(actual)
    if not exp_set and not act_set:
        return 1.0
    if not exp_set or not act_set:
        return 0.0

    tp = len(exp_set.intersection(act_set))
    precision = tp / len(act_set)
    recall = tp / len(exp_set)
    if precision + recall == 0:
        return 0.0
    return clamp01((2.0 * precision * recall) / (precision + recall))
