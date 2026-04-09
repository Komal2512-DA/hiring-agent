from __future__ import annotations

import json
import re
from typing import Dict, Optional

from openai import OpenAI

from app.config import get_settings
from app.models import CandidateProfile, EnvironmentState, LLMScoreDetail, TaskDefinition
from app.utils import clamp01, clamp_open01


def _extract_json_object(text: str) -> Optional[Dict[str, object]]:
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


class DecisionLLMScorer:
    """Optional LLM-based decision quality scorer with deterministic fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.use_llm_scoring and bool(settings.api_base_url) and bool(settings.api_key)
        self.model_name = settings.model_name
        self._base_url = settings.api_base_url
        self._api_key = settings.api_key
        self.client: Optional[OpenAI] = None

        if self.enabled:
            self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def _fallback(self, task: TaskDefinition, state: EnvironmentState, candidates: Dict[str, CandidateProfile], source: str) -> LLMScoreDetail:
        decision = state.final_decision
        if decision is None:
            return LLMScoreDetail(
                score=clamp_open01(0.0, epsilon=1e-2),
                rationale="No final decision for LLM scoring.",
                source=source,
            )

        offer = decision.offer_candidate_id or ""
        offer_match = 1.0 if offer == task.expected_offer_candidate_id else 0.0
        band = decision.compensation_band or state.chosen_compensation_band or ""
        band_match = 1.0 if band == task.expected_compensation_band else 0.0
        justification = (decision.justification or "").strip()
        justification_score = clamp01(len(justification) / 100.0)
        evidence_score = clamp01(len(state.fit_summaries) / max(1.0, float(len(state.interview_advances) or 1)))

        score = clamp01((0.35 * offer_match) + (0.25 * band_match) + (0.25 * justification_score) + (0.15 * evidence_score))
        rationale = (
            f"fallback_scoring offer_match={offer_match:.2f}, band_match={band_match:.2f}, "
            f"justification={justification_score:.2f}, evidence={evidence_score:.2f}"
        )
        return LLMScoreDetail(score=round(clamp_open01(score, epsilon=1e-2), 6), rationale=rationale, source=source)

    def score(self, task: TaskDefinition, state: EnvironmentState, candidates: Dict[str, CandidateProfile]) -> LLMScoreDetail:
        decision = state.final_decision
        if decision is None:
            return LLMScoreDetail(
                score=clamp_open01(0.0, epsilon=1e-2),
                rationale="No final decision for LLM scoring.",
                source="disabled",
            )

        if not self.enabled or self.client is None:
            return self._fallback(task, state, candidates, source="disabled")

        offer = decision.offer_candidate_id or ""
        candidate = candidates.get(offer)
        payload = {
            "task_id": task.task_id,
            "objective": task.objective,
            "required_skills": task.job_requisition.required_skills,
            "preferred_skills": task.job_requisition.preferred_skills,
            "offer_candidate_id": offer,
            "offer_candidate_profile": candidate.model_dump(mode="json") if candidate else {},
            "shortlist": state.shortlist,
            "interview_advances": state.interview_advances,
            "chosen_compensation_band": decision.compensation_band or state.chosen_compensation_band or "",
            "justification": decision.justification,
            "fit_summaries": state.fit_summaries,
            "feedback_count": {cid: len(rows) for cid, rows in state.feedback.items()},
        }

        prompt = (
            "You are evaluating hiring decision quality. "
            "Score from 0.0 to 1.0 based on requirement fit, evidence quality, policy consistency, "
            "and compensation alignment. Return strict JSON only with keys: "
            "{\"score\": <float_0_to_1>, \"rationale\": \"short_reason\"}. "
            f"Context: {json.dumps(payload, separators=(',', ':'))}"
        )

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0,
                max_output_tokens=220,
            )
            text = getattr(response, "output_text", "").strip()
            parsed = _extract_json_object(text)
            if parsed:
                score_val = clamp01(float(parsed.get("score", 0.0)))
                rationale = str(parsed.get("rationale", "llm_scored"))
                return LLMScoreDetail(
                    score=round(clamp_open01(score_val, epsilon=1e-2), 6),
                    rationale=rationale,
                    source="llm",
                )
        except Exception:
            pass

        return self._fallback(task, state, candidates, source="fallback")
