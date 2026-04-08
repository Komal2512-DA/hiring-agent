from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from app.models import CandidateProfile, InterviewFeedback, JobRequisition

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json(name: str):
    path = FIXTURE_DIR / name
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_requisitions() -> Dict[str, JobRequisition]:
    raw_items = _load_json("roles.json")
    requisitions: Dict[str, JobRequisition] = {}
    for item in raw_items:
        req = JobRequisition(**item)
        requisitions[req.requisition_id] = req
    return requisitions


def load_candidates() -> Dict[str, CandidateProfile]:
    raw_items = _load_json("candidates.json")
    candidates: Dict[str, CandidateProfile] = {}
    for item in raw_items:
        candidate = CandidateProfile(**item)
        candidates[candidate.candidate_id] = candidate
    return candidates


def load_interview_feedback() -> Dict[str, List[InterviewFeedback]]:
    raw_items = _load_json("interviews.json")
    grouped: Dict[str, List[InterviewFeedback]] = {}
    for item in raw_items:
        feedback = InterviewFeedback(**item)
        grouped.setdefault(feedback.task_id, []).append(feedback)
    return grouped

