from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from app.data import load_requisitions
from app.models import Difficulty, HiringConstraintSet, TaskDefinition


def _requisition_map() -> Dict[str, object]:
    return load_requisitions()


@lru_cache(maxsize=1)
def get_task_registry() -> List[TaskDefinition]:
    requisitions = _requisition_map()

    tasks = [
        TaskDefinition(
            task_id="task_easy_screen_backend",
            name="Backend Screening Sprint",
            difficulty=Difficulty.EASY,
            description=(
                "Screen a small backend candidate pool and produce a high-signal shortlist using "
                "hard must-have filters."
            ),
            objective=(
                "Shortlist only candidates who satisfy hard requirements for REQ-BE-001, "
                "advance the strongest one to interview, and finalize a robust hire recommendation."
            ),
            job_requisition=requisitions["REQ-BE-001"],
            candidate_ids=["C001", "C002", "C003", "C004", "C005"],
            constraints=HiringConstraintSet(
                max_shortlist_size=2,
                max_interview_advances=1,
                fairness_guardrails_enabled=True,
                require_reason_for_reject=True,
                disallow_out_of_band_offer=True,
            ),
            max_steps=10,
            expected_shortlist=["C001", "C002"],
            expected_advances=["C001"],
            expected_offer_candidate_id="C001",
            expected_compensation_band="B2",
        ),
        TaskDefinition(
            task_id="task_medium_tradeoff_ml",
            name="ML Tradeoff Routing",
            difficulty=Difficulty.MEDIUM,
            description=(
                "Handle realistic tradeoffs for an applied ML role across budget, notice period, "
                "timezone overlap, and interview feedback."
            ),
            objective=(
                "Build a shortlist under constraints, route the right candidates to interviews, "
                "and choose a final decision aligned to requisition and interview evidence."
            ),
            job_requisition=requisitions["REQ-ML-002"],
            candidate_ids=["C006", "C007", "C008", "C009", "C010"],
            constraints=HiringConstraintSet(
                max_shortlist_size=3,
                max_interview_advances=2,
                fairness_guardrails_enabled=True,
                require_reason_for_reject=True,
                disallow_out_of_band_offer=True,
            ),
            max_steps=14,
            expected_shortlist=["C006", "C007", "C008"],
            expected_advances=["C006", "C007"],
            expected_offer_candidate_id="C006",
            expected_compensation_band="M3",
        ),
        TaskDefinition(
            task_id="task_hard_hiring_manager_e2e",
            name="End-to-End Hiring Manager Decision",
            difficulty=Difficulty.HARD,
            description=(
                "Run a complete hiring-manager workflow with ambiguous signals, interview evidence, "
                "budget limits, and fairness/process checks."
            ),
            objective=(
                "Make an evidence-backed final recommendation for REQ-ENGM-003 while maintaining "
                "policy consistency, guardrails, and process efficiency."
            ),
            job_requisition=requisitions["REQ-ENGM-003"],
            candidate_ids=["C011", "C012", "C013", "C014", "C015"],
            constraints=HiringConstraintSet(
                max_shortlist_size=3,
                max_interview_advances=2,
                fairness_guardrails_enabled=True,
                require_reason_for_reject=True,
                disallow_out_of_band_offer=True,
            ),
            max_steps=16,
            expected_shortlist=["C011", "C012", "C015"],
            expected_advances=["C011", "C012"],
            expected_offer_candidate_id="C011",
            expected_compensation_band="L1",
        ),
    ]

    return tasks

