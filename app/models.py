from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PipelineStage(str, Enum):
    APPLIED = "applied"
    SCREENED = "screened"
    SHORTLISTED = "shortlisted"
    INTERVIEW = "interview"
    FINAL_REVIEW = "final_review"
    OFFER = "offer"
    HOLD = "hold"
    REJECTED = "rejected"


class ActionType(str, Enum):
    SCREEN_CANDIDATE = "screen_candidate"
    SHORTLIST_CANDIDATES = "shortlist_candidates"
    ADVANCE_STAGE = "advance_stage"
    ASSIGN_INTERVIEWER = "assign_interviewer"
    REQUEST_INTERVIEW = "request_interview"
    REJECT_CANDIDATE = "reject_candidate"
    HOLD_CANDIDATE = "hold_candidate"
    CHOOSE_COMPENSATION_BAND = "choose_compensation_band"
    SUMMARIZE_FIT = "summarize_fit"
    FINALIZE_DECISION = "finalize_decision"


class CompensationBand(BaseModel):
    name: str
    min_lpa: float
    max_lpa: float
    currency: str = "INR"


class JobRequisition(BaseModel):
    requisition_id: str
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str] = Field(default_factory=list)
    min_years_experience: float
    max_compensation_lpa: float
    team_timezone_offset_hours: int
    min_timezone_overlap_hours: float
    max_notice_period_days: int
    min_communication_score: float
    min_leadership_score: float
    remote_allowed: bool
    hiring_urgency: str
    budget_band: CompensationBand


class HiringConstraintSet(BaseModel):
    max_shortlist_size: int
    max_interview_advances: int
    fairness_guardrails_enabled: bool = True
    require_reason_for_reject: bool = True
    disallow_out_of_band_offer: bool = True


class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    current_title: str
    location: str
    timezone_offset_hours: int
    years_experience: float
    notice_period_days: int
    expected_compensation_lpa: float
    skill_ratings: Dict[str, float]
    communication_score: float
    leadership_score: float
    domain_score: float
    gender: Optional[str] = None
    underrepresented_group: Optional[bool] = None


class InterviewFeedback(BaseModel):
    feedback_id: str
    task_id: str
    candidate_id: str
    interviewer_id: str
    technical_score: float
    communication_score: float
    recommendation: str
    notes: str


class HiringDecision(BaseModel):
    offer_candidate_id: Optional[str] = None
    compensation_band: Optional[str] = None
    reject_candidate_ids: List[str] = Field(default_factory=list)
    hold_candidate_ids: List[str] = Field(default_factory=list)
    justification: str = ""


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    difficulty: Difficulty
    description: str
    objective: str
    job_requisition: JobRequisition
    candidate_ids: List[str]
    constraints: HiringConstraintSet
    max_steps: int

    expected_shortlist: List[str]
    expected_advances: List[str]
    expected_offer_candidate_id: str
    expected_compensation_band: str


class Action(BaseModel):
    action_type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)


class ActionRecord(BaseModel):
    step_index: int
    action: Action


class ObservationCandidateView(BaseModel):
    candidate_id: str
    stage: PipelineStage
    weighted_fit_score: float
    summary: str


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: Difficulty
    objective: str
    step_index: int
    done: bool
    pipeline_overview: Dict[str, str]
    candidate_views: List[ObservationCandidateView]
    current_progress_score: float
    message: str


class BiasMetric(BaseModel):
    metric_name: str
    group_rates: Dict[str, float]
    disparity: float
    threshold: float
    passed: bool
    rationale: str


class BiasAuditResult(BaseModel):
    audited_dimension: str
    sample_size: int
    adverse_impact_ratio: float
    representation_parity_score: float
    advancement_parity_score: float
    overall_score: float
    passed: bool
    flagged_metrics: List[str] = Field(default_factory=list)
    metrics: List[BiasMetric] = Field(default_factory=list)
    summary: str


class LLMScoreDetail(BaseModel):
    score: float
    rationale: str
    source: str


class EnvironmentState(BaseModel):
    task: TaskDefinition
    candidates: Dict[str, CandidateProfile]
    stages: Dict[str, PipelineStage]
    shortlist: List[str] = Field(default_factory=list)
    interview_advances: List[str] = Field(default_factory=list)
    interviewer_assignments: Dict[str, List[str]] = Field(default_factory=dict)
    fit_summaries: Dict[str, str] = Field(default_factory=dict)
    chosen_compensation_band: Optional[str] = None
    final_decision: Optional[HiringDecision] = None
    feedback: Dict[str, List[InterviewFeedback]] = Field(default_factory=dict)
    action_history: List[ActionRecord] = Field(default_factory=list)
    bias_audit: Optional[BiasAuditResult] = None
    step_index: int = 0
    done: bool = False
    previous_progress_score: float = 0.0


class GraderSubscore(BaseModel):
    name: str
    score: float
    rationale: str


class GraderResult(BaseModel):
    task_id: str
    final_score: float
    subscores: List[GraderSubscore]
    bias_audit: Optional[BiasAuditResult] = None
    llm_score_detail: Optional[LLMScoreDetail] = None
    summary: str


class RewardOutput(BaseModel):
    step_reward: float
    progress_score: float
    final_score: Optional[float] = None


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: RewardOutput


class ResetResponse(BaseModel):
    observation: Observation
    state: EnvironmentState
