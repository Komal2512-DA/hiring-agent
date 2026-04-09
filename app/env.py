from __future__ import annotations

import random
from typing import Dict, List, Optional

from app.bias_auditor import run_bias_audit
from app.data import load_candidates, load_interview_feedback
from app.graders import grade_progress, grade_task_state
from app.models import (
    Action,
    ActionRecord,
    ActionType,
    CandidateProfile,
    EnvironmentState,
    HiringDecision,
    Observation,
    ObservationCandidateView,
    PipelineStage,
    RewardOutput,
    TaskDefinition,
)
from app.policy import build_fit_summary
from app.tasks import get_task_registry
from app.utils import SCORE_MIN, clamp_open01, heuristic_candidate_score


class HiringOpenEnv:
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

        self._all_candidates = load_candidates()
        self._all_feedback = load_interview_feedback()
        self._tasks = get_task_registry()

        self._state: Optional[EnvironmentState] = None

    def list_tasks(self) -> List[TaskDefinition]:
        return [task.model_copy(deep=True) for task in self._tasks]

    def reset(self, task_id: Optional[str] = None) -> Observation:
        task = self._resolve_task(task_id)
        candidates = {cid: self._all_candidates[cid].model_copy(deep=True) for cid in task.candidate_ids}
        stages = {cid: PipelineStage.APPLIED for cid in task.candidate_ids}

        task_feedback = self._all_feedback.get(task.task_id, [])
        feedback_map: Dict[str, List[object]] = {}
        for row in task_feedback:
            feedback_map.setdefault(row.candidate_id, []).append(row)

        self._state = EnvironmentState(
            task=task.model_copy(deep=True),
            candidates=candidates,
            stages=stages,
            shortlist=[],
            interview_advances=[],
            interviewer_assignments={},
            fit_summaries={},
            chosen_compensation_band=None,
            final_decision=None,
            feedback=feedback_map,
            action_history=[],
            bias_audit=None,
            step_index=0,
            done=False,
            previous_progress_score=clamp_open01(SCORE_MIN, epsilon=SCORE_MIN),
        )
        self._state.bias_audit = run_bias_audit(self._state.task, self._state, self._state.candidates)
        return self._build_observation("Environment reset. Ready for hiring actions.")

    def state(self) -> EnvironmentState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> tuple[Observation, RewardOutput]:
        if self._state is None:
            self.reset()
        assert self._state is not None

        if self._state.done:
            final = clamp_open01(
                grade_task_state(self._state.task, self._state, self._state.candidates).final_score,
                epsilon=SCORE_MIN,
            )
            observation = self._build_observation("Task already completed.")
            reward = RewardOutput(
                step_reward=round(clamp_open01(SCORE_MIN, epsilon=SCORE_MIN), 2),
                progress_score=round(clamp_open01(final, epsilon=SCORE_MIN), 2),
                final_score=round(clamp_open01(final, epsilon=SCORE_MIN), 2),
            )
            return observation, reward

        previous_progress = self._state.previous_progress_score
        message = self._apply_action(action)

        self._state.step_index += 1
        self._state.action_history.append(ActionRecord(step_index=self._state.step_index, action=action))
        self._state.bias_audit = run_bias_audit(self._state.task, self._state, self._state.candidates)

        if self._state.step_index >= self._state.task.max_steps and not self._state.done:
            if self._state.final_decision is None:
                self._state.final_decision = HiringDecision(
                    offer_candidate_id="",
                    compensation_band=self._state.chosen_compensation_band or self._state.task.expected_compensation_band,
                    reject_candidate_ids=[],
                    hold_candidate_ids=[],
                    justification="Auto-finalized due to max-step limit.",
                )
            self._state.done = True
            message = f"{message} Max steps reached; task auto-finalized."

        progress_score = clamp_open01(
            grade_progress(self._state.task, self._state, self._state.candidates),
            epsilon=SCORE_MIN,
        )
        final_score: Optional[float] = None
        if self._state.done:
            final_score = clamp_open01(
                grade_task_state(self._state.task, self._state, self._state.candidates).final_score,
                epsilon=SCORE_MIN,
            )
            progress_score = final_score

        step_reward = clamp_open01(progress_score - previous_progress, epsilon=SCORE_MIN)
        self._state.previous_progress_score = clamp_open01(progress_score, epsilon=SCORE_MIN)

        observation = self._build_observation(message)
        reward = RewardOutput(
            step_reward=round(step_reward, 2),
            progress_score=round(clamp_open01(progress_score, epsilon=SCORE_MIN), 2),
            final_score=round(clamp_open01(final_score, epsilon=SCORE_MIN), 2) if final_score is not None else None,
        )
        return observation, reward

    def _resolve_task(self, task_id: Optional[str]) -> TaskDefinition:
        if task_id is None:
            return self._tasks[0]
        for task in self._tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Unknown task_id: {task_id}")

    def _validate_candidate_id(self, cid: str) -> CandidateProfile:
        assert self._state is not None
        if cid not in self._state.candidates:
            raise ValueError(f"Candidate {cid} is not part of current task.")
        return self._state.candidates[cid]

    def _normalize_candidate_ids(self, candidate_ids: List[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for cid in candidate_ids:
            if cid in seen:
                continue
            self._validate_candidate_id(cid)
            seen.add(cid)
            unique.append(cid)
        return unique

    def _apply_action(self, action: Action) -> str:
        assert self._state is not None
        payload = action.payload

        if action.action_type == ActionType.SCREEN_CANDIDATE:
            cid = str(payload.get("candidate_id", "")).strip()
            decision = str(payload.get("decision", "screened")).strip().lower()
            self._validate_candidate_id(cid)

            if decision == "reject":
                self._state.stages[cid] = PipelineStage.REJECTED
            elif decision == "hold":
                self._state.stages[cid] = PipelineStage.HOLD
            elif decision in {"advance", "screened"}:
                self._state.stages[cid] = PipelineStage.SCREENED
            else:
                raise ValueError("Unsupported screening decision.")
            return f"Candidate {cid} screened with decision={decision}."

        if action.action_type == ActionType.SHORTLIST_CANDIDATES:
            raw_ids = payload.get("candidate_ids", [])
            if not isinstance(raw_ids, list):
                raise ValueError("candidate_ids must be a list.")

            shortlist = self._normalize_candidate_ids(raw_ids)
            shortlist = shortlist[: self._state.task.constraints.max_shortlist_size]
            self._state.shortlist = shortlist
            for cid in shortlist:
                self._state.stages[cid] = PipelineStage.SHORTLISTED
            return f"Shortlist updated with {len(shortlist)} candidate(s)."

        if action.action_type == ActionType.ADVANCE_STAGE:
            raw_ids = payload.get("candidate_ids", [])
            stage_name = str(payload.get("stage", "interview")).strip().lower()
            if not isinstance(raw_ids, list):
                raise ValueError("candidate_ids must be a list.")

            target_stage = PipelineStage(stage_name)
            ids = self._normalize_candidate_ids(raw_ids)
            for cid in ids:
                self._state.stages[cid] = target_stage
                if target_stage == PipelineStage.INTERVIEW and cid not in self._state.interview_advances:
                    self._state.interview_advances.append(cid)
            if len(self._state.interview_advances) > self._state.task.constraints.max_interview_advances:
                self._state.interview_advances = self._state.interview_advances[
                    : self._state.task.constraints.max_interview_advances
                ]
            return f"Advanced {len(ids)} candidate(s) to stage={target_stage.value}."

        if action.action_type == ActionType.REQUEST_INTERVIEW:
            cid = str(payload.get("candidate_id", "")).strip()
            self._validate_candidate_id(cid)
            self._state.stages[cid] = PipelineStage.INTERVIEW
            if cid not in self._state.interview_advances:
                self._state.interview_advances.append(cid)
            return f"Interview requested for candidate {cid}."

        if action.action_type == ActionType.ASSIGN_INTERVIEWER:
            cid = str(payload.get("candidate_id", "")).strip()
            interviewer = str(payload.get("interviewer_id", "")).strip()
            self._validate_candidate_id(cid)
            if not interviewer:
                raise ValueError("interviewer_id is required.")
            assigned = self._state.interviewer_assignments.setdefault(cid, [])
            if interviewer not in assigned:
                assigned.append(interviewer)
            self._state.stages[cid] = PipelineStage.INTERVIEW
            if cid not in self._state.interview_advances:
                self._state.interview_advances.append(cid)
            return f"Assigned interviewer {interviewer} to candidate {cid}."

        if action.action_type == ActionType.REJECT_CANDIDATE:
            cid = str(payload.get("candidate_id", "")).strip()
            self._validate_candidate_id(cid)
            self._state.stages[cid] = PipelineStage.REJECTED
            return f"Candidate {cid} marked rejected."

        if action.action_type == ActionType.HOLD_CANDIDATE:
            cid = str(payload.get("candidate_id", "")).strip()
            self._validate_candidate_id(cid)
            self._state.stages[cid] = PipelineStage.HOLD
            return f"Candidate {cid} marked hold."

        if action.action_type == ActionType.CHOOSE_COMPENSATION_BAND:
            band = str(payload.get("compensation_band", "")).strip()
            if not band:
                raise ValueError("compensation_band is required.")
            self._state.chosen_compensation_band = band
            return f"Compensation band selected as {band}."

        if action.action_type == ActionType.SUMMARIZE_FIT:
            cid = str(payload.get("candidate_id", "")).strip()
            self._validate_candidate_id(cid)
            summary = str(payload.get("summary", "")).strip()
            if not summary:
                summary = build_fit_summary(self._state.task, self._state.candidates[cid])
            self._state.fit_summaries[cid] = summary
            return f"Fit summary captured for candidate {cid}."

        if action.action_type == ActionType.FINALIZE_DECISION:
            decision = HiringDecision(**payload)
            offered = (decision.offer_candidate_id or "").strip()
            rejected = [cid for cid in decision.reject_candidate_ids if cid]
            held = [cid for cid in decision.hold_candidate_ids if cid]

            if offered:
                self._validate_candidate_id(offered)
                rejected = [cid for cid in rejected if cid != offered]
                held = [cid for cid in held if cid != offered]

            decision.reject_candidate_ids = self._normalize_candidate_ids(rejected)
            decision.hold_candidate_ids = self._normalize_candidate_ids(held)
            decision.compensation_band = (
                decision.compensation_band
                or self._state.chosen_compensation_band
                or self._state.task.expected_compensation_band
            )

            self._state.final_decision = decision
            self._state.done = True

            if offered:
                self._state.stages[offered] = PipelineStage.OFFER
            for cid in decision.reject_candidate_ids:
                self._state.stages[cid] = PipelineStage.REJECTED
            for cid in decision.hold_candidate_ids:
                self._state.stages[cid] = PipelineStage.HOLD
            return "Final hiring decision submitted."

        raise ValueError(f"Unsupported action type: {action.action_type.value}")

    def _build_observation(self, message: str) -> Observation:
        assert self._state is not None
        task = self._state.task
        views: List[ObservationCandidateView] = []
        for cid in task.candidate_ids:
            profile = self._state.candidates[cid]
            score = heuristic_candidate_score(profile, task.job_requisition)
            summary = self._state.fit_summaries.get(cid)
            if not summary:
                summary = (
                    f"exp={profile.years_experience:.2f}, notice={profile.notice_period_days}d, "
                    f"comp={profile.expected_compensation_lpa:.2f}"
                )
            views.append(
                ObservationCandidateView(
                    candidate_id=cid,
                    stage=self._state.stages[cid],
                    weighted_fit_score=round(score, 2),
                    summary=summary,
                )
            )

        pipeline_overview = {cid: self._state.stages[cid].value for cid in task.candidate_ids}
        return Observation(
            task_id=task.task_id,
            task_name=task.name,
            difficulty=task.difficulty,
            objective=task.objective,
            step_index=self._state.step_index,
            done=self._state.done,
            pipeline_overview=pipeline_overview,
            candidate_views=views,
            current_progress_score=round(clamp_open01(self._state.previous_progress_score, epsilon=SCORE_MIN), 2),
            message=message,
        )
