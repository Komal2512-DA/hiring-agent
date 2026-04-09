from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from app.models import BiasAuditResult, BiasMetric, CandidateProfile, EnvironmentState, TaskDefinition
from app.utils import candidate_hard_filter, clamp01, clamp_open01


def _normalize_gender(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    mapping = {
        "m": "male",
        "man": "male",
        "male": "male",
        "f": "female",
        "woman": "female",
        "female": "female",
        "nb": "non_binary",
        "non-binary": "non_binary",
        "non_binary": "non_binary",
    }
    return mapping.get(cleaned, cleaned)


def _location_bucket(location: str) -> str:
    loc = location.strip().lower()
    if not loc:
        return "unknown"
    return loc


def _group_mapping(task: TaskDefinition, candidates: Dict[str, CandidateProfile]) -> Tuple[str, Dict[str, str]]:
    selected = {cid: candidates[cid] for cid in task.candidate_ids if cid in candidates}

    urg_values = [c.underrepresented_group for c in selected.values()]
    if urg_values and all(v is not None for v in urg_values):
        unique = {bool(v) for v in urg_values}
        if len(unique) >= 2:
            mapping = {cid: ("underrepresented" if selected[cid].underrepresented_group else "non_underrepresented") for cid in selected}
            return "underrepresented_group", mapping

    gender_values = [_normalize_gender(c.gender) for c in selected.values()]
    gender_non_null = [g for g in gender_values if g is not None]
    if len(set(gender_non_null)) >= 2:
        mapping = {cid: _normalize_gender(selected[cid].gender) or "unknown" for cid in selected}
        return "gender", mapping

    mapping = {cid: _location_bucket(selected[cid].location) for cid in selected}
    return "location", mapping


def _group_rate(
    population: Iterable[str],
    selected: Iterable[str],
    group_map: Dict[str, str],
) -> Dict[str, float]:
    pop_counts: Dict[str, int] = {}
    sel_counts: Dict[str, int] = {}

    pop_list = [cid for cid in population if cid in group_map]
    sel_set = {cid for cid in selected if cid in group_map}

    for cid in pop_list:
        g = group_map[cid]
        pop_counts[g] = pop_counts.get(g, 0) + 1
    for cid in sel_set:
        g = group_map[cid]
        sel_counts[g] = sel_counts.get(g, 0) + 1

    rates: Dict[str, float] = {}
    for g, count in pop_counts.items():
        rates[g] = 0.0 if count == 0 else sel_counts.get(g, 0) / float(count)
    return rates


def _disparity_from_rates(rates: Dict[str, float]) -> float:
    if not rates:
        return 0.0
    values = list(rates.values())
    return max(values) - min(values)


def _adverse_impact_ratio(rates: Dict[str, float]) -> float:
    if not rates:
        return 1.0
    positives = [v for v in rates.values() if v > 0]
    if not positives:
        return 1.0
    lo = min(positives)
    hi = max(positives)
    if hi <= 0:
        return 1.0
    return clamp01(lo / hi)


def run_bias_audit(
    task: TaskDefinition,
    state: EnvironmentState,
    candidates: Dict[str, CandidateProfile],
    parity_threshold: float = 0.20,
    adverse_impact_threshold: float = 0.80,
) -> BiasAuditResult:
    dimension, group_map = _group_mapping(task, candidates)
    pool_ids = [cid for cid in task.candidate_ids if cid in group_map]

    shortlist_rates = _group_rate(pool_ids, state.shortlist, group_map)
    shortlist_disparity = _disparity_from_rates(shortlist_rates)
    shortlist_parity_score = clamp_open01(1.0 - shortlist_disparity, epsilon=1e-2)

    eligible_ids = [
        cid
        for cid in task.candidate_ids
        if cid in candidates and cid in group_map and candidate_hard_filter(candidates[cid], task.job_requisition)
    ]
    advance_population = eligible_ids if eligible_ids else pool_ids
    advance_rates = _group_rate(advance_population, state.interview_advances, group_map)
    advance_disparity = _disparity_from_rates(advance_rates)
    advance_parity_score = clamp_open01(1.0 - advance_disparity, epsilon=1e-2)

    air_shortlist = _adverse_impact_ratio(shortlist_rates)
    air_advance = _adverse_impact_ratio(advance_rates)
    air_min = clamp_open01(min(air_shortlist, air_advance), epsilon=1e-2)

    metrics: List[BiasMetric] = [
        BiasMetric(
            metric_name="shortlist_parity",
            group_rates={k: round(v, 4) for k, v in shortlist_rates.items()},
            disparity=round(shortlist_disparity, 4),
            threshold=parity_threshold,
            passed=shortlist_disparity <= parity_threshold,
            rationale="Difference between highest and lowest shortlist rates by audited group.",
        ),
        BiasMetric(
            metric_name="advance_parity",
            group_rates={k: round(v, 4) for k, v in advance_rates.items()},
            disparity=round(advance_disparity, 4),
            threshold=parity_threshold,
            passed=advance_disparity <= parity_threshold,
            rationale="Difference between highest and lowest interview-advance rates by audited group.",
        ),
        BiasMetric(
            metric_name="adverse_impact_ratio",
            group_rates={
                "shortlist_air": round(air_shortlist, 4),
                "advance_air": round(air_advance, 4),
                "minimum_air": round(air_min, 4),
            },
            disparity=round(1.0 - air_min, 4),
            threshold=adverse_impact_threshold,
            passed=air_min >= adverse_impact_threshold,
            rationale="Four-fifths style ratio check using group selection rates.",
        ),
    ]

    flagged = [m.metric_name for m in metrics if not m.passed]
    overall = clamp_open01((0.40 * shortlist_parity_score) + (0.35 * advance_parity_score) + (0.25 * air_min), epsilon=1e-2)
    passed = len(flagged) == 0

    summary = (
        f"Bias audit on {dimension}: overall={overall:.3f}, "
        f"shortlist_parity={shortlist_parity_score:.3f}, "
        f"advance_parity={advance_parity_score:.3f}, "
        f"min_air={air_min:.3f}, flagged={flagged or ['none']}."
    )

    return BiasAuditResult(
        audited_dimension=dimension,
        sample_size=len(pool_ids),
        adverse_impact_ratio=round(air_min, 6),
        representation_parity_score=round(shortlist_parity_score, 6),
        advancement_parity_score=round(advance_parity_score, 6),
        overall_score=round(overall, 6),
        passed=passed,
        flagged_metrics=flagged,
        metrics=metrics,
        summary=summary,
    )
