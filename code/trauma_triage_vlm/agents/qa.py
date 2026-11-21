from __future__ import annotations


from trauma_triage_vlm.types import ClinicalContext, GradedFinding, TriageDecision


def plausibility_flags(graded: tuple[GradedFinding, ...], decision: TriageDecision, context: ClinicalContext) -> tuple[str, ...]:
    flags: list[str] = []
    bilateral_renal = sum(1 for item in graded if item.finding.organ == "kidney" and item.ais >= 5)
    active_bleeding = any(item.finding.organ == "extravasation" and item.ais >= 3 for item in graded)
    stable_hr = context.heart_rate is not None and context.heart_rate < 90
    stable_bp = context.systolic_bp is not None and context.systolic_bp >= 100
    if bilateral_renal >= 2 and stable_hr and stable_bp:
        flags.append("renal_grade_v_stable_vitals")
    if active_bleeding and stable_hr:
        flags.append("active_extravasation_without_tachycardia")
    if decision.confidence < 0.3:
        flags.append("low_confidence_triage")
    for item in graded:
        if item.finding.present and item.finding.confidence < 0.3:
            flags.append(f"low_confidence_{item.finding.organ}")
    return tuple(flags)


def react_review(flags: tuple[str, ...], max_iterations: int) -> tuple[str, ...]:
    reviewed = list(flags)
    for index in range(max_iterations):
        if not reviewed:
            break
        reviewed = [f"{flag}:review_{index + 1}" for flag in reviewed]
    return tuple(reviewed)
