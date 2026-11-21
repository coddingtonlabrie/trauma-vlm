from __future__ import annotations


from trauma_triage_vlm.rubric.knowledge import esi_from_grid
from trauma_triage_vlm.types import ClinicalContext, TriageDecision


def shock_index(context: ClinicalContext) -> float:
    if context.heart_rate is None or context.systolic_bp is None or context.systolic_bp <= 0:
        return 0.0
    return float(context.heart_rate / context.systolic_bp)


def expected_resources(context: ClinicalContext) -> int:
    if context.resources_expected is not None:
        return max(0, min(context.resources_expected, 4))
    text = f"{context.chief_complaint} {context.mechanism}".lower()
    count = 0
    for token in ("ct", "trauma", "crash", "fall", "laceration", "assault", "fracture"):
        if token in text:
            count += 1
    return min(count, 4)


def decide_triage(iss: int, context: ClinicalContext, threshold: float) -> TriageDecision:
    shock = shock_index(context)
    resources = expected_resources(context)
    pain = context.pain if context.pain is not None else 5
    esi = esi_from_grid(iss, shock, resources, pain)
    acuity = min(1.0, 0.018 * iss + 0.35 * shock + (0.15 if esi <= 2 else 0.0))
    high = acuity >= threshold or esi <= 2
    confidence = min(0.95, 0.45 + abs(acuity - threshold) + 0.05 * max(0, resources - 1))
    rationale = f"iss={iss};shock={shock:.3f};resources={resources};pain={pain}"
    return TriageDecision(esi=esi, high_acuity_score=acuity, high_acuity=high, confidence=confidence, rationale=rationale)
