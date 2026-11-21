from __future__ import annotations


from typing import cast

from trauma_triage_vlm.rubric.generated_scale_bank import ESI_RULE_GRID, SCALE_BANK
from trauma_triage_vlm.types import BodyRegion, Finding, GradedFinding


REGION_BY_ORGAN: dict[str, BodyRegion] = {
    "liver": BodyRegion.abdomen,
    "spleen": BodyRegion.abdomen,
    "kidney": BodyRegion.abdomen,
    "bowel_mesenteric": BodyRegion.abdomen,
    "extravasation": BodyRegion.abdomen,
    "ich": BodyRegion.head_neck,
    "calvarial_fracture": BodyRegion.head_neck,
}


def grade_from_descriptors(organ: str, descriptors: tuple[str, ...], confidence: float) -> int:
    grade = 0
    for entry in SCALE_BANK.values():
        if entry["organ"] != organ:
            continue
        signal = cast(str, entry["signal"])
        lower = cast(float, entry["lower"])
        if signal in descriptors and confidence >= lower:
            grade = max(grade, cast(int, entry["grade"]))
    return min(grade, 6)


def grade_finding(finding: Finding) -> GradedFinding:
    ais = grade_from_descriptors(finding.organ, finding.descriptors, finding.confidence)
    aast = min(ais, 5)
    rationale = f"{finding.organ}:{ais}:{','.join(finding.descriptors)}"
    return GradedFinding(finding=finding, ais=ais, aast=aast, rationale=rationale)


def esi_from_grid(iss: int, shock: float, resources: int, pain: int) -> int:
    best = 5
    distance = float("inf")
    for row in ESI_RULE_GRID:
        current = abs(float(row["iss"]) - iss) + abs(float(row["shock"]) - shock) * 10.0 + abs(int(row["resources"]) - resources) + abs(int(row["pain"]) - pain) / 3.0
        if current < distance:
            distance = current
            best = int(row["esi"])
    return best
