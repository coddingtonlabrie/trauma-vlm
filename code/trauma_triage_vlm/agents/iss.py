from __future__ import annotations


from trauma_triage_vlm.types import BodyRegion, GradedFinding


def regional_maxima(graded: tuple[GradedFinding, ...]) -> dict[BodyRegion, int]:
    values = {region: 0 for region in BodyRegion}
    for item in graded:
        values[item.finding.region] = max(values[item.finding.region], item.ais)
    return values


def injury_severity_score(graded: tuple[GradedFinding, ...]) -> int:
    if any(item.ais == 6 for item in graded):
        return 75
    maxima = sorted(regional_maxima(graded).values(), reverse=True)
    return int(sum(value * value for value in maxima[:3]))


def partial_scope(graded: tuple[GradedFinding, ...]) -> tuple[str, ...]:
    return tuple(sorted({item.finding.region.value for item in graded if item.ais > 0}))
