from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum


class BodyRegion(str, Enum):
    head_neck = "head_neck"
    face = "face"
    thorax = "thorax"
    abdomen = "abdomen"
    extremity = "extremity"
    external = "external"


@dataclass(frozen=True)
class ClinicalContext:
    chief_complaint: str
    mechanism: str
    heart_rate: float | None = None
    systolic_bp: float | None = None
    gcs: int | None = None
    pain: int | None = None
    resources_expected: int | None = None


@dataclass(frozen=True)
class ImageVolume:
    array: object
    spacing_mm: tuple[float, float, float]
    modality: str
    series_uid: str


@dataclass(frozen=True)
class Finding:
    organ: str
    region: BodyRegion
    present: bool
    confidence: float
    descriptors: tuple[str, ...] = field(default_factory=tuple)
    slices: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GradedFinding:
    finding: Finding
    ais: int
    aast: int
    rationale: str


@dataclass(frozen=True)
class TriageDecision:
    esi: int
    high_acuity_score: float
    high_acuity: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class PipelineReport:
    findings: tuple[Finding, ...]
    graded: tuple[GradedFinding, ...]
    iss: int
    decision: TriageDecision
    flags: tuple[str, ...]
