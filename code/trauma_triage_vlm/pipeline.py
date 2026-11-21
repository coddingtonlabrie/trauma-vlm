from __future__ import annotations


import numpy as np

from trauma_triage_vlm.agents.detection import detect_all
from trauma_triage_vlm.agents.iss import injury_severity_score
from trauma_triage_vlm.agents.qa import plausibility_flags, react_review
from trauma_triage_vlm.agents.triage import decide_triage
from trauma_triage_vlm.config import ProjectConfig
from trauma_triage_vlm.imaging.slices import landmark_slices
from trauma_triage_vlm.imaging.windowing import normalize_volume, render_windows
from trauma_triage_vlm.rubric.knowledge import grade_finding
from trauma_triage_vlm.types import ClinicalContext, PipelineReport


DEFAULT_ORGANS: tuple[str, ...] = ("liver", "spleen", "kidney", "bowel_mesenteric", "extravasation")
DEFAULT_WINDOWS: tuple[tuple[float, float], ...] = ((400.0, 50.0), (150.0, 30.0), (1800.0, 400.0))


def prepare_rendered(volume: np.ndarray, config: ProjectConfig, organs: tuple[str, ...]) -> dict[str, tuple[np.ndarray, tuple[int, ...]]]:
    normalized = normalize_volume(volume)
    rendered: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {}
    for organ in organs:
        source = "liver" if organ in {"bowel_mesenteric", "extravasation"} else organ
        slices = landmark_slices(
            normalized,
            source,
            config.imaging.min_component_area,
            config.imaging.target_slices_min,
            config.imaging.target_slices_max,
        )
        stack = np.stack([render_windows(normalized[idx], DEFAULT_WINDOWS) for idx in slices], axis=0)
        rendered[organ] = (stack, slices)
    return rendered


def run_visual_pipeline(volume: np.ndarray, context: ClinicalContext, config: ProjectConfig, threshold: float = 0.5) -> PipelineReport:
    rendered = prepare_rendered(volume, config, DEFAULT_ORGANS)
    findings = detect_all(rendered)
    graded = tuple(grade_finding(finding) for finding in findings)
    iss = injury_severity_score(graded)
    decision = decide_triage(iss, context, threshold)
    flags = react_review(plausibility_flags(graded, decision, context), config.backbone.max_react_iterations)
    return PipelineReport(findings=findings, graded=graded, iss=iss, decision=decision, flags=flags)
