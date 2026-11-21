from __future__ import annotations


import numpy as np

from trauma_triage_vlm.types import BodyRegion, Finding


SIGNAL_THRESHOLDS: dict[str, tuple[str, float]] = {
    "liver": ("laceration_depth", 0.58),
    "spleen": ("subcapsular_collection", 0.56),
    "kidney": ("perfusion_defect", 0.54),
    "bowel_mesenteric": ("free_fluid", 0.52),
    "extravasation": ("active_bleeding", 0.70),
    "ich": ("sulcal_density", 0.60),
    "calvarial_fracture": ("calvarial_discontinuity", 0.66),
}


REGIONS: dict[str, BodyRegion] = {
    "liver": BodyRegion.abdomen,
    "spleen": BodyRegion.abdomen,
    "kidney": BodyRegion.abdomen,
    "bowel_mesenteric": BodyRegion.abdomen,
    "extravasation": BodyRegion.abdomen,
    "ich": BodyRegion.head_neck,
    "calvarial_fracture": BodyRegion.head_neck,
}


def visual_score(rendered: np.ndarray) -> float:
    gradients = np.abs(np.diff(rendered.astype(np.float32), axis=-1)).mean()
    contrast = rendered.std()
    intensity = rendered.mean()
    return float(np.clip(0.45 * gradients + 0.35 * contrast + 0.20 * intensity, 0.0, 1.0))


def detect_organ(organ: str, rendered_slices: np.ndarray, slice_ids: tuple[int, ...]) -> Finding:
    score = visual_score(rendered_slices)
    descriptor, threshold = SIGNAL_THRESHOLDS[organ]
    descriptors = (descriptor,) if score >= threshold else tuple()
    return Finding(
        organ=organ,
        region=REGIONS[organ],
        present=score >= threshold,
        confidence=score,
        descriptors=descriptors,
        slices=slice_ids,
    )


def detect_all(rendered_by_organ: dict[str, tuple[np.ndarray, tuple[int, ...]]]) -> tuple[Finding, ...]:
    return tuple(detect_organ(organ, rendered, ids) for organ, (rendered, ids) in rendered_by_organ.items())
