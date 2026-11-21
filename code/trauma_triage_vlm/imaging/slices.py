from __future__ import annotations


from typing import cast

import numpy as np


HU_RANGES: dict[str, tuple[int, int]] = {
    "liver": (40, 80),
    "spleen": (45, 65),
    "kidney": (30, 60),
    "brain": (20, 45),
    "bone": (300, 2000),
}


def component_area(mask: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, mask.reshape(mask.shape[0], -1).sum(axis=1))


def axial_range_for_organ(volume: np.ndarray, organ: str, min_area: int) -> tuple[int, int] | None:
    low, high = HU_RANGES[organ]
    mask = (volume >= low) & (volume <= high)
    area = component_area(mask)
    hits = np.flatnonzero(area >= min_area)
    if hits.size == 0:
        return None
    return int(hits[0]), int(hits[-1])


def uniform_slices(start: int, stop: int, count: int) -> tuple[int, ...]:
    if count <= 1:
        return (int(round((start + stop) / 2)),)
    return tuple(int(round(x)) for x in np.linspace(start, stop, count))


def landmark_slices(volume: np.ndarray, organ: str, min_area: int, lower_count: int, upper_count: int) -> tuple[int, ...]:
    organ_range = axial_range_for_organ(volume, organ, min_area)
    if organ_range is None:
        return uniform_slices(0, volume.shape[0] - 1, lower_count)
    start, stop = organ_range
    span = max(stop - start + 1, 1)
    count = min(max(lower_count, span // 18), upper_count)
    return uniform_slices(start, stop, count)


def entropy_slices(volume: np.ndarray, count: int) -> tuple[int, ...]:
    flat = volume.reshape(volume.shape[0], -1)
    scaled = np.clip((flat + 1024.0) / 4095.0, 0.0, 1.0)
    scores = -(scaled * np.log2(scaled + 1e-7) + (1.0 - scaled) * np.log2(1.0 - scaled + 1e-7)).mean(axis=1)
    picks = np.argsort(scores)[-count:]
    return tuple(int(x) for x in sorted(picks))
