from __future__ import annotations


import numpy as np


def window_ct(slice_hu: np.ndarray, width: float, level: float) -> np.ndarray:
    low = level - width / 2.0
    high = level + width / 2.0
    clipped = np.clip(slice_hu.astype(np.float32), low, high)
    return (clipped - low) / max(high - low, 1e-6)


def render_windows(slice_hu: np.ndarray, presets: tuple[tuple[float, float], ...]) -> np.ndarray:
    channels = [window_ct(slice_hu, width, level) for width, level in presets]
    return np.stack(channels, axis=0)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(volume.astype(np.float32), nan=-1024.0, posinf=3071.0, neginf=-1024.0)
    return np.clip(finite, -1024.0, 3071.0)
