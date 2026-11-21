from __future__ import annotations


from collections.abc import Callable

import numpy as np


Metric = Callable[[np.ndarray, np.ndarray], float]


def stratified_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric: Metric, iterations: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    estimates: list[float] = []
    for _ in range(iterations):
        indices: list[int] = []
        for cls in classes:
            cls_idx = np.flatnonzero(y_true == cls)
            draw = rng.choice(cls_idx, size=cls_idx.size, replace=True)
            indices.extend(int(x) for x in draw)
        idx = np.array(indices, dtype=int)
        estimates.append(metric(y_true[idx], y_score[idx]))
    point = metric(y_true, y_score)
    lo, hi = np.nanpercentile(np.array(estimates, dtype=float), [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_difference_ci(a: np.ndarray, b: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    delta = a - b
    draws = []
    for _ in range(iterations):
        idx = rng.choice(np.arange(delta.size), size=delta.size, replace=True)
        draws.append(float(delta[idx].mean()))
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)
