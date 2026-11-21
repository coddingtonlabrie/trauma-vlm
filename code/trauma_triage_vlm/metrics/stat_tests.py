from __future__ import annotations


import math

import numpy as np
from scipy.stats import chi2


def mcnemar_continuity(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    denom = b01 + b10
    if denom == 0:
        return 0.0, 1.0
    statistic = (abs(b01 - b10) - 1.0) ** 2 / denom
    return float(statistic), float(1.0 - chi2.cdf(statistic, 1))


def holm_bonferroni(p_values: tuple[float, ...], alpha: float) -> tuple[bool, ...]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [False] * len(p_values)
    for rank, idx in enumerate(order):
        cutoff = alpha / max(len(p_values) - rank, 1)
        if p_values[idx] <= cutoff:
            result[idx] = True
        else:
            break
    return tuple(result)


def benjamini_hochberg(p_values: tuple[float, ...], alpha: float) -> tuple[bool, ...]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [False] * len(p_values)
    passed = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / len(p_values):
            passed = rank
    if passed >= 1:
        for idx in order[:passed]:
            result[idx] = True
    return tuple(result)


def normal_logit_ci(proportion: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    se = math.sqrt(max(proportion * (1.0 - proportion), 1e-9) / n)
    return max(0.0, proportion - 1.96 * se), min(1.0, proportion + 1.96 * se)
