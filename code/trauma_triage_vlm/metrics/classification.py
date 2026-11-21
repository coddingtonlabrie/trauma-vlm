from __future__ import annotations


import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def exact_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def within_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) <= 1))


def off_by_two_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) >= 2))


def quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def high_acuity_sensitivity(y_true_esi: np.ndarray, y_pred_esi: np.ndarray) -> float:
    mask = y_true_esi <= 2
    if not bool(mask.any()):
        return float("nan")
    return float(np.mean(y_pred_esi[mask] <= 2))
