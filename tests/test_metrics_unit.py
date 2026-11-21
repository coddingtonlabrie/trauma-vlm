import numpy as np

from trauma_triage_vlm.metrics.classification import off_by_two_rate, within_one_accuracy


def test_ordinal_error_profile() -> None:
    y = np.array([1, 2, 3, 4, 5])
    p = np.array([1, 3, 3, 2, 5])
    assert within_one_accuracy(y, p) == 0.8
    assert off_by_two_rate(y, p) == 0.2
