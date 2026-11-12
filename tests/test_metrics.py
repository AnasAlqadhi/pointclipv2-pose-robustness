"""Tests for the metrics. No GPU, no CLIP, no dataset needed."""

import numpy as np
import pytest

from src.metrics import (
    accuracy_with_ci,
    mean_class_accuracy,
    paired_bootstrap_delta,
    per_class_accuracy,
)


def test_accuracy_value():
    pred = np.array([0, 1, 2, 3])
    true = np.array([0, 1, 2, 9])
    a = accuracy_with_ci(pred, true, n_boot=2000)
    assert a.value == pytest.approx(0.75)
    assert a.n == 4
    assert a.lo <= a.value <= a.hi


def test_perfect_and_zero_have_degenerate_intervals():
    true = np.arange(50)
    perfect = accuracy_with_ci(true, true, n_boot=2000)
    assert perfect.value == 1.0 and perfect.lo == 1.0 and perfect.hi == 1.0
    wrong = accuracy_with_ci(true + 1, true, n_boot=2000)
    assert wrong.value == 0.0 and wrong.hi == 0.0


def test_interval_narrows_with_more_data():
    rng = np.random.default_rng(0)
    widths = []
    for n in (100, 10_000):
        true = np.zeros(n, dtype=int)
        pred = (rng.random(n) < 0.3).astype(int)
        a = accuracy_with_ci(pred, true, n_boot=2000)
        widths.append(a.hi - a.lo)
    assert widths[1] < widths[0]


def test_bootstrap_is_deterministic_given_a_seed():
    rng = np.random.default_rng(1)
    true = rng.integers(0, 5, 200)
    pred = np.where(rng.random(200) < 0.7, true, (true + 1) % 5)
    a = accuracy_with_ci(pred, true, n_boot=2000, seed=7)
    b = accuracy_with_ci(pred, true, n_boot=2000, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        accuracy_with_ci(np.zeros(3), np.zeros(4))


def test_per_class_and_macro():
    true = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 1])
    pc = per_class_accuracy(pred, true, n_classes=2)
    assert pc[0].value == pytest.approx(0.5)
    assert pc[1].value == pytest.approx(1.0)
    assert mean_class_accuracy(pc) == pytest.approx(0.75)
    # macro differs from overall (0.8) because the classes are imbalanced
    assert accuracy_with_ci(pred, true, n_boot=1000).value == pytest.approx(0.8)


def test_absent_class_is_skipped():
    true = np.array([0, 0])
    pred = np.array([0, 0])
    assert set(per_class_accuracy(pred, true, n_classes=3)) == {0}


def test_paired_delta_sign_and_zero_case():
    true = np.arange(100) % 4
    worse = np.where(np.arange(100) % 2 == 0, true, (true + 1) % 4)  # 50%
    better = true.copy()  # 100%
    d = paired_bootstrap_delta(worse, better, true, n_boot=2000)
    assert d.value == pytest.approx(0.5)
    assert d.lo > 0  # a real improvement: interval excludes zero

    same = paired_bootstrap_delta(worse, worse.copy(), true, n_boot=2000)
    assert same.value == 0.0 and same.lo <= 0 <= same.hi
