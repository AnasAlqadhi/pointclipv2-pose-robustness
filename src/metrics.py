"""Accuracy, per-class accuracy and bootstrap confidence intervals.

"Honest uncertainty" is a named assessment criterion, so every accuracy reported
in the project goes through `accuracy_with_ci` and every yaw curve carries error
bars. The method is deterministic given the seed, which is recorded per run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Accuracy:
    """An accuracy with its bootstrap interval, over `n` test items."""

    value: float
    lo: float
    hi: float
    n: int
    confidence: float = 0.95

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{100 * self.value:.2f}% [{100 * self.lo:.2f}, {100 * self.hi:.2f}] (n={self.n})"


def _check(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} differ")
    if pred.size == 0:
        raise ValueError("empty predictions")
    return pred, true


def accuracy_with_ci(
    pred: np.ndarray,
    true: np.ndarray,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Accuracy:
    """Overall accuracy with a percentile bootstrap interval over test items."""
    pred, true = _check(pred, true)
    correct = (pred == true).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, correct.size, size=(n_boot, correct.size))
    boots = correct[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return Accuracy(float(correct.mean()), float(lo), float(hi), int(correct.size), confidence)


def per_class_accuracy(
    pred: np.ndarray, true: np.ndarray, n_classes: int | None = None
) -> dict[int, Accuracy]:
    """Accuracy per class, each with its own interval.

    This is the evidence for C3: the prediction is that rotationally symmetric
    classes (bottle, vase, cone) barely move under yaw while oriented ones
    (chair, desk, bed) collapse. A mean accuracy cannot show that; this can.
    """
    pred, true = _check(pred, true)
    classes = range(n_classes) if n_classes is not None else sorted(set(true.tolist()))
    out: dict[int, Accuracy] = {}
    for c in classes:
        mask = true == c
        if not mask.any():
            continue
        out[int(c)] = accuracy_with_ci(pred[mask], true[mask], seed=int(c))
    return out


def mean_class_accuracy(per_class: dict[int, Accuracy]) -> float:
    """Macro average. ModelNet40 is imbalanced, so report this next to overall."""
    if not per_class:
        raise ValueError("no classes")
    return float(np.mean([a.value for a in per_class.values()]))


def paired_bootstrap_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Accuracy:
    """Interval on (accuracy of B minus accuracy of A) on the SAME test items.

    Use this for every comparison in the report - fix versus baseline, fix versus
    same-budget control. The two systems see identical items, so pairing removes
    test-set variance and the interval is much tighter than comparing two
    independent intervals. If the interval contains 0, say so plainly: a
    difference that is not resolvable at this sample size is not an improvement.
    """
    pred_a, true = _check(pred_a, true)
    pred_b, _ = _check(pred_b, true)
    diff = (pred_b == true).astype(np.float64) - (pred_a == true).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return Accuracy(float(diff.mean()), float(lo), float(hi), int(diff.size), confidence)
