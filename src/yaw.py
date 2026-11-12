"""Yaw rotation and canonicalisation for point clouds.

Convention: clouds are (N, 3) float arrays, yaw is rotation about the +Z axis (the
"up" axis in ModelNet40 and ScanObjectNN after their standard preprocessing).
If a dataset turns out to use +Y as up, set UP_AXIS once here rather than
sprinkling permutations through the experiment scripts.

Nothing in this module touches CLIP, a GPU or a dataset, so it is unit-testable
on its own. That is deliberate: the yaw maths is the part of the project that
must be provably right, because every claim rests on it.
"""

from __future__ import annotations

import numpy as np

# 0=x, 1=y, 2=z. **+Y, established by looking at the renders, not assumed.**
# scripts/check_up_axis.py rotates a chair, a person, an airplane and a bottle
# about each candidate axis and renders them: with up=y every object turns while
# staying upright, while up=x and up=z tip them over. Figure:
# results/figures/up_axis_check.png. This matches PointNet's own augmentation,
# which rotates about Y. Changing this invalidates every yaw result, so if it is
# ever changed, re-run the sweep and log it.
UP_AXIS = 1


def _axes_in_plane(up: int = UP_AXIS) -> tuple[int, int]:
    """The two axes spanning the plane perpendicular to the up axis."""
    return tuple(a for a in (0, 1, 2) if a != up)  # type: ignore[return-value]


def rotation_matrix(angle_rad: float, up: int = UP_AXIS) -> np.ndarray:
    """Rotation by `angle_rad` about the up axis. Right-handed, det = +1."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    a, b = _axes_in_plane(up)
    R = np.eye(3, dtype=np.float64)
    R[a, a], R[a, b] = c, -s
    R[b, a], R[b, b] = s, c
    return R


def rotate_yaw(points: np.ndarray, angle_deg: float, up: int = UP_AXIS) -> np.ndarray:
    """Rotate a cloud about the up axis by `angle_deg` degrees."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {points.shape}")
    return points @ rotation_matrix(np.deg2rad(angle_deg), up).T


def yaw_grid(step_deg: float = 30.0) -> np.ndarray:
    """The sweep angles: 0, step, ... up to but not including 360."""
    if step_deg <= 0 or 360.0 % step_deg != 0:
        raise ValueError("step_deg must divide 360")
    return np.arange(0.0, 360.0, step_deg)


def principal_yaw(points: np.ndarray, up: int = UP_AXIS) -> float:
    """Estimated yaw of the cloud, in degrees, from PCA in the horizontal plane.

    The first principal direction of the projected points is taken as the
    object's own axis. Returned in [0, 180): PCA gives a direction, not an
    orientation, so the sign of the eigenvector is arbitrary and a 180-degree
    ambiguity is unavoidable. Resolving it needs a semantic cue (CLIP itself, or
    front/back asymmetry) - see `docs/REPRO_LOG.md` note on C4.
    """
    points = np.asarray(points, dtype=np.float64)
    a, b = _axes_in_plane(up)
    xy = points[:, [a, b]]
    xy = xy - xy.mean(axis=0, keepdims=True)
    # Covariance eigenvector with the largest eigenvalue.
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.rad2deg(np.arctan2(major[1], major[0])))
    return angle % 180.0


def canonicalise_yaw(points: np.ndarray, up: int = UP_AXIS) -> tuple[np.ndarray, float]:
    """Rotate the cloud so its principal horizontal axis lies along the first axis.

    Returns the rotated cloud and the yaw that was removed (degrees). This is the
    cheap, training-free candidate fix for C4: it costs one eigen-decomposition
    of a 2x2 matrix per object and no extra CLIP forward passes, which is what
    makes it a fair comparison against the baseline on the compute axis.
    """
    angle = principal_yaw(points, up)
    return rotate_yaw(points, -angle, up), angle


def extent_yaw(points: np.ndarray, step_deg: float = 5.0, up: int = UP_AXIS) -> float:
    """Alternative canonicaliser: the yaw minimising the horizontal bounding-box area.

    An object-aligned bounding box is a different inductive bias from PCA - it
    follows the silhouette rather than the mass distribution. Kept as an ablation
    so the report can say which canonicaliser mattered, instead of claiming
    "canonicalisation helps" from a single implementation.

    A bounding box is symmetric under a 90-degree turn: a bar lying along x and
    the same bar lying along y have the same box area. That tie is broken by
    preferring the orientation whose extent along the first in-plane axis is
    larger, so the output is a single well-defined angle rather than one of two.
    """
    a, b = _axes_in_plane(up)
    angles = np.arange(0.0, 180.0, step_deg)
    areas, spreads = [], []
    for angle in angles:
        r = rotate_yaw(points, -angle, up)[:, [a, b]]
        p0, p1 = float(np.ptp(r[:, 0])), float(np.ptp(r[:, 1]))
        areas.append(p0 * p1)
        spreads.append(p0)
    areas, spreads = np.asarray(areas), np.asarray(spreads)
    # Candidates within a small relative tolerance of the smallest area.
    tol = 1e-6 * max(float(areas.min()), 1e-12) + 1e-12
    candidates = np.flatnonzero(areas <= areas.min() + tol)
    best = candidates[int(np.argmax(spreads[candidates]))]
    return float(angles[best])


def yaw_ensemble_angles(n: int, span_deg: float = 360.0, up: int = UP_AXIS) -> np.ndarray:
    """`n` yaw angles spread evenly over `span_deg`, for the ensembling fix.

    **Do not leave the span at 360 for this projection.** Its ten viewpoints sit
    at azimuths 0, 45, 90, ... 315 (two groups of four, plus top and bottom), so
    the view set is closed under a 90-degree turn of the object: rotating by a
    multiple of 90 degrees re-renders almost the same images in a different
    order and the ensemble is nearly a no-op. Measured: averaging over
    [0, 90, 180, 270] changed the logits by at most 4.3 and flipped no
    predictions at all on a 32-object probe.

    To add views that are genuinely new, spread the offsets across one *tooth*
    of the grid: span 45 degrees.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if span_deg <= 0:
        raise ValueError("span_deg must be positive")
    return np.linspace(0.0, span_deg, n, endpoint=False)
