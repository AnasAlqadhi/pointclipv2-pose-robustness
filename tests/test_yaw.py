"""Tests for the yaw maths. No GPU, no CLIP, no dataset needed."""

import numpy as np
import pytest

from src.yaw import (
    UP_AXIS,
    canonicalise_yaw,
    extent_yaw,
    principal_yaw,
    rotate_yaw,
    rotation_matrix,
    yaw_ensemble_angles,
    yaw_grid,
)


def _bar(n=500, seed=0):
    """An elongated cloud along +x, so its principal yaw is 0."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(-1, 1, n), rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n)], axis=1
    )


def test_rotation_matrix_is_a_rotation():
    for angle in (0.0, 0.3, 1.9, -2.5):
        R = rotation_matrix(angle)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_preserves_shape_and_distances():
    p = _bar()
    r = rotate_yaw(p, 47.0)
    assert r.shape == p.shape
    assert np.allclose(np.linalg.norm(p, axis=1), np.linalg.norm(r, axis=1))


def test_rotation_is_invertible():
    p = _bar()
    assert np.allclose(rotate_yaw(rotate_yaw(p, 123.0), -123.0), p)


def test_360_is_identity():
    p = _bar()
    assert np.allclose(rotate_yaw(p, 360.0), p, atol=1e-9)


def test_rotation_does_not_touch_the_up_axis():
    """Whichever axis UP_AXIS names, that coordinate must be left alone.

    This used to hard-code column 2 from the old default. The default moved to
    +Y once the renders settled it, so the assertion now follows UP_AXIS instead
    of repeating the constant - the property being checked is unchanged.
    """
    p = _bar()
    assert np.allclose(rotate_yaw(p, 77.0)[:, UP_AXIS], p[:, UP_AXIS])
    moved = [a for a in (0, 1, 2) if a != UP_AXIS]
    assert not np.allclose(rotate_yaw(p, 77.0)[:, moved], p[:, moved])


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        rotate_yaw(np.zeros((10, 2)), 0.0)


def _circular_gap(a, b, period=180.0):
    """Distance between two angles on a circle of the given period."""
    d = abs(a - b) % period
    return min(d, period - d)


@pytest.mark.parametrize("applied", [0.0, 15.0, 42.0, 90.0, 137.0, 179.0])
def test_principal_yaw_recovers_an_applied_rotation(applied):
    """Compared circularly: 0 and 179.9 are the same orientation, not 180 apart."""
    p = rotate_yaw(_bar(), applied)
    assert _circular_gap(principal_yaw(p), applied) < 2.0


def test_canonicalise_is_idempotent():
    p = rotate_yaw(_bar(), 61.0)
    once, _ = canonicalise_yaw(p)
    twice, second_angle = canonicalise_yaw(once)
    assert second_angle == pytest.approx(0.0, abs=2.0) or second_angle == pytest.approx(180.0, abs=2.0)
    assert np.allclose(once, twice, atol=1e-6)


def test_canonicalise_removes_the_difference_between_two_poses():
    """The point of the fix: two yaws of the same object land in the same place.

    Only up to the 180-degree PCA ambiguity, which is why the report must treat
    that ambiguity as a known limitation rather than a bug.
    """
    base = _bar()
    a, _ = canonicalise_yaw(rotate_yaw(base, 20.0))
    b, _ = canonicalise_yaw(rotate_yaw(base, 200.0))
    same = np.allclose(a, b, atol=1e-6)
    flipped = np.allclose(a, rotate_yaw(b, 180.0), atol=1e-6)
    assert same or flipped


def test_extent_yaw_finds_the_bar_axis():
    p = rotate_yaw(_bar(), 35.0)
    assert extent_yaw(p, step_deg=1.0) == pytest.approx(35.0, abs=3.0)


def test_yaw_grid():
    g = yaw_grid(30.0)
    assert g[0] == 0.0 and len(g) == 12 and g[-1] == 330.0
    with pytest.raises(ValueError):
        yaw_grid(7.0)


def test_ensemble_angles():
    assert np.allclose(yaw_ensemble_angles(4), [0.0, 90.0, 180.0, 270.0])
    with pytest.raises(ValueError):
        yaw_ensemble_angles(0)
