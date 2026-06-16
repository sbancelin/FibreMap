"""Family A — structure tensor: analytical orientation tests on synthetic patterns."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.metrics.structure_tensor import structure_tensor_2d, structure_tensor_3d
from collagen_shg.representations import conventions as cv


def _coherence_weighted_mean_phi(orientation, coherence):
    """Coherence-weighted circular mean of an axial orientation field (doubled angle)."""
    w = coherence
    c = np.sum(w * np.cos(2 * orientation))
    s = np.sum(w * np.sin(2 * orientation))
    return float(cv.angle_from_doubled(c, s))


def _plane_wave_2d(n, lam, a, b):
    """sin(2π (a·x + b·y)/λ): wavevector along (a, b); fibre axis is perpendicular."""
    yy, xx = np.mgrid[0:n, 0:n]
    return np.sin(2 * np.pi * (a * xx + b * yy) / lam)


@pytest.mark.parametrize(
    "a, b, expected_fiber",
    [
        (1.0, 0.0, np.pi / 2),  # varies along x -> fibre along y
        (0.0, 1.0, 0.0),  # varies along y -> fibre along x
        (1.0, 1.0, cv.wrap_axial(np.pi / 4 + np.pi / 2)),  # 45° wavevector -> 135° fibre
    ],
)
def test_structure_tensor_2d_orientation(a, b, expected_fiber):
    img = _plane_wave_2d(128, lam=8.0, a=a, b=b)
    res = structure_tensor_2d(img, sigma=1.0, rho=4.0)
    interior = (slice(24, -24), slice(24, -24))
    phi = _coherence_weighted_mean_phi(res.orientation[interior], res.coherence[interior])
    # circular distance on axial angles
    d = np.angle(np.exp(1j * 2 * (phi - expected_fiber))) / 2
    assert abs(d) < np.deg2rad(3), f"phi={np.rad2deg(phi):.1f} exp={np.rad2deg(expected_fiber):.1f}"


def test_structure_tensor_2d_coherence_high_for_oriented_low_for_uniform():
    oriented = _plane_wave_2d(128, lam=8.0, a=1.0, b=0.0)
    res = structure_tensor_2d(oriented, sigma=1.0, rho=4.0)
    interior = (slice(24, -24), slice(24, -24))
    assert res.coherence[interior].mean() > 0.8

    uniform = np.ones((128, 128))
    res_u = structure_tensor_2d(uniform, sigma=1.0, rho=4.0)
    assert res_u.coherence.mean() < 1e-6


def _fiber_volume(n, lam, axis):
    """A linear structure along ``axis`` (constant along it, two perpendicular plane waves)."""
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    coords = {"x": xx, "y": yy, "z": zz}
    perp = [c for c in ("z", "y", "x") if c != axis]
    return np.sin(2 * np.pi * coords[perp[0]] / lam) + np.sin(2 * np.pi * coords[perp[1]] / lam)


@pytest.mark.parametrize("axis, comp", [("x", 0), ("y", 1), ("z", 2)])
def test_structure_tensor_3d_director_axis(axis, comp):
    vol = _fiber_volume(40, lam=6.0, axis=axis)
    res = structure_tensor_3d(vol, sigma=1.0, rho=3.0)
    interior = (slice(8, -8), slice(8, -8), slice(8, -8))
    fa = res.fa[interior]
    # FA-weighted mean magnitude of each director component (axial => use |component|)
    mags = [
        float(np.sum(fa * np.abs(res.director[k][interior])) / np.sum(fa)) for k in range(3)
    ]
    assert mags[comp] > 0.9
    for k in range(3):
        if k != comp:
            assert mags[k] < 0.35


def test_structure_tensor_3d_fa_zero_for_uniform():
    vol = np.ones((20, 20, 20))
    res = structure_tensor_3d(vol, sigma=1.0, rho=3.0)
    assert res.fa.mean() < 1e-6
