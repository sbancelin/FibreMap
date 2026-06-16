"""Family D — Fourier: known orientation and spacing recovered from synthetic sinusoids."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.metrics.fourier import power_spectrum_orientation
from collagen_shg.representations import conventions as cv


def _axis_distance(phi, expected):
    return abs(np.angle(np.exp(1j * 2 * (phi - expected))) / 2)


@pytest.mark.parametrize(
    "a, b, lam, expected_fiber, expected_spacing",
    [
        (1.0, 0.0, 8.0, np.pi / 2, 8.0),  # varies along x -> fibre along y
        (0.0, 1.0, 8.0, 0.0, 8.0),  # varies along y -> fibre along x
    ],
)
def test_orientation_and_spacing(a, b, lam, expected_fiber, expected_spacing):
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(2 * np.pi * (a * xx + b * yy) / lam)
    res = power_spectrum_orientation(img)
    assert _axis_distance(res.orientation, expected_fiber) < np.deg2rad(3)
    assert res.spacing == pytest.approx(expected_spacing, rel=0.1)


def test_diagonal_orientation_and_spacing():
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    lam = 8.0
    img = np.sin(2 * np.pi * (xx + yy) / lam)  # wavevector along (1,1)
    res = power_spectrum_orientation(img)
    expected_fiber = cv.wrap_axial(np.pi / 4 + np.pi / 2)  # 135°
    assert _axis_distance(res.orientation, expected_fiber) < np.deg2rad(4)
    # spatial period along the wavevector is lam / sqrt(2)
    assert res.spacing == pytest.approx(lam / np.sqrt(2), rel=0.12)


def test_isotropic_has_flat_angular_distribution():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((128, 128))
    res = power_spectrum_orientation(img)
    A = res.angular_distribution
    # roughly flat: peak-to-mean ratio modest (no dominant orientation)
    assert A.max() / A.mean() < 3.0
