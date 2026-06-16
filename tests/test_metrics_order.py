"""Family B — order parameters: uniform => S=1, isotropic => S=0, and intermediate."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.metrics.order import order_parameter_2d, order_tensor_3d
from collagen_shg.representations import conventions as cv


def test_order_parameter_2d_uniform_gives_S2_one():
    theta = np.full((64, 64), 0.7)
    res = order_parameter_2d(theta)
    assert res.S2 == pytest.approx(1.0, abs=1e-6)
    assert abs(res.theta_bar - 0.7) < 1e-9
    assert res.kappa > 50  # very concentrated


def test_order_parameter_2d_isotropic_gives_S2_zero():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, size=200_000)  # uniform axial angles
    res = order_parameter_2d(theta)
    assert res.S2 < 0.01
    assert res.kappa < 0.05


def test_order_parameter_2d_recovers_mean_and_intermediate_S2():
    rng = np.random.default_rng(1)
    mu = 1.0
    # von Mises on the doubled angle => axial distribution centred on mu
    two_theta = rng.vonmises(2 * mu, kappa=4.0, size=100_000)
    theta = cv.wrap_axial(two_theta / 2)
    res = order_parameter_2d(theta)
    assert 0.2 < res.S2 < 0.95
    d = np.angle(np.exp(1j * 2 * (res.theta_bar - mu))) / 2
    assert abs(d) < np.deg2rad(2)


def test_order_tensor_3d_aligned_gives_S3_one():
    director = np.zeros((3, 1000))
    director[0] = 1.0  # all along +x
    res = order_tensor_3d(director)
    assert res.S3 == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.abs(res.director), [1, 0, 0], atol=1e-9)


def test_order_tensor_3d_isotropic_gives_S3_zero():
    rng = np.random.default_rng(2)
    v = rng.standard_normal((3, 200_000))
    v /= np.linalg.norm(v, axis=0)
    res = order_tensor_3d(v)
    assert res.S3 < 0.02


def test_order_tensor_3d_axial_symmetry_n_equals_minus_n():
    rng = np.random.default_rng(3)
    director = np.zeros((3, 1000))
    director[2] = 1.0
    # flip half the directors: axial => must not change S3 or the axis
    flip = rng.random(1000) < 0.5
    director[:, flip] *= -1
    res = order_tensor_3d(director)
    assert res.S3 == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.abs(res.director), [0, 0, 1], atol=1e-9)
