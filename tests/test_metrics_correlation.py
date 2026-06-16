"""Family C — orientation correlation: plateau = global order, ξ = correlation length."""

from __future__ import annotations

import numpy as np

from collagen_shg.metrics.correlation import orientation_correlation


def _block_field(n, block, rng):
    """A field of square domains of size ``block``, each a random axial orientation."""
    nb = n // block
    angles = rng.uniform(0, np.pi, (nb, nb))
    return np.repeat(np.repeat(angles, block, axis=0), block, axis=1)


def test_uniform_field_no_decay():
    theta = np.full((128, 128), 0.6)
    res = orientation_correlation(theta, max_r=40)
    assert np.allclose(res.C[np.isfinite(res.C)], 1.0, atol=1e-6)
    assert res.plateau > 0.99
    assert np.isinf(res.xi)


def test_isotropic_white_field_decays_immediately():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, (128, 128))
    res = orientation_correlation(theta, max_r=40)
    assert res.C[0] == 1.0
    assert res.plateau < 0.05  # globally isotropic
    assert res.xi < 2.0  # order does not persist


def test_xi_increases_with_domain_size():
    rng = np.random.default_rng(1)
    small = orientation_correlation(_block_field(128, 4, rng), max_r=40)
    large = orientation_correlation(_block_field(128, 16, rng), max_r=40)
    assert large.xi > small.xi
    # many random domains => globally isotropic (low plateau) despite local order
    assert small.plateau < 0.15
    assert large.plateau < 0.25


def test_plateau_distinguishes_globally_aligned():
    rng = np.random.default_rng(2)
    theta = 0.5 + rng.normal(0, 0.1, (128, 128))  # globally aligned with small dispersion
    res = orientation_correlation(theta, max_r=40)
    assert res.plateau > 0.8  # high global order, unlike the multi-domain case


def test_director_field_3d_uniform_vs_random():
    uni = np.zeros((3, 24, 24, 24))
    uni[0] = 1.0
    res_u = orientation_correlation(uni, max_r=8)
    assert res_u.plateau > 0.99

    rng = np.random.default_rng(3)
    v = rng.standard_normal((3, 24, 24, 24))
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    res_r = orientation_correlation(v, max_r=8)
    assert res_r.plateau < 0.1
    assert res_r.xi < 2.0
