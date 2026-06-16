"""Family E — texture: GLCM anisotropy, LBP histogram, Gabor orientation."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.metrics.texture import gabor_energy, glcm_features, lbp_histogram


def _stripes(n=96, lam=8.0, along="x"):
    yy, xx = np.mgrid[0:n, 0:n]
    coord = xx if along == "x" else yy
    return np.sin(2 * np.pi * coord / lam)


def test_glcm_anisotropy_high_for_stripes_low_for_noise():
    stripes = _stripes(along="x")
    iso = np.random.default_rng(0).standard_normal((96, 96))
    a_stripes = glcm_features(stripes)["anisotropy"]
    a_iso = glcm_features(iso)["anisotropy"]
    assert a_stripes > 0.5
    assert a_iso < 0.2
    assert a_stripes > a_iso


def test_glcm_returns_haralick_keys():
    feats = glcm_features(_stripes())
    for key in ("contrast", "homogeneity", "ASM", "energy", "correlation", "anisotropy"):
        assert key in feats


def test_lbp_histogram_normalized_and_discriminative():
    h_stripes = lbp_histogram(_stripes(), P=8, R=1.0)
    h_iso = lbp_histogram(np.random.default_rng(1).standard_normal((96, 96)), P=8, R=1.0)
    assert h_stripes.shape == (10,)  # P + 2 for uniform
    assert h_stripes.sum() == pytest.approx(1.0)
    assert not np.allclose(h_stripes, h_iso)


def test_gabor_energy_orientation_and_frequency():
    # stripes varying along x -> fibre along y (pi/2); spatial frequency 1/lam
    res = gabor_energy(_stripes(lam=10.0, along="x"))
    d = abs(np.angle(np.exp(1j * 2 * (res.orientation - np.pi / 2))) / 2)
    assert d < np.deg2rad(15)
    assert res.peak_frequency == pytest.approx(0.1, abs=0.05)
