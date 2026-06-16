"""Preprocessing tests: shading removal, orientation-preserving denoise, background subtraction."""

from __future__ import annotations

import numpy as np

from collagen_shg.analysis_resolved.preprocess import (
    denoise,
    flat_field_correct,
    preprocess,
    subtract_background,
)
from collagen_shg.metrics.structure_tensor import structure_tensor_2d
from collagen_shg.representations import conventions as cv


def _oriented_texture(n=128, lam=8.0, rng=None):
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    base = 1.0 + 0.5 * np.sin(2 * np.pi * xx / lam)  # stripes -> fibre along y
    return base + 0.0 * rng.standard_normal((n, n))


def test_flat_field_removes_multiplicative_shading():
    n = 128
    rng = np.random.default_rng(0)
    texture = 1.0 + 0.3 * rng.standard_normal((n, n))
    ramp = np.linspace(0.5, 1.5, n)[None, :] * np.ones((n, 1))  # shading along x
    shaded = texture * ramp

    def halves_ratio(img):
        left = img[:, : n // 2].mean()
        right = img[:, n // 2 :].mean()
        return right / left

    corrected = flat_field_correct(shaded)
    # shading ratio driven from ~3.0 (=1.5/0.5) toward ~1.0
    assert abs(halves_ratio(corrected) - 1.0) < abs(halves_ratio(shaded) - 1.0)
    assert abs(halves_ratio(corrected) - 1.0) < 0.2


def test_denoise_preserves_orientation():
    rng = np.random.default_rng(1)
    clean = _oriented_texture(rng=rng)
    noisy = clean + 0.5 * rng.standard_normal(clean.shape)

    def mean_phi(img):
        res = structure_tensor_2d(img, 1.0, 4.0)
        interior = (slice(24, -24), slice(24, -24))
        c = np.sum(res.coherence[interior] * np.cos(2 * res.orientation[interior]))
        s = np.sum(res.coherence[interior] * np.sin(2 * res.orientation[interior]))
        return float(cv.angle_from_doubled(c, s))

    den = denoise(noisy, method="gaussian", size=1.5)
    # orientation after denoise matches the clean orientation (pi/2 for x-stripes)
    d = abs(np.angle(np.exp(1j * 2 * (mean_phi(den) - np.pi / 2))) / 2)
    assert d < np.deg2rad(4)
    assert den.std() < noisy.std()  # noise reduced


def test_subtract_background_removes_offset():
    rng = np.random.default_rng(2)
    img = 5.0 + np.abs(rng.standard_normal((64, 64)))  # constant veil + signal
    out = subtract_background(img)
    assert out.min() >= 0.0
    assert out.mean() < img.mean()


def test_preprocess_chain_runs():
    img = _oriented_texture()
    out = preprocess(img, flat_field=True, subtract_bg=True, denoise_sigma=1.0)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
