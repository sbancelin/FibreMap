"""Orientation field + organization descriptors + bootstrap CIs."""

from __future__ import annotations

import numpy as np

from collagen_shg.analysis_resolved.descriptors import (
    DESCRIPTOR_NAMES,
    bootstrap_order_ci,
    descriptor_vector,
    organization_descriptors_3d,
)
from collagen_shg.analysis_resolved.orientation_field import multiscale_orientation_3d
from collagen_shg.representations import conventions as cv

VOXEL = (0.5, 0.2, 0.2)


def _director_field(shape, mean_phi, sigma_phi, rng):
    """A director field with azimuth ~ N(mean_phi, sigma_phi), elevation ~ 0."""
    phi = cv.wrap_axial(rng.normal(mean_phi, sigma_phi, size=shape))
    theta = np.zeros(shape)
    d = cv.director_from_angles(phi, theta)  # (..., 3)
    return np.moveaxis(d, -1, 0)  # [3, Z, Y, X]


def test_descriptors_aligned_field():
    rng = np.random.default_rng(0)
    director = _director_field((8, 32, 32), mean_phi=np.pi / 2, sigma_phi=0.1, rng=rng)
    fa = np.ones((8, 32, 32))
    desc = organization_descriptors_3d(director, fa, VOXEL)
    assert desc.S2 > 0.9
    assert desc.S3 > 0.8
    d = abs(np.angle(np.exp(1j * 2 * (desc.mean_phi - np.pi / 2))) / 2)
    assert d < np.deg2rad(3)


def test_descriptors_isotropic_field():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((3, 8, 32, 32))
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    desc = organization_descriptors_3d(v, np.ones((8, 32, 32)), VOXEL)
    assert desc.S2 < 0.2
    assert desc.S3 < 0.2


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(2)
    director = _director_field((6, 24, 24), mean_phi=0.0, sigma_phi=0.3, rng=rng)
    fa = np.ones((6, 24, 24))
    desc = organization_descriptors_3d(director, fa, VOXEL)
    ci = bootstrap_order_ci(director, fa, n_boot=100, rng=rng)
    lo, hi = ci["S2"]
    assert lo <= desc.S2 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_descriptor_vector_shape_and_order():
    rng = np.random.default_rng(3)
    director = _director_field((6, 16, 16), mean_phi=0.5, sigma_phi=0.2, rng=rng)
    desc = organization_descriptors_3d(director, np.ones((6, 16, 16)), VOXEL)
    vec = descriptor_vector(desc)
    assert vec.shape == (len(DESCRIPTOR_NAMES),)
    assert np.isfinite(vec).all()


def test_multiscale_orientation_recovers_axis():
    # a 3D linear structure along x (two perpendicular plane waves), like the metrics test
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    vol = np.sin(2 * np.pi * yy / 6.0) + np.sin(2 * np.pi * zz / 6.0)
    field = multiscale_orientation_3d(vol, sigma=1.0, rhos=(2.0, 4.0))
    interior = (slice(6, -6), slice(6, -6), slice(6, -6))
    fa = field.fa[interior]
    mags = [float(np.sum(fa * np.abs(field.director[k][interior])) / np.sum(fa)) for k in range(3)]
    assert mags[0] > 0.9  # director along x
