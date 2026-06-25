"""Director-field architectures + biaxial dispersion sampling."""

from __future__ import annotations

import numpy as np

from collagen_shg.representations import conventions as cv
from collagen_shg.structure_generator.architecture import (
    Arcade,
    Lamellar,
    Tubular,
    Uniaxial,
    build_architecture,
    sample_axial_directions,
)

SHAPE = (16, 64, 64)
VOXEL = (0.5, 0.2, 0.2)


def test_uniaxial_is_constant():
    f = Uniaxial(phi0=np.pi / 2, theta0=0.0)
    pts = np.random.default_rng(0).uniform(0, 10, (50, 3))
    d = f.at(pts)
    assert np.allclose(d, [0, 1, 0], atol=1e-9)


def test_lamellar_steps_in_z():
    f = Lamellar(thickness_um=2.0, dphi=np.pi / 2, phi_start=0.0)
    d0 = f.at(np.array([[1.0, 1.0, 0.5]]))[0]  # layer 0 -> phi 0 -> +x
    d1 = f.at(np.array([[1.0, 1.0, 2.5]]))[0]  # layer 1 -> phi 90 -> +y
    assert np.allclose(np.abs(d0), [1, 0, 0], atol=1e-6)
    assert np.allclose(np.abs(d1), [0, 1, 0], atol=1e-6)


def test_arcade_elevation_gradient():
    f = Arcade(theta_deep=np.pi / 2, theta_surface=0.0, z_max_um=10.0, phi0=0.0)
    surface = f.at(np.array([[0.0, 0.0, 0.0]]))[0]  # theta=0 -> in-plane +x
    deep = f.at(np.array([[0.0, 0.0, 10.0]]))[0]  # theta=90 -> +z
    assert abs(surface[2]) < 1e-6
    assert np.allclose(np.abs(deep), [0, 0, 1], atol=1e-6)


def test_tubular_is_circumferential():
    f = Tubular(center_xy=(0.0, 0.0), beta=0.0)
    # at point on +x axis, circumferential direction is +/- y
    d = f.at(np.array([[5.0, 0.0, 0.0]]))[0]
    assert np.allclose(np.abs(d), [0, 1, 0], atol=1e-6)
    # director is perpendicular to the radial vector everywhere
    pts = np.random.default_rng(1).uniform(-5, 5, (40, 3))
    dirs = f.at(pts)
    radial = pts.copy()
    radial[:, 2] = 0
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.where(rn > 0, rn, 1)
    dots = np.abs(np.einsum("ij,ij->i", dirs, radial))
    assert np.all(dots < 1e-6)


def test_tubular_helix_has_axial_component():
    f = Tubular(center_xy=(0.0, 0.0), beta=np.deg2rad(30))
    d = f.at(np.array([[5.0, 0.0, 0.0]]))[0]
    assert abs(d[2] - np.sin(np.deg2rad(30))) < 1e-6


def test_build_architecture_populations():
    assert len(build_architecture("uniaxial", {}, SHAPE, VOXEL)) == 1
    assert len(build_architecture("biaxial", {"phi_a_deg": 0, "phi_b_deg": 90}, SHAPE, VOXEL)) == 2
    crossed = build_architecture("tubular", {"helix_beta_deg": 30, "crossed": True}, SHAPE, VOXEL)
    assert len(crossed) == 2
    assert abs(sum(p.weight for p in crossed) - 1.0) < 1e-9


def test_dispersion_concentrated_vs_isotropic():
    rng = np.random.default_rng(2)
    mean = np.tile(cv.director_from_angles(0.0, 0.0), (5000, 1))  # +x
    tight = sample_axial_directions(mean, kappa_par=100, kappa_perp=100, rng=rng)
    assert np.mean(np.abs(tight[:, 0])) > 0.95  # stays near +x

    iso = sample_axial_directions(mean, kappa_par=0.0, kappa_perp=0.0, rng=rng)
    # near-uniform on sphere: mean |x| ~ 0.5
    assert abs(np.mean(np.abs(iso[:, 0])) - 0.5) < 0.1


def test_dispersion_biaxial_anisotropy():
    rng = np.random.default_rng(3)
    mean = np.tile(cv.director_from_angles(0.0, 0.0), (8000, 1))  # +x
    # spread in-plane (e1 ~ y) but tight out-of-plane (e2 ~ z)
    d = sample_axial_directions(mean, kappa_par=2.0, kappa_perp=200.0, rng=rng)
    assert np.std(d[:, 1]) > 3 * np.std(d[:, 2])  # more y-spread than z-spread