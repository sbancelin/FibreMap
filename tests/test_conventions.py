"""Convention tests on known vectors and angles (phase0 acceptance: conventions tested)."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.representations import conventions as cv


# ----------------------------------------------------------------- director from known angles
@pytest.mark.parametrize(
    "phi, theta, expected",
    [
        (0.0, 0.0, [1.0, 0.0, 0.0]),  # +x
        (np.pi / 2, 0.0, [0.0, 1.0, 0.0]),  # +y
        (0.0, np.pi / 2, [0.0, 0.0, 1.0]),  # +z (out of plane)
        (0.0, -np.pi / 2, [0.0, 0.0, -1.0]),  # -z
        (np.pi / 4, 0.0, [np.sqrt(0.5), np.sqrt(0.5), 0.0]),
    ],
)
def test_director_from_angles_known(phi, theta, expected):
    n = cv.director_from_angles(phi, theta)
    assert np.allclose(n, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_director_is_unit_on_grid():
    phi = np.linspace(0, np.pi, 17, endpoint=False)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 17)
    pp, tt = np.meshgrid(phi, theta, indexing="ij")
    n = cv.director_from_angles(pp, tt)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-12)


def test_angles_director_roundtrip():
    # phi in [0, pi), theta in (-pi/2, pi/2): canonical representatives must round-trip exactly.
    phi = np.linspace(0.0, np.pi, 19, endpoint=False)
    theta = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 19)
    pp, tt = np.meshgrid(phi, theta, indexing="ij")
    n = cv.director_from_angles(pp, tt)
    phi_r, theta_r = cv.angles_from_director(n)
    assert np.allclose(phi_r, pp, atol=1e-9)
    assert np.allclose(theta_r, tt, atol=1e-9)


def test_axial_equivalence_n_equals_minus_n():
    # A director and its negation are the same axial orientation -> identical (phi, theta).
    phi, theta = 0.7, 0.3
    n = cv.director_from_angles(phi, theta)
    phi_p, theta_p = cv.angles_from_director(n)
    phi_m, theta_m = cv.angles_from_director(-n)
    assert np.isclose(phi_p, phi_m)
    assert np.isclose(theta_p, theta_m)
    assert np.isclose(phi_p, phi)
    assert np.isclose(theta_p, theta)


# ----------------------------------------------------------------------------- doubled angle
def test_doubled_angle_roundtrip():
    phi = np.linspace(0.0, np.pi, 37, endpoint=False)
    c2, s2 = cv.doubled_angle(phi)[..., 0], cv.doubled_angle(phi)[..., 1]
    phi_r = cv.angle_from_doubled(c2, s2)
    assert np.allclose(phi_r, phi, atol=1e-12)


def test_doubled_angle_identifies_0_and_pi():
    # Axial: phi and phi+pi map to the same doubled-angle point.
    for phi in (0.0, 0.3, 1.2):
        d0 = cv.doubled_angle(phi)
        d1 = cv.doubled_angle(phi + np.pi)
        assert np.allclose(d0, d1, atol=1e-12)


def test_wrap_axial():
    assert np.isclose(cv.wrap_axial(np.pi + 0.1), 0.1)
    assert np.isclose(cv.wrap_axial(-0.1), np.pi - 0.1)
    assert np.isclose(cv.wrap_axial(2 * np.pi), 0.0)


# --------------------------------------------------------------------------------- units boundary
def test_deg_rad_boundary():
    assert np.isclose(cv.deg2rad(90.0), np.pi / 2)
    assert np.isclose(cv.rad2deg(np.pi), 180.0)
    assert np.allclose(cv.rad2deg(cv.deg2rad([0, 45, 90, 135])), [0, 45, 90, 135])


# ------------------------------------------------------------------------------ voxel <-> physical
def test_voxel_to_physical_centre_and_order():
    vox = (0.5, 0.2, 0.2)  # (dz, dy, dx)
    # origin voxel -> origin
    assert np.allclose(cv.voxel_to_physical((0, 0, 0), vox), [0.0, 0.0, 0.0])
    # voxel (iz=2, iy=3, ix=4) -> (x=4*dx, y=3*dy, z=2*dz)
    p = cv.voxel_to_physical((2, 3, 4), vox)
    assert np.allclose(p, [4 * 0.2, 3 * 0.2, 2 * 0.5])


def test_voxel_physical_roundtrip():
    vox = (0.5, 0.2, 0.3)
    idx = np.array([[2.0, 3.0, 4.0], [10.0, 0.0, 7.0]])
    p = cv.voxel_to_physical(idx, vox)
    idx_r = cv.physical_to_voxel(p, vox)
    assert np.allclose(idx_r, idx)


def test_bounds_um():
    b = cv.bounds_um((4, 5, 6), (0.5, 0.2, 0.2))
    assert b == (0.0, 3 * 0.5, 0.0, 4 * 0.2, 0.0, 5 * 0.2)


# ----------------------------------------------------------------- structure-tensor minor axis
def test_minor_eigenvector_is_fiber_axis():
    # A fibre along +x: gradients live in (y, z); structure tensor is diag(small, big, big).
    # The minor (smallest-eigenvalue) eigenvector must be the fibre axis ~ +x.
    tensor = np.diag([0.01, 1.0, 1.0])
    axis = cv.minor_eigenvector(tensor)
    assert np.allclose(np.abs(axis), [1.0, 0.0, 0.0], atol=1e-9)
    grad = cv.major_eigenvector(tensor)
    assert np.isclose(np.abs(np.dot(grad, [1, 0, 0])), 0.0, atol=1e-9)


def test_minor_eigenvector_batch():
    tensors = np.stack([np.diag([0.01, 1.0, 1.0]), np.diag([1.0, 0.02, 1.0])])
    axes = cv.minor_eigenvector(tensors)
    assert np.allclose(np.abs(axes[0]), [1, 0, 0], atol=1e-9)
    assert np.allclose(np.abs(axes[1]), [0, 1, 0], atol=1e-9)