"""Coordinate, axis, unit and angle conventions — fixed once, verified by tests.

These conventions are invariant across the whole project (see ``CLAUDE.md`` and
``docs/phase0_*``). Orientation ambiguities are the first source of bugs in fibrillar
analysis, so everything here is explicit and covered by ``tests/test_conventions.py`` on
known vectors and angles.

Summary
-------
- **Axis order**: arrays are indexed ``[z, y, x]`` (z slowest, x fastest), C-contiguous.
- **Physical frame**: right-handed — x→right, y→up, z = depth into the tissue (≥ 0 at the
  surface, increasing inward). At pixel display (origin top-left, y downward) the angle
  appears mirrored; computation always uses the physical frame.
- **Units**: lengths in micrometres (µm), angles in radians internally. Degrees appear only
  at the GUI boundary (``deg2rad`` / ``rad2deg`` helpers).
- **Voxel scale**: the physical coordinate of voxel ``(iz, iy, ix)`` is
  ``(ix·dx, iy·dy, iz·dz)`` (voxel centres), with voxel size ``(dz, dy, dx)`` in µm.
- **Director**: unit vector ``n = (cosθ·cosφ, cosθ·sinφ, sinθ)`` with azimuth ``φ`` measured
  in the (x, y) plane from +x toward +y, and elevation ``θ`` between the fibril and the
  (x, y) plane.
- **Axial orientation**: fibres are axial (``n ≡ −n``, period π); azimuth lives in ``[0, π)``
  and is manipulated via the **doubled angle** ``(cos 2φ, sin 2φ)`` to avoid the 0/π jump.
- **Structure tensor**: the fibre axis is the **minor** eigenvector (smallest eigenvalue) of
  the structure tensor (gradients are perpendicular to the fibre). Do not confuse it with the
  gradient axis (major eigenvector).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AXIS_ORDER",
    "PHI_RANGE",
    "THETA_RANGE",
    "ALPHA_RANGE",
    "deg2rad",
    "rad2deg",
    "wrap_axial",
    "wrap_polarization",
    "voxel_to_physical",
    "physical_to_voxel",
    "bounds_um",
    "director_from_angles",
    "angles_from_director",
    "doubled_angle",
    "angle_from_doubled",
    "minor_eigenvector",
    "major_eigenvector",
]

#: Index order of every N-D array in the project (z slowest, x fastest).
AXIS_ORDER: tuple[str, str, str] = ("z", "y", "x")

#: Azimuth range for axial orientations.
PHI_RANGE: tuple[float, float] = (0.0, np.pi)
#: Elevation range.
THETA_RANGE: tuple[float, float] = (-np.pi / 2, np.pi / 2)
#: Incident linear-polarization range.
ALPHA_RANGE: tuple[float, float] = (0.0, np.pi)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- units (boundary)
def deg2rad(degrees: ArrayLike) -> NDArray[np.float64]:
    """Degrees → radians. Use only at the GUI boundary; internals are radians."""
    return np.deg2rad(np.asarray(degrees, dtype=np.float64))


def rad2deg(radians: ArrayLike) -> NDArray[np.float64]:
    """Radians → degrees. Use only at the GUI boundary."""
    return np.rad2deg(np.asarray(radians, dtype=np.float64))


# ------------------------------------------------------------------------------- angle wrapping
def wrap_axial(phi: ArrayLike) -> NDArray[np.float64]:
    """Wrap an axial azimuth into ``[0, π)`` (period π, since ``n ≡ −n``)."""
    phi = np.asarray(phi, dtype=np.float64)
    return np.mod(phi, np.pi)


def wrap_polarization(alpha: ArrayLike) -> NDArray[np.float64]:
    """Wrap an incident polarization angle into ``[0, π)`` (period π)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    return np.mod(alpha, np.pi)


# ------------------------------------------------------------------------------- voxel <-> µm
def voxel_to_physical(index_zyx: ArrayLike, voxel_size_zyx: ArrayLike) -> NDArray[np.float64]:
    """Voxel index ``(iz, iy, ix)`` → physical coordinate ``(x, y, z)`` in µm (voxel centres).

    Returns the point in **physical (x, y, z) order** so it composes with directors/angles.
    Accepts a single index (shape ``(3,)``) or a batch (shape ``(..., 3)``).
    """
    idx = np.asarray(index_zyx, dtype=np.float64)
    dz, dy, dx = (float(v) for v in voxel_size_zyx)
    iz, iy, ix = idx[..., 0], idx[..., 1], idx[..., 2]
    return np.stack([ix * dx, iy * dy, iz * dz], axis=-1)


def physical_to_voxel(point_xyz: ArrayLike, voxel_size_zyx: ArrayLike) -> NDArray[np.float64]:
    """Physical coordinate ``(x, y, z)`` in µm → fractional voxel index ``(iz, iy, ix)``.

    Inverse of :func:`voxel_to_physical` (no rounding; caller decides nearest/floor).
    """
    pt = np.asarray(point_xyz, dtype=np.float64)
    dz, dy, dx = (float(v) for v in voxel_size_zyx)
    x, y, z = pt[..., 0], pt[..., 1], pt[..., 2]
    return np.stack([z / dz, y / dy, x / dx], axis=-1)


def bounds_um(shape_zyx: ArrayLike, voxel_size_zyx: ArrayLike) -> tuple[float, ...]:
    """Physical extent ``(zmin, zmax, ymin, ymax, xmin, xmax)`` in µm for a voxel grid.

    Spans voxel-centre 0 to voxel-centre ``(n-1)`` along each axis.
    """
    nz, ny, nx = (int(s) for s in shape_zyx)
    dz, dy, dx = (float(v) for v in voxel_size_zyx)
    return (0.0, (nz - 1) * dz, 0.0, (ny - 1) * dy, 0.0, (nx - 1) * dx)


# ------------------------------------------------------------------------- director <-> angles
def director_from_angles(phi: ArrayLike, theta: ArrayLike) -> NDArray[np.float64]:
    """Azimuth ``φ`` and elevation ``θ`` (radians) → unit director ``n = (nx, ny, nz)``.

    ``n = (cosθ·cosφ, cosθ·sinφ, sinθ)``. Supports scalars or broadcastable arrays; the
    director is returned in physical ``(x, y, z)`` order along the last axis.
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    ct = np.cos(theta)
    nx = ct * np.cos(phi)
    ny = ct * np.sin(phi)
    nz = np.sin(theta)
    return np.stack([nx, ny, nz], axis=-1)


def angles_from_director(n: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit director ``n = (nx, ny, nz)`` → canonical axial ``(φ, θ)`` with ``φ ∈ [0, π)``.

    Because fibres are axial (``n ≡ −n``), the representative with ``φ ∈ [0, π)`` is chosen;
    flipping the vector flips the sign of ``θ`` accordingly, so that
    ``angles_from_director(director_from_angles(φ, θ)) == (φ, θ)`` for ``φ ∈ [0, π)`` and
    ``θ ∈ (−π/2, π/2)``.
    """
    n = np.asarray(n, dtype=np.float64)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz_n = np.where(norm > 0, nz / norm, 0.0)
    phi = np.arctan2(ny, nx)  # (-π, π]
    theta = np.arcsin(np.clip(nz_n, -1.0, 1.0))
    # Canonicalize azimuth into [0, π); flipping the director flips theta's sign.
    flip = (phi < 0) | (phi >= np.pi)
    phi = np.where(flip, phi + np.pi, phi)
    phi = np.mod(phi, np.pi)  # collapse the phi == π edge to 0
    theta = np.where(flip, -theta, theta)
    return phi, theta


# ---------------------------------------------------------------------------------- doubled angle
def doubled_angle(phi: ArrayLike) -> NDArray[np.float64]:
    """Axial azimuth ``φ`` → doubled-angle representation ``(cos 2φ, sin 2φ)``.

    This is the correct representation for circular statistics on axial (period-π) data: it
    removes the 0/π discontinuity. Returned along the last axis.
    """
    phi = np.asarray(phi, dtype=np.float64)
    return np.stack([np.cos(2.0 * phi), np.sin(2.0 * phi)], axis=-1)


def angle_from_doubled(cos2phi: ArrayLike, sin2phi: ArrayLike) -> NDArray[np.float64]:
    """Doubled-angle ``(cos 2φ, sin 2φ)`` → axial azimuth ``φ ∈ [0, π)``."""
    cos2phi = np.asarray(cos2phi, dtype=np.float64)
    sin2phi = np.asarray(sin2phi, dtype=np.float64)
    phi = 0.5 * np.arctan2(sin2phi, cos2phi)  # (-π/2, π/2]
    return np.mod(phi, np.pi)


# --------------------------------------------------------------------------- structure-tensor axis
def minor_eigenvector(tensor: ArrayLike) -> NDArray[np.float64]:
    """Eigenvector of the **smallest** eigenvalue of a symmetric tensor = the fibre axis.

    Accepts a single ``(D, D)`` tensor or a batch ``(..., D, D)``; returns unit eigenvectors
    of shape ``(..., D)``. Sign is fixed by making the largest-magnitude component positive
    (axial vectors are defined up to sign).
    """
    t = np.asarray(tensor, dtype=np.float64)
    t = 0.5 * (t + np.swapaxes(t, -1, -2))  # symmetrize defensively
    eigvals, eigvecs = np.linalg.eigh(t)  # ascending eigenvalues
    vec = eigvecs[..., :, 0]  # column 0 = smallest eigenvalue
    return _canonical_sign(vec)


def major_eigenvector(tensor: ArrayLike) -> NDArray[np.float64]:
    """Eigenvector of the **largest** eigenvalue (gradient axis — *not* the fibre axis)."""
    t = np.asarray(tensor, dtype=np.float64)
    t = 0.5 * (t + np.swapaxes(t, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(t)
    vec = eigvecs[..., :, -1]
    return _canonical_sign(vec)


def _canonical_sign(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fix the sign of axial vectors: largest-magnitude component made non-negative."""
    idx = np.argmax(np.abs(vec), axis=-1)
    sign = np.sign(np.take_along_axis(vec, idx[..., None], axis=-1))
    sign = np.where(sign == 0, 1.0, sign)
    return vec * sign