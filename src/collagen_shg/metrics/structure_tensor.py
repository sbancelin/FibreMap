"""Family A — structure tensor (local orientation + coherence).

The structure tensor encodes the dominant local orientation and its coherence from the
smoothed outer product of the image gradient. The gradient is computed at a noise scale
``sigma``; the outer product is integrated at a scale ``rho`` (the integration scale; sweeping
``rho`` reveals organization at different scales).

Convention (CLAUDE.md / phase0): the **fibre axis is the MINOR eigenvector** (smallest
eigenvalue) of the tensor — gradients are perpendicular to the fibre. Do not confuse it with
the gradient axis (major eigenvector). Azimuth ``φ ∈ [0, π)`` measured from +x toward +y; in
3D the director is ``(x, y, z)`` per :mod:`collagen_shg.representations.conventions`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from collagen_shg.representations import conventions as cv

__all__ = [
    "StructureTensor2DResult",
    "StructureTensor3DResult",
    "structure_tensor_2d",
    "structure_tensor_3d",
]


@dataclass
class StructureTensor2DResult:
    """2D structure-tensor field. ``orientation`` is the fibre azimuth ``φ ∈ [0, π)``."""

    orientation: np.ndarray  # [Y, X] float, fibre axis azimuth in [0, pi)
    coherence: np.ndarray  # [Y, X] float in [0, 1] (0 isotropic, 1 perfectly oriented)


@dataclass
class StructureTensor3DResult:
    """3D structure-tensor field with the director and fractional anisotropy."""

    director: np.ndarray  # [3, Z, Y, X] float, fibre axis (minor eigenvector), (x, y, z)
    fa: np.ndarray  # [Z, Y, X] float in [0, 1], fractional anisotropy
    eigenvalues: np.ndarray  # [Z, Y, X, 3] ascending eigenvalues


def structure_tensor_2d(image: np.ndarray, sigma: float, rho: float) -> StructureTensor2DResult:
    """2D structure tensor → (fibre orientation φ∈[0,π), coherence∈[0,1]) per pixel.

    ``sigma`` is the gradient (noise) scale, ``rho`` the integration scale.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D [Y, X], got shape {img.shape}")

    # Gradients (axis 0 = y, axis 1 = x).
    ix = gaussian_filter(img, sigma, order=(0, 1))
    iy = gaussian_filter(img, sigma, order=(1, 0))

    jxx = gaussian_filter(ix * ix, rho)
    jyy = gaussian_filter(iy * iy, rho)
    jxy = gaussian_filter(ix * iy, rho)

    # Orientation of the major eigenvector (gradient); fibre axis is perpendicular.
    phi_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    orientation = cv.wrap_axial(phi_grad + np.pi / 2.0)

    trace = jxx + jyy
    anis = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(trace > 0, anis / trace, 0.0)
    return StructureTensor2DResult(orientation=orientation, coherence=np.clip(coherence, 0, 1))


def structure_tensor_3d(volume: np.ndarray, sigma: float, rho: float) -> StructureTensor3DResult:
    """3D structure tensor → (director[3,Z,Y,X] = minor eigenvector, fractional anisotropy)."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D [Z, Y, X], got shape {vol.shape}")

    # Gradients along x (axis 2), y (axis 1), z (axis 0).
    gx = gaussian_filter(vol, sigma, order=(0, 0, 1))
    gy = gaussian_filter(vol, sigma, order=(0, 1, 0))
    gz = gaussian_filter(vol, sigma, order=(1, 0, 0))

    jxx = gaussian_filter(gx * gx, rho)
    jyy = gaussian_filter(gy * gy, rho)
    jzz = gaussian_filter(gz * gz, rho)
    jxy = gaussian_filter(gx * gy, rho)
    jxz = gaussian_filter(gx * gz, rho)
    jyz = gaussian_filter(gy * gz, rho)

    # Assemble the per-voxel symmetric tensor in (x, y, z) ordering.
    z, y, x = vol.shape
    j = np.empty((z, y, x, 3, 3), dtype=np.float64)
    j[..., 0, 0] = jxx
    j[..., 1, 1] = jyy
    j[..., 2, 2] = jzz
    j[..., 0, 1] = j[..., 1, 0] = jxy
    j[..., 0, 2] = j[..., 2, 0] = jxz
    j[..., 1, 2] = j[..., 2, 1] = jyz

    eigvals, eigvecs = np.linalg.eigh(j)  # ascending eigenvalues; eigvecs columns
    director = cv._canonical_sign(eigvecs[..., :, 0])  # minor eigenvector, (..., 3) in (x,y,z)
    director = np.ascontiguousarray(np.moveaxis(director, -1, 0))  # [3, Z, Y, X]

    l1, l2, l3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    mean = (l1 + l2 + l3) / 3.0
    num = np.sqrt((l1 - mean) ** 2 + (l2 - mean) ** 2 + (l3 - mean) ** 2)
    den = np.sqrt(l1**2 + l2**2 + l3**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.where(den > 0, np.sqrt(1.5) * num / den, 0.0)
    return StructureTensor3DResult(
        director=director, fa=np.clip(fa, 0, 1), eigenvalues=eigvals
    )
