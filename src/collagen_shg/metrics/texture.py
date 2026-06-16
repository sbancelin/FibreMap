"""Family E — texture descriptors (GLCM / LBP / Gabor).

Texture descriptors work even when fibres are not individually resolved, and serve global
signature and classification. The GLCM gives Haralick attributes whose variation with angle is
a directional anisotropy; the LBP encodes the local binary pattern; a Gabor filter bank gives an
energy ``E(f, φ)`` whose argmax indicates orientation. Complementary to families A–C for
quasi-homogeneous tissues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

from collagen_shg.representations import conventions as cv

__all__ = [
    "GaborEnergy",
    "glcm_features",
    "lbp_histogram",
    "gabor_energy",
]

_HARALICK_PROPS = ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation")


def _quantize(image: np.ndarray, levels: int) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    q = np.floor((img - lo) / (hi - lo) * (levels - 1e-9)).astype(np.int64)
    return np.clip(q, 0, levels - 1).astype(np.uint8)


def glcm_features(
    image: np.ndarray,
    distances=(1,),
    angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    *,
    levels: int = 16,
) -> dict[str, float | np.ndarray]:
    """Gray-level co-occurrence Haralick features + directional anisotropy.

    Returns each Haralick property averaged over distances/angles, the per-angle ``contrast``
    profile, and ``anisotropy`` = ``(max − min) / (max + min)`` of contrast across angles.
    """
    q = _quantize(image, levels)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles), levels=levels,
        symmetric=True, normed=True,
    )
    out: dict[str, float | np.ndarray] = {}
    for prop in _HARALICK_PROPS:
        vals = graycoprops(glcm, prop)  # [n_distances, n_angles]
        out[prop] = float(vals.mean())
    contrast_per_angle = graycoprops(glcm, "contrast").mean(axis=0)  # over distances
    out["contrast_per_angle"] = contrast_per_angle
    cmax, cmin = float(contrast_per_angle.max()), float(contrast_per_angle.min())
    out["anisotropy"] = (cmax - cmin) / (cmax + cmin) if (cmax + cmin) > 0 else 0.0
    return out


def lbp_histogram(
    image: np.ndarray, P: int = 8, R: float = 1.0, *, method: str = "uniform"
) -> np.ndarray:
    """Normalized Local Binary Pattern histogram (rotation-invariant ``uniform`` by default)."""
    img = _quantize(image, 256)  # integer input (recommended for LBP)
    codes = local_binary_pattern(img, P, R, method=method)
    n_bins = P + 2 if method == "uniform" else int(codes.max()) + 1
    hist, _ = np.histogram(codes.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


@dataclass
class GaborEnergy:
    energy: np.ndarray  # E[f, phi]
    freqs: np.ndarray
    angles: np.ndarray
    orientation: float  # dominant fibre orientation in [0, pi)
    peak_frequency: float


def gabor_energy(
    image: np.ndarray,
    freqs=(0.05, 0.1, 0.2, 0.3),
    angles=None,
) -> GaborEnergy:
    """Gabor filter-bank energy ``E(f, φ)`` and the dominant fibre orientation.

    The argmax over the bank gives the modulation direction; the fibre orientation is that
    rotated by 90° (energy concentrates perpendicular to the fibre).
    """
    img = np.asarray(image, dtype=np.float64)
    if angles is None:
        angles = np.linspace(0.0, np.pi, 12, endpoint=False)
    freqs = np.asarray(freqs, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    energy = np.zeros((freqs.size, angles.size))
    for i, f in enumerate(freqs):
        for j, th in enumerate(angles):
            real, imag = gabor(img, frequency=f, theta=th)
            energy[i, j] = float(np.mean(real**2 + imag**2))

    fi, ai = np.unravel_index(int(np.argmax(energy)), energy.shape)
    orientation = float(cv.wrap_axial(angles[ai] + np.pi / 2.0))
    return GaborEnergy(
        energy=energy,
        freqs=freqs,
        angles=angles,
        orientation=orientation,
        peak_frequency=float(freqs[fi]),
    )
