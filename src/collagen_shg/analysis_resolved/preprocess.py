"""Preprocessing for resolved-tissue analysis (Livrable 3).

Flat-field (shading) correction, denoising and background subtraction — applied before the
orientation field so that depth/SNR artifacts do not bias the measured organization. Denoising
must **not** bias orientation (validated by tests): isotropic Gaussian/median filters preserve
the structure-tensor orientation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

__all__ = [
    "flat_field_correct",
    "denoise",
    "subtract_background",
    "preprocess",
]


def _shading_sigma(shape: tuple[int, ...], frac: float) -> tuple[float, ...]:
    return tuple(max(2.0, dim * frac) for dim in shape)


def flat_field_correct(image: np.ndarray, *, frac: float = 0.25) -> np.ndarray:
    """Remove low-frequency multiplicative shading by dividing out a smooth illumination estimate.

    The illumination is estimated as a strong Gaussian blur (scale ``frac`` of each dimension),
    which averages out the texture and keeps the slowly varying shading.
    """
    img = np.asarray(image, dtype=np.float64)
    illum = gaussian_filter(img, _shading_sigma(img.shape, frac))
    eps = 1e-6
    return img / (illum + eps) * float(illum.mean())


def denoise(image: np.ndarray, *, method: str = "gaussian", size: float = 1.0) -> np.ndarray:
    """Isotropic denoising that preserves orientation (``gaussian`` or ``median``)."""
    img = np.asarray(image, dtype=np.float64)
    if size <= 0:
        return img
    if method == "gaussian":
        return gaussian_filter(img, size)
    if method == "median":
        return median_filter(img, size=int(max(1, round(size))))
    raise ValueError(f"unknown denoise method: {method!r}")


def subtract_background(image: np.ndarray, *, frac: float = 0.25) -> np.ndarray:
    """Subtract a smooth (out-of-focus/veil) background estimate; clip negatives to zero."""
    img = np.asarray(image, dtype=np.float64)
    background = gaussian_filter(img, _shading_sigma(img.shape, frac))
    return np.clip(img - background, 0.0, None)


def preprocess(
    image: np.ndarray,
    *,
    flat_field: bool = True,
    subtract_bg: bool = False,
    denoise_sigma: float = 0.0,
    denoise_method: str = "gaussian",
) -> np.ndarray:
    """Run the configurable preprocessing chain and return a float64 image."""
    out = np.asarray(image, dtype=np.float64)
    if flat_field:
        out = flat_field_correct(out)
    if subtract_bg:
        out = subtract_background(out)
    if denoise_sigma > 0:
        out = denoise(out, method=denoise_method, size=denoise_sigma)
    return out
