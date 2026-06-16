"""Family D — Fourier / power spectrum (global orientation + characteristic spacing).

The 2D power spectrum gives a global view of orientation and spacing. The angular power
distribution ``A(φ)`` yields the dominant orientation and its dispersion — with the important
caveat that a fibre bundle at angle θ concentrates spectral energy **perpendicular** to it, so
the fibre orientation is the spectral peak rotated by 90°. The radial profile gives the
characteristic spacing/diameter ``Λ*`` from the peak wavenumber ``k*``. A Hann apodization is
applied to avoid edge artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collagen_shg.representations import conventions as cv

__all__ = ["PowerSpectrumResult", "power_spectrum_orientation"]


@dataclass
class PowerSpectrumResult:
    angular_distribution: np.ndarray  # A(φ) over phi_bins (spectral angle, axial)
    phi_bins: np.ndarray  # bin centres in [0, pi)
    orientation: float  # dominant FIBRE orientation in [0, pi) (spectral peak + 90°)
    spacing: float  # characteristic spacing Λ* (pixels); inf if none found
    radial_profile: np.ndarray  # azimuthally averaged power vs integer wavenumber
    k_peak: float  # peak wavenumber (FFT bins from DC)


def power_spectrum_orientation(
    image: np.ndarray, *, n_bins: int = 180, r_min: int = 2
) -> PowerSpectrumResult:
    """Angular + radial analysis of the 2D power spectrum of ``image``."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D [Y, X], got shape {img.shape}")
    img = img - img.mean()

    ny, nx = img.shape
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(img * window))) ** 2

    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    ky = yy - cy
    kx = xx - cx
    r = np.sqrt(ky**2 + kx**2)
    ang = cv.wrap_axial(np.arctan2(ky, kx))

    r_int = np.round(r).astype(int)
    r_max = int(r_int.max())
    radial = np.bincount(r_int.ravel(), weights=power.ravel(), minlength=r_max + 1)
    counts = np.bincount(r_int.ravel(), minlength=r_max + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = np.where(counts > 0, radial / counts, 0.0)

    # Characteristic spacing from the dominant radial peak (excluding DC / very low freq).
    hi = min(r_max, max(cy, cx))
    band = radial.copy()
    band[: r_min + 1] = 0.0
    band[hi + 1 :] = 0.0
    k_peak = float(np.argmax(band))
    n_side = (ny + nx) / 2.0
    spacing = float(n_side / k_peak) if k_peak >= 1 else float("inf")

    # Angular power distribution over an informative annulus (exclude DC neighborhood).
    annulus = (r >= r_min) & (r <= hi)
    bin_idx = np.minimum((ang[annulus] / np.pi * n_bins).astype(int), n_bins - 1)
    A = np.bincount(bin_idx, weights=power[annulus], minlength=n_bins)[:n_bins]
    phi_bins = (np.arange(n_bins) + 0.5) * (np.pi / n_bins)

    spectral_peak = float(phi_bins[int(np.argmax(A))])
    orientation = float(cv.wrap_axial(spectral_peak + np.pi / 2.0))

    return PowerSpectrumResult(
        angular_distribution=A,
        phi_bins=phi_bins,
        orientation=orientation,
        spacing=spacing,
        radial_profile=radial,
        k_peak=k_peak,
    )
