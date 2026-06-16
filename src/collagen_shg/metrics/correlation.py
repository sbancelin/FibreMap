"""Family C — orientation correlation and correlation length ξ.

The two-point orientation correlation measures how far order persists — the descriptor that
separates "locally aligned but globally isotropic" (dermis) from "globally aligned" (tendon).
In 2D the doubled-angle field is correlated; in 3D the Legendre ``P2`` of the directors is used.
The computation is efficient by FFT (Wiener–Khinchin) on a periodic field. The radial profile is
fit to ``C(r) = plateau + (1 − plateau)·exp(−r/ξ)`` where the plateau is the global order
(``S2²`` in 2D) and ``ξ`` (in voxels; convert with the voxel size) is the correlation length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["OrientationCorrelation", "orientation_correlation"]


@dataclass
class OrientationCorrelation:
    r: np.ndarray  # radial lags (voxels), 0..max_r
    C: np.ndarray  # correlation profile C(r), C(0) = 1
    xi: float  # correlation length (voxels); inf if no decay within max_r
    plateau: float  # large-r asymptote (global order; S2**2 in 2D)


def _periodic_autocorr(component: np.ndarray) -> np.ndarray:
    """Periodic autocorrelation ``<f(x) f(x+r)>`` via FFT (Wiener–Khinchin)."""
    f = np.fft.fftn(component)
    return np.fft.ifftn(f * np.conj(f)).real / component.size


def _radial_average(corr: np.ndarray, max_r: int) -> tuple[np.ndarray, np.ndarray]:
    """Radially average an FFT-autocorrelation array (origin at index 0) up to ``max_r``."""
    grids = [np.fft.fftfreq(n, d=1.0 / n) for n in corr.shape]  # signed displacements
    mesh = np.meshgrid(*grids, indexing="ij")
    rr = np.sqrt(np.sum([m**2 for m in mesh], axis=0))
    r_int = np.round(rr).astype(int)
    nbins = max_r + 1
    mask = r_int <= max_r
    sums = np.bincount(r_int[mask].ravel(), weights=corr[mask].ravel(), minlength=nbins)
    counts = np.bincount(r_int[mask].ravel(), minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(counts > 0, sums / counts, np.nan)
    return np.arange(nbins), prof[:nbins]


def _is_director(field: np.ndarray) -> bool:
    return field.ndim >= 3 and field.shape[0] == 3


def _correlation_array(field: np.ndarray, is_director: bool) -> np.ndarray:
    if is_director:
        n = np.asarray(field, dtype=np.float64)
        norm = np.linalg.norm(n, axis=0, keepdims=True)
        n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
        nx, ny, nz = n[0], n[1], n[2]
        # <(n·n')^2> = sum_ij T_ij T'_ij with T = n nᵀ (off-diagonals counted twice)
        comps_diag = [nx * nx, ny * ny, nz * nz]
        comps_off = [nx * ny, nx * nz, ny * nz]
        corr = sum(_periodic_autocorr(c) for c in comps_diag)
        corr = corr + 2.0 * sum(_periodic_autocorr(c) for c in comps_off)
        return (3.0 * corr - 1.0) / 2.0  # Legendre P2
    theta = np.asarray(field, dtype=np.float64)
    c, s = np.cos(2.0 * theta), np.sin(2.0 * theta)
    return _periodic_autocorr(c) + _periodic_autocorr(s)


def orientation_correlation(
    field: np.ndarray, max_r: int, *, is_director: bool | None = None
) -> OrientationCorrelation:
    """Orientation correlation profile and correlation length ξ.

    ``field`` is a 2D/3D θ array, or a director field ``[3, ...]`` (auto-detected; override with
    ``is_director``). ``max_r`` is the maximum lag (voxels) of the profile.
    """
    field = np.asarray(field, dtype=np.float64)
    is_dir = _is_director(field) if is_director is None else is_director
    corr = _correlation_array(field, is_dir)
    r, C = _radial_average(corr, max_r)

    # Normalize so C(0) = 1 (guards against tiny numerical drift).
    if C[0] > 0:
        C = C / C[0]

    plateau, xi = _fit_decay(r, C)
    return OrientationCorrelation(r=r, C=C, xi=xi, plateau=plateau)


def _fit_decay(r: np.ndarray, C: np.ndarray) -> tuple[float, float]:
    """Fit ``C(r) = plateau + (1 − plateau) exp(−r/ξ)``; robust to no-decay / instant-decay."""
    rr, cc = r[1:], C[1:]
    valid = np.isfinite(cc)
    rr, cc = rr[valid], cc[valid]
    if rr.size < 3:
        return float(np.nanmean(C[1:])), float("inf")

    # No appreciable decay -> effectively uniform/global order.
    if np.nanmin(cc) > 0.999:
        return float(np.nanmean(cc)), float("inf")

    def model(x, plateau, xi):
        return plateau + (1.0 - plateau) * np.exp(-x / xi)

    p0 = [float(np.clip(cc[-1], 0, 1)), max(1.0, rr[len(rr) // 4])]
    try:
        popt, _ = curve_fit(
            model, rr, cc, p0=p0, bounds=([0.0, 1e-3], [1.0, 1e6]), maxfev=10000
        )
        plateau, xi = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        plateau, xi = float(np.clip(cc[-1], 0, 1)), float("inf")
    return plateau, xi
